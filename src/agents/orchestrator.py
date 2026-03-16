"""
Orchestrator Agent — the LangGraph StateGraph that connects all sub-agents.
FLOW:
  summarize intent:
    START → classify_intent → fetch_transcript → summarize → END

  qa intent:
    START → classify_intent → fetch_transcript → build_index
           → retrieve → grade_docs → [if irrelevant + rewrites left] → rewrite_query → retrieve
                                    → [if relevant or max rewrites] → generate → check_hallucination → END

  search intent:
    START → classify_intent → mcp_search → combine_mcp_with_rag → END
"""
from langgraph.graph import StateGraph,END 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage

from src.core.state import AgentState
from src.core.llm import LLMFactory
from src.agents.transcriptor_agent import TranscriptAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.rag_agent import AgenticRAGAgent
from src.agents.mcp_tool_agent import MCPToolAgent

from src.logger.custom_logger import logger 
import asyncio

class OrchestratorAgent:
    """
    Builds and manages the LangGraph StateGraph for the full multi-agent pipeline.

    The orchestrator:
    1. Classifies user intent (summarize / qa / search)
    2. Routes to the appropriate sub-agent pipeline
    3. Manages shared state across all agents
    """
    _INTENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classifier for a YouTube video assistant.
    Classify the user's request into EXACTLY one of these three categories:

    'summarize' — ONLY when user explicitly asks for a summary/overview of the whole video.
    Examples: "summarize", "give me a summary", "overview", "recap", "tldr"

    'qa' — when user asks ANY question or wants explanation of SPECIFIC content.
    Examples: "explain X", "what is X", "how does X work", "what did they say about X",
                "describe X", "tell me about X", "main idea", "key concept", "what does X mean"

    'search' — when user wants external/web information beyond the video.
    Examples: "search for", "find online", "latest news on", "current status of"

    IMPORTANT: "explain", "describe", "what is", "main idea" → always classify as 'qa', NOT 'summarize'.
    Only use 'summarize' if the user literally asks for a summary of the entire video.

    Respond with ONLY the single word: summarize, qa, or search."""),
        ("human", "User request: {request}")
    ])
    def __init__(self):
        # instantiate all sub agents 
        self._transcript_agent = TranscriptAgent()
        self._summary_agent = SummaryAgent()
        self._rag_agent = AgenticRAGAgent()
        self._mcp_agent = MCPToolAgent()

        # intent classification chain
        self._intent_chain = (
            self._INTENT_PROMPT
            | LLMFactory.get_grader_llm()
            | StrOutputParser()
        )

        # build the graph reuse for all requests
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # Add all nodes 
        graph.add_node("classify_intent",self._classify_intent)
        graph.add_node("fetch_transcript",self._transcript_agent.run)
        graph.add_node("summarize",self._summary_agent.run)
        graph.add_node("handle_error",self._handle_error)

        # RAG sub pipeline nodes 
        graph.add_node("build_index",self._rag_agent.build_index)
        graph.add_node("retrieve",self._rag_agent.retrieve)
        graph.add_node("grade_docs",self._rag_agent.grade_docs)
        graph.add_node("rewrite_query",self._rag_agent.rewrite_query)
        graph.add_node("generate",self._rag_agent.generate)
        graph.add_node("check_hallucination",self._rag_agent.check_hallucination)

        # mcp search node 
        graph.add_node("mcp_search",self._mcp_agent.run)
        graph.add_node("combine_results",self._combine_mcp_and_rag)

        # set entry point 
        graph.set_entry_point("classify_intent")

        # always classify intent -> fetch_transcript
        graph.add_edge("classify_intent","fetch_transcript")

        # after transcript fetch branch on intent
        graph.add_conditional_edges("fetch_transcript", # Returns "summarize" | "build_index" | "mcp_search"
                                    self._route_after_transcript,
                                    {
                                        "summarize":"summarize",
                                        "build_index":"build_index",
                                        "mcp_search":"mcp_search",
                                        "handle_error":"handle_error",
                                    })
        # summarize path 
        graph.add_edge("summarize",END)
        graph.add_edge("handle_error", END)

        # RAG path 
        graph.add_edge("build_index","retrieve")
        graph.add_edge("retrieve","grade_docs")

        # Check if retriever relevant
        graph.add_conditional_edges("grade_docs",
                                    self._rag_agent.should_rewrite, # returns rewrite | generate
                                    {"rewrite":"rewrite_query",
                                     "generate":"generate"})
        
        # after rewriting go back to retrieve 
        graph.add_edge("rewrite_query","retrieve")

        # check for hallucinations then done 
        graph.add_edge("generate","check_hallucination")
        graph.add_conditional_edges("check_hallucination",
                                    self._route_after_hallucination,
                                    {"end": END, "combine": "combine_results"})

        
        graph.add_edge("combine_results", END)

        # mcp search path 
        # After MCP search, also do RAG and combine results
        graph.add_edge("mcp_search", "build_index")
        # build_index → retrieve → grade → generate → check → combine
        # Override the normal END: after hallucination check, go to combine
        # We need a separate "combine" step for the mcp+rag results
        
        return graph.compile()
    
    async def _classify_intent(self,state: AgentState) -> dict:
        """Node: Determine what user wants"""
        question = state.get("user_question","")
        video_url = state.get("video_url","")

        if not question or question.strip() == "":
            logger.info("OrchestratorAgent: no question -> Summarize intent")
            return {"intent":"summarize",
                    "agent_trace":[f"Orchestrator: intent = summarize (no question)"],}
        
        request_text = f"Video URL: {video_url}\nUser question: {question}"
        intent_raw = await self._intent_chain.ainvoke({"request":request_text})
        intent = intent_raw.strip().lower()
        
        if intent not in ("summarize","qa","search"):
            intent = "qa"
        
        logger.info(f"OrchestratorAgent: classified intent='{intent}'")
        
        return {"intent":intent,
                "rewrite_count":0,
                "agent_trace":[f"Orchestrator: intent={intent}"]}
    
    def _route_after_transcript(self,state: AgentState) -> str:
        """Conditional edge function - returns name of the next node,
        This is not a graph node itself, so it never updates state or agent_state"""

        error = state.get("error")
        if error:
            logger.warning(f"OrchestratorAgent: transcript error, routing to END: {state.get("error")}")
            return "handle_error"  # SummaryAgent will gracefully return no-transcript message
        
        intent = state.get("intent", "summarize")
        route_map = {"summarize": "summarize", "qa": "build_index", "search": "mcp_search"}
        route = route_map.get(intent, "build_index")
        logger.info(f"OrchestratorAgent: routing to '{route}'")
        return route
    
    def _handle_error(self, state: AgentState) -> dict:
        """NEW: terminal node for pipeline errors"""
        logger.error(f"Pipeline terminated early: {state.get('error')}")
        return {"agent_trace": [f"Orchestrator: pipeline stopped — {state.get('error')}"]}
    

    def _combine_mcp_and_rag(self,state: AgentState) -> dict:
        """Node: Combine Tavily web results with RAG answer"""
        rag_answer = state.get("answer","")
        mcp_results = state.get("mcp_results",[])
        mcp_context = "\n\n".join(mcp_results) if mcp_results else ""

        if mcp_context and rag_answer:
            combined = (
                f"From the video:\n{rag_answer}\n\n"
                f"Additional web context:\n{mcp_context}"
            )
        
        elif mcp_context:
            combined =  f"Web search results:\n{mcp_context}"
        else:
            combined = rag_answer 
        
        return {"answer":combined,
                "agent_trace":["Orchestrator: combined RAG and web results"]}
    
    def _route_after_hallucination(self, state: AgentState) -> str:
        """NEW: qa ends here; search continues to combine_results"""
        intent = state.get("intent", "qa")
        return "combine" if intent == "search" else "end"

    # Public API 
    def _build_initial_state(self, video_url: str, question: str | None) -> AgentState:
        """Single source of truth for initial graph state."""
        return {
            "video_url": video_url,
            "user_question": question or "",
            "raw_transcript": None,
            "processed_transcript": None,
            "chunks": None,
            "retrieved_docs": None,
            "is_relevant": None,
            "rewrite_count": 0,
            "summary": None,
            "answer": None,
            "agent_trace": [],
            "messages": [HumanMessage(content=question or f"Summarize this video: {video_url}")],
            "mcp_results": None,
            "intent": None,
            "error": None,
        }


    # def run(self,
    #         video_url: str,
    #         question: str|None = None) -> AgentState:
    #     """
    #     Synchronous entry point. Builds initial state and invokes the graph.

    #     Args:
    #         video_url: YouTube video URL
    #         question:  Optional question (None = summary mode)

    #     Returns:
    #         Final AgentState after all nodes have run.

    #     NOTE on messages initialisation:
    #       We seed messages with a HumanMessage so the conversation history
    #       starts correctly. MessagesState's add_messages reducer will append
    #       subsequent AIMessages from agents automatically.
    #       We do NOT set "messages": [] — MessagesState handles that default.
    #     """

    #     initial_state: AgentState = {
    #         "video_url": video_url,
    #         "user_question": question or "",
    #         "raw_transcript": None,
    #         "processed_transcript":None,
    #         "chunks": None,
    #         "retrieved_docs": None,
    #         "is_relevant": None,
    #         "rewrite_count": 0,
    #         "summary": None,
    #         "answer": None,
    #         "agent_trace": [],
    #         # Seed messages with the user's request as a HumanMessage.
    #         # add_messages reducer appends further messages from agent nodes.
    #         "messages":[HumanMessage(content=question or f"Summarize this video: {video_url}")],
    #         "mcp_results": None,
    #         "intent":None,
    #         "error": None 
    #     }

    #     logger.info(f"Orchestrator: starting graph for url: {video_url[:50]}")
    #     final_state = self._graph.invoke(initial_state)
    #     logger.info(f"Orchestrator: graph complete, trace={final_state.get("agent_trace")}")
    #     return final_state 
    def run(self, video_url: str, question: str | None = None) -> AgentState:
        """Sync shim for CLI / tests only. FastAPI always uses arun()."""
        return asyncio.run(self.arun(video_url, question))

    

    async def arun(self,video_url: str,question: str | None = None) -> AgentState:
        # async entry point for fastapi
        # initial_state: AgentState = {
        #     "video_url": video_url,
        #     "user_question": question or "",
        #     "raw_transcript": None,
        #     "processed_transcript": None,
        #     "chunks": None,
        #     "retrieved_docs": None,
        #     "is_relevant": None,
        #     "rewrite_count": 0,
        #     "summary": None,
        #     "answer": None,
        #     "agent_trace": [],
        #     "messages": [HumanMessage(content=question or f"Summarize this video: {video_url}")],
        #     "mcp_results": None,
        #     "intent": None,
        #     "error": None,
        # }

        final_state = await self._graph.ainvoke(self._build_initial_state(video_url,question))
        logger.info(f"Orchestrator: graph complete, trace={final_state.get('agent_trace')}")
        return final_state
    
    def stream_run(self, video_url: str, question: str | None = None):
        """
        Generator that yields state updates as each node completes.
        Perfect for streaming progress to the UI.

        Usage:
            for chunk in orchestrator.stream_run(url, question):
                print(chunk)  # Each chunk is a partial state update
        """
        initial_state: AgentState = self._build_initial_state(video_url,question)
        # graph.stream() yields {"node_name": partial_state_update} after each node
        for chunk in self._graph.stream(initial_state):
            yield chunk
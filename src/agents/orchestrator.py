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

class OrchestratorAgent:
    """
    Builds and manages the LangGraph StateGraph for the full multi-agent pipeline.

    The orchestrator:
    1. Classifies user intent (summarize / qa / search)
    2. Routes to the appropriate sub-agent pipeline
    3. Manages shared state across all agents
    """

    _INTENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system","""Classify the user's intent intent for a YouTube video bot.
         Return ONLY one of these exact words:
         - 'summarize' - user wants summary of the video
         - 'qa' - user has specific question about the video content
         - 'search' - user wants web search for context beyond the video

         No explanation, just the single word."""),
         ("human","User request: {request}")
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
                                        "mcp_search":"mcp_search"
                                    })
        # summarize path 
        graph.add_edge("summarize",END)

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
        graph.add_edge("check_hallucination",END)

        # mcp search path 
        # After MCP search, also do RAG and combine results
        graph.add_edge("mcp_search", "build_index")
        # build_index → retrieve → grade → generate → check → combine
        # Override the normal END: after hallucination check, go to combine
        # We need a separate "combine" step for the mcp+rag results
        graph.add_edge("combine_results", END)

        return graph.compile()
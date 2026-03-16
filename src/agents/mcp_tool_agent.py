import asyncio 
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from src.core.state import AgentState
from src.core.llm import LLMFactory
from src.core.config import settings
from src.logger.custom_logger import logger

def _build_mcp_config() -> dict:
    if not settings.tavily_api_key:
        logger.warning("MCPToolAgent: TAVILY_API_KEY not set. Create a free api key at https://app.tavily.com")
        return {}
    
    return {
        "tavily":{
            "transport":"streamable_http",
            "url":f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.tavily_api_key}",
        }
    }

def _build_mcp_config_stdio() -> dict:
    if not settings.tavily_api_key:
        return {}

    return {
        "tavily": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            # env passes the API key to the npx subprocess
            "env": {"TAVILY_API_KEY": settings.tavily_api_key},
        }
    }

class MCPToolAgent:
    """
    Agent uses Tavily MCP tools (search + extract) to
    augment answers with real-time web information 
    beyond video transcript

    Called by orchestrator when intent is classified as search
    """

    _SYSTEM_PROMPT = """You are a research assistant helping to answer questions about 
    YouTube video content with supplementary web information.

    You have access to Tavily web search and extract tools:
        - Use tavily-search to find relevant current information
        - Use tavily-extract to read the full content of a specific URL
    
    Strategy:
    1. Search for information directly relevant to the question
    2. If a result looks especially useful, extract the full page
    3. Synthesise findings into a clear, concise answer
    4. Cite your sources (include URLs)

    Be concise. Focus only on what is relevant to the question.
    """

    def __init__(self) -> None:
        self._llm = LLMFactory.get_qa_llm()
        self._mcp_config = _build_mcp_config()
        self._use_http = bool(self._mcp_config)
        
        if self._mcp_config:
            transport = list(self._mcp_config.values())[0].get("transport")
            logger.info(f"MCPToolAgent: configured with transport='{transport}'")
        else:
            logger.warning("MCPToolAgent: no MCP config — set TAVILY_API_KEY")
    
    #  LangGraph Node 
    # def run(self,state: AgentState) -> dict: 
    #     question = state.get("user_question","")
        
    #     if not question: 
    #         return {"error":"No question for MCP agent.",
    #                 "agent_trace":["MCPToolAgent: No question in state"]}
        
        
    #     if not self._mcp_config:
    #         return {"mcp_results":[],
    #                 "agent_trace":["MCPToolAgent: skipped TAVILY_API_KEY not set"],
    #                 "error":"TAVILY_API_KEY not set. Get a free key at https://app.tavily.com",} 
    #     try:
    #         result = asyncio.run(self._arun(question))
    #         return {"mcp_results":[result],
    #                 "agent_trace":[f"MCPToolAgent: web search complete ({len(result):,} chars)"]}
    #     except Exception as e:
    #         logger.error(f"MCPToolAgent error: {str(e)}") 

    #         # use stdio fallback if http fails
    #         if self._use_http:
    #             logger.info("MCPToolAgent: http failed, trying stdio fallback")
    #             return self._run_with_stdio(question)
            
    #         return {"error":f"MCP search failed: {str(e)}",
    #                 "mcp_results":[],
    #                 "agent_trace":[f"MCPToolAgent: {str(e)}"]} 
    
    async def run(self, state: AgentState) -> dict:
        """Async LangGraph node — used by graph.ainvoke()."""
        question = state.get("user_question", "")

        if not self._mcp_config:
            return {"mcp_results": [],
                    "agent_trace": ["MCPToolAgent: skipped — TAVILY_API_KEY not set"]}
        try:
            result = await self._arun(question)
            return {"mcp_results": [result],
                    "agent_trace": [f"MCPToolAgent: web search complete ({len(result):,} chars)"]}
        except Exception as e:
            logger.error(f"MCPToolAgent error: {str(e)}")
            if self._use_http:
                logger.info("MCPToolAgent: HTTP failed, trying stdio fallback")
                return await self._arun_with_stdio(question)
            return {"error": f"MCP search failed: {str(e)}",
                    "mcp_results": [],
                    "agent_trace": [f"MCPToolAgent: {str(e)}"]}
        
    async def _arun(self,question: str) -> str:
        """Opens MultiServerMCPClient, gets tools, runs react agent"""

        transport = list(self._mcp_config.values())[0].get("transport")
        
        logger.info(f"MCPToolAgent: connecting to MCP server (transport={list(self._mcp_config.values())[0].get("transport")})")

        client = MultiServerMCPClient(self._mcp_config) 
            
        all_tools = await client.get_tools()
        if not all_tools: 
            return "No MCP tools available tools from Tavily server"
        
        tool_names = [tool.name for tool in all_tools]
        logger.info(f"MCPToolAgent available tools: {tool_names}")

        desired = {"tavily_search","tavily_extract"}
        tools = [t for t in all_tools if t.name in desired] or all_tools

        logger.info(f"MCPToolAgent: using tools: {[t.name for t in tools]}")

        agent = create_agent(model=self._llm,
                                tools=tools,
                                system_prompt=self._SYSTEM_PROMPT)
        
        result = await agent.ainvoke({"messages":[HumanMessage(content=question)]})

        final_message = result['messages'][-1]
        return (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

    # mcp_tool_agent.py — replace _run_with_stdio entirely
    async def _arun_with_stdio(self, question: str) -> dict:
        """Async stdio fallback when HTTP transport fails."""
        stdio_config = _build_mcp_config_stdio()
        if not stdio_config:
            return {"error": "Both HTTP and stdio transports failed",
                    "mcp_results": [],
                    "agent_trace": ["MCPToolAgent: all transports failed"]}
        try:
            original = self._mcp_config
            self._mcp_config = stdio_config
            result = await self._arun(question)
            self._mcp_config = original
            return {"mcp_results": [result],
                    "agent_trace": ["MCPToolAgent: web search via stdio fallback"]}
        except Exception as e:
            self._mcp_config = original
            return {"error": f"stdio fallback also failed: {str(e)}",
                    "mcp_results": [],
                    "agent_trace": [f"MCPToolAgent: stdio failed: {str(e)}"]}
        
    async def search(self,query: str) -> list[dict]:
        """
        Direct search — bypasses the ReAct agent and calls the
        tavily-search MCP tool directly. Useful for testing.

        Returns: list of result dicts
          [{"title": ..., "url": ..., "content": ...}, ...]
        """

        if not self._mcp_config:
            logger.warning("MCPToolAgent.search: no MCP config")
            return []
        
        client = MultiServerMCPClient(self._mcp_config) 
        
        tools = await client.get_tools()
        search_tool = next((t for t in tools if t.name == "tavily_search"), None)

        if not search_tool:
            logger.warning("MCPToolAgent.search: tavily_search tool not found")
            return []
    
        result = await search_tool.ainvoke({"query":query})
        if isinstance(result,list):
            return result 
        
        return [{"content": str(result), "url": "", "title": "Search Result"}] 
            
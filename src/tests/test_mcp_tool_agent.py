# src/tests/test_mcp_tool_agent.py

import asyncio
from src.agents.mcp_tool_agent import MCPToolAgent
from src.logger.custom_logger import logger


def test_no_question():
    """Test MCP agent when no question is provided"""
    logger.info("Test MCPToolAgent: No question")
    agent = MCPToolAgent()
    state = {
        "agent_trace":[]
    }

    result = agent.run(state)
    assert "error" in result
    print("\nResult:", result)

def test_no_api_key():
    """Test MCP agent when Tavily API key is missing"""
    logger.info("Test MCPToolAgent: No API key")

    agent = MCPToolAgent()
    agent._mcp_config = {}          # ← force empty config regardless of .env
    agent._use_http = False

    state = {"user_question": "What is LangGraph?", "agent_trace": []}
    result = agent.run(state)

    print("\nResult:", result)
    assert "mcp_results" in result
    assert result["mcp_results"] == []
    
async def test_async_search():
    """Test async MCP search execution"""

    logger.info("Test MCPToolAgent: Async search")

    agent = MCPToolAgent()

    state = {"user_question":"Latest developments in LangGraph",
             "agent_trace":[]}

    result = await agent.arun(state)

    print("\nAsync Result:")
    print(result)


async def test_direct_tool_search():
    """Test direct Tavily search tool"""

    logger.info("Test MCPToolAgent: Direct tool search")

    agent = MCPToolAgent()

    results = await agent.search("What is RAG in LLMs")

    print("\nSearch Results:")
    print(results)


if __name__ == "__main__":
    test_no_question()
    test_no_api_key()

    # run async tests
    asyncio.run(test_async_search())
    asyncio.run(test_direct_tool_search())
# YouTube Multi-Agent Bot

A production-grade multi-agent AI application that summarizes YouTube videos and answers questions
about their content using LangGraph, Agentic RAG, MCP tools, A2A protocol, FastAPI, and Streamlit.

## Architecture Overview

```
Streamlit UI  →  FastAPI Gateway  →  Orchestrator Agent (LangGraph)
                                          ├── Transcript Agent
                                          ├── Summary Agent (Groq)
                                          ├── Agentic RAG Agent (Retrieve→Grade→Rewrite→Answer)
                                          └── MCP Tool Agent (Brave Search, Fetch)
                                     ↕
                                  A2A Protocol (Agent-to-Agent communication)
```

## Tech Stack
- **LLM**: Groq (free tier) — llama-3.3-70b-versatile
- **Orchestration**: LangGraph StateGraph
- **RAG**: FAISS + HuggingFace sentence-transformers
- **MCP**: langchain-mcp-adapters (Brave Search, Fetch MCP)
- **A2A**: Custom A2A protocol implementation
- **API**: FastAPI with async streaming
- **UI**: Streamlit

## Project Structure
```
ytbot/
├── agents/
│   ├── orchestrator.py      # LangGraph StateGraph orchestrator
│   ├── transcript_agent.py  # YouTube transcript fetching
│   ├── summary_agent.py     # Groq-based summarization
│   ├── rag_agent.py         # Agentic RAG with self-correction loop
│   └── mcp_tool_agent.py    # MCP tools agent (Brave, Fetch)
├── core/
│   ├── config.py            # Pydantic Settings
│   ├── state.py             # LangGraph AgentState TypedDict
│   ├── llm.py               # Groq LLM factory
│   └── embeddings.py        # Embedding model factory
├── mcp_tools/
│   └── mcp_client.py        # MCP server connections via langchain-mcp-adapters
├── a2a/
│   ├── agent_card.py        # A2A AgentCard definitions
│   ├── task_manager.py      # A2A task lifecycle manager
│   └── a2a_server.py        # A2A server (FastAPI routes)
├── api/
│   └── main.py              # FastAPI app entry point
├── ui/
│   └── app.py               # Streamlit frontend
├── tests/
│   └── test_agents.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Quick Start
```bash
cp .env.example .env
# Fill in GROQ_API_KEY and BRAVE_API_KEY in .env
docker-compose up --build
# Streamlit: http://localhost:8501
# FastAPI docs: http://localhost:8000/docs
```

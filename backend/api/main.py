from contextlib import asynccontextmanager
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routers import summarize,qa,health 

from src.agents.orchestrator import OrchestratorAgent
from src.logger.custom_logger import logger

_orchestrator: OrchestratorAgent | None = None 

@asynccontextmanager # this manages app startup and shutfown
async def lifespan(app: FastAPI):
    """Startup: Initialize orhcestrator once. Shutdown: clean up"""
    global _orchestrator
    
    logger.info("API startup: initializing OrchestratorAgent")
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
        
    app.state.orchestrator = _orchestrator
    # this stored globally for routes
    logger.info("API startup: ready")
    yield
    logger.info("API shutdown")


app = FastAPI(title="Youtube RAG Assistant API",
              description="Multi-agent YouTube summarizer and Q&A powered by LangGraph and Groq LLM",
              version="1.0.0",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(summarize.router,prefix="/api/v1")
app.include_router(qa.router,prefix="/api/v1")
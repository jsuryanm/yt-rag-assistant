from typing import List,Optional,Annotated
import operator 
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    video_url: str 
    user_question: Optional[str]

    raw_transcript: Optional[str]
    processed_transcript: Optional[str] # cleaned transcript as a single string 

    chunks: Optional[List[str]]
    retrieved_docs: Optional[List[str]]
    is_relevant: Optional[bool] # are docs relevant 
    rewrite_count: int  # how many query rewrites happened

    summary: Optional[str] # Final video summary 
    answer: Optional[str] # Final Q&A answer

    # operator.add = append-only (each node adds its own trace lines)
    # This is separate from `messages` — it's a human-readable step log for the UI
    agent_trace: Annotated[List[str], operator.add] # appends messages

    mcp_results: Optional[List[str]] # Results from Tavily / MCP tool calls
    intent: Optional[str] # "summarize" | "qa" | "search"
    error: Optional[str] 
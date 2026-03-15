from fastapi import APIRouter,Request,HTTPException
from pydantic import BaseModel,ConfigDict
from typing import Optional,List

router = APIRouter(tags=["summarize"])

class SummarizeRequest(BaseModel):
    # defines request in JSON format
    model_config = ConfigDict(arbitrary_types_allowed=True)
    video_url: str 

class SummarizeResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    summary: str 
    agent_trace: List[str]
    error: Optional[str]  = None

"""Response object contains headers,body,cookies,app reference,middleware data
app.state is a place we store global objects like db connection, llm client, orchestrator,etc
we create in orchestrator in main.py"""

@router.post("/summarize",response_model=SummarizeResponse)
async def summarize_video(body: SummarizeRequest,
                          request: Request):
    orchestrator = request.app.state.orchestrator

    try:
        state = await orchestrator.arun(video_url=body.video_url)
        
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
    if state.get("error") and not state.get("summary"):
        raise HTTPException(status_code=422,detail=state['error'])
    
    return SummarizeResponse(summary=state.get("summary") or "",
                             agent_trace=state.get("agent_trace") or [],
                             error=state.get("error"))


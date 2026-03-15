from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel,ConfigDict
from typing import Optional,List 

router = APIRouter(tags=["qa"])

class QARequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    video_url: str 
    question: str 

class QAResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    answer: str 
    agent_trace: List[str]
    error: Optional[str] = None

@router.post("/qa",response_model=QAResponse)
async def answer_question(body: QARequest,
                          request: Request):
    
    orchestrator = request.app.state.orchestrator
    
    if not body.question.strip():
        raise HTTPException(status_code=400,detail="question must not be empty")
    
    try:
        state = await orchestrator.arun(video_url=body.video_url,
                                        question=body.question)
        
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
    if state.get("error") and not state.get("answer"):
        raise HTTPException(status_code=422,detail=state['error'])
    
    return QAResponse(answer=state.get("answer") or "",
                      agent_trace=state.get("agent_trace") or [],
                      error=state.get("error"))
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.logger.custom_logger import logger
from src.core.state import AgentState
from src.core.llm import LLMFactory
from src.exceptions.custom_exception import YtException

import sys 


class SummaryAgent:
    """
    Generates video summaries using groq llm
    Implement rag chain 
    """

    _SYSTEM_PROMPT = """You are an expert at summarizing YouTube video transcripts.
    Your task is to produce a clear, concise summary that captures:
    1. The main topic and purpose of the video
    2. Key points and insights discussed
    3. Any conclusions or takeaways

    Rules:
    - Write 3-5 sentences maximum
    - Do NOT mention timestamps or formatting artifacts
    - Write in third person (e.g. "The video explains...")
    - Focus on substance, not delivery style"""

    _HUMAN_PROMPT = """Please summarize the following YouTube video transcript:
    {transcript}

    Summary:"""

    def __init__(self):
        self.llm = LLMFactory.get_summary_llm()
        self._chain = self._build_chain()

    
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",self._SYSTEM_PROMPT),
            ("human",self._HUMAN_PROMPT)
        ])

        chain = prompt | self.llm | StrOutputParser()

        return chain 
    
    def run(self,state: AgentState) -> dict:
        """
        LangGraph node Expects state to have 'processed_transcript'
        Returns state dict with 'summary'
        """

        logger.info("SummaryAgent: Generating summary")
        
        transcript = state.get("processed_transcript")
        
        if not transcript:
            return {"error":"No transcript available for summarization",
                    "agent_trace": ["SummaryAgent: no transcript in state"]}
        try:
            truncated = transcript[:6000] if len(transcript) > 6000 else transcript

            summary = self._chain.invoke({"transcript":truncated})
            logger.info(f"SummaryAgent: summary generated ({len(summary)} chars)")
            
            return {"summary":summary,
                    "agent_trace":f"summary generated ({len(summary)}) chunks"}
        
        except Exception as e:  
            logger.error(f"Summarization Agent error:{str(e)}")

            return {"error": f"Summary generation failed: {str(e)}",
                    "agent_trace": [f"SummaryAgent: error — {str(e)}"]}
    
    async def arun(self, transcript: str) -> str:
        """Async version for direct calls from FastAPI endpoints."""
        truncated = transcript[:6000] if len(transcript) > 6000 else transcript
        return await self._chain.ainvoke({"transcript": truncated})

from functools import lru_cache
from langchain_groq import ChatGroq
from src.core.config import settings
from src.logger.custom_logger import logger 

class LLMFactory:

    @staticmethod
    @lru_cache(maxsize=4)
    # lru_cache is a caching decorator in Python that stores function results 
    # so repeated calls with the same inputs don't recompute.
    # maxsize is the max no of cached function calls 
    def get_llm(model: str | None = None,
                temperature: float | None=None,
                max_tokens: int | None=None) -> ChatGroq:
        
        """Returns cached groq llm"""
        
        _model = model or settings.groq_model 
        _temp = temperature if temperature is not None else settings.groq_temp
        _max = max_tokens or settings.groq_max_tokens

        logger.info(f"Creating ChatGroq model: model={_model}, temp={_temp}")

        return ChatGroq(api_key=settings.groq_api_key,
                        model=_model,
                        temperature=_temp,
                        max_tokens=_max)
    
    @staticmethod
    def get_grader_llm() -> ChatGroq:
        """
        LLM for RAG grading — uses low temperature for consistent
        yes/no relevance judgements.
        """
        return LLMFactory.get_llm(temperature=0.0,max_tokens=64)
    
    @staticmethod
    def get_summary_llm() -> ChatGroq:
        """Slightly more creative for summaries"""
        return LLMFactory.get_llm(temperature=0.1,max_tokens=1024)
    
    @staticmethod
    def get_qa_llm() -> ChatGroq:
        """Deterministic for Q&A answers"""
        return LLMFactory.get_llm(temperature=0.0,max_tokens=2048)
    
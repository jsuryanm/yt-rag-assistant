from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.config import settings 
from src.logger.custom_logger import logger

class EmbeddingFactory:

    @staticmethod
    @lru_cache(maxsize=2)
    def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
        """Returns cached hf embeddings"""
        _model = model_name or settings.embeddings_model 
        logger.info(f"Loading embeddings model: {_model}")
        return HuggingFaceEmbeddings(model_name=_model)
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8",
                                      case_sensitive=False,
                                      extra="ignore")
    
    groq_api_key: str = Field(...,description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile",
                            description="Primary Groq model id")
    groq_temp: float = Field(default=0.0)
    groq_max_tokens: int = Field(default=2048)

    embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2",
                                  description="HF sentence transformer model name")
    
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    rag_top_k: int = Field(default=3)
    max_rewrite_attempts: int = Field(default=2)

    tavily_api_key: str = Field(default="", description="Tavily Search API key")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    streamlit_api_url: str = Field(default="http://localhost:8000")

if __name__ == "__main__":
    settings = Settings()
    
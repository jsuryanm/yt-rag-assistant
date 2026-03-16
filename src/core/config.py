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

    embeddings_model: str = Field(default="all-MiniLM-L6-v2",
                                  description="HF sentence transformer model name")
    
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    rag_top_k: int = Field(default=3)
    max_rewrite_attempts: int = Field(default=2)

    a2a_base_url: str = Field(default="",description="Public base URL for A2A agent card (https://myapp.com)")

    tavily_api_key: str = Field(default="", description="Tavily Search API key")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    streamlit_api_url: str = Field(default="http://localhost:8000")

    # RAGAS Evaluation
    openai_api_key: str = Field(default="",description="OpenAI API key used by RAGAS judge LLM only")
    openai_judge_model: str = Field(default="gpt-4o-mini",description="OpenAI model used as RAGAS judge")
    openai_embedding_model: str = Field(default="text-embedding-3-small",
                                        description="OpenAI embeddings model for RAGAS answer_relevancy metric") 
    
    dagshub_repo_owner: str = Field(default="",description="DagsHub username")
    dagshub_repo_name: str = Field(default="",description="Dagshub repo name")
    dagshub_token: str = Field(default="",description="Dagshub personal access token")

    mlflow_environment_name: str = Field(default="youtube-rag-ragas-eval",description="MLFlow experiment-name")

settings = Settings()
    
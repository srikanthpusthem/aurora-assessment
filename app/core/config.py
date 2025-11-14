from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # HuggingFace Configuration
    hf_api_token: str
    
    # Groq Configuration
    groq_api_key: str
    
    # Messages API Configuration
    messages_api_url: str = "https://november7-730026606190.europe-west1.run.app/messages"
    
    # Model Configuration
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    groq_model: str = "llama3-8b-instruct"
    
    # Retrieval Configuration
    top_k: int = 5
    
    # Server Configuration
    port: int = 8000
    
    # Logging Configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_file: Optional[str] = "logs/aurora_qa.log"  # Path to log file (empty = console only)
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB per log file
    log_backup_count: int = 5  # Keep 5 backup files
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


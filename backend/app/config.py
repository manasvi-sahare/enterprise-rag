from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    qdrant_url: str = "http://localhost:6333"

    class Config:
        env_file = ".env"

settings = Settings()

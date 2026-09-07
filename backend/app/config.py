from pydantic_settings import BaseSettings
from slowapi import Limiter # type: ignore
from slowapi.util import get_remote_address # type: ignore

class Settings(BaseSettings):
    groq_api_key: str = ""
    sarvam_api_key: str = ""
    redis_url: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()
limiter = Limiter(key_func=get_remote_address)
GENERATION_BACKEND = "groq"

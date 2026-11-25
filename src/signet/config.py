from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

OPENAI_MODEL: str = "gpt-4o-mini"
QDRANT_URL: str = "http://localhost:6333"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr # obligatoire: plante au démarrage si absente
    openai_model: str = OPENAI_MODEL
    qdrant_url: str = QDRANT_URL

@lru_cache
def get_settings() -> Settings:
    return Settings()


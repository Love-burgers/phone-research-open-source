"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from .model_credentials import current_model_api_key


load_dotenv(Path.cwd() / ".env")

APP_NAME = "多轮选机助手"


class Settings(BaseModel):
    """Typed runtime settings. Secret values are redacted in representations."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    fetch_timeout_seconds: float = Field(default=12, gt=0)
    max_page_bytes: int = Field(default=1500000, gt=0)
    max_page_characters: int = Field(default=120000, gt=0)
    max_redirects: int = Field(default=3, ge=0)
    model_api_url: str = "https://api.deepseek.com/chat/completions"
    model_name: str = "deepseek-chat"
    model_provider: str | None = None
    model_timeout_seconds: float = Field(default=40, gt=0)
    model_max_tokens: int = Field(default=1800, gt=0)
    model_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_local_cache_dir: str = "runs/model-cache"
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str = ""
    embedding_timeout_seconds: float = Field(default=40, gt=0)
    chat_knowledge_index_path: str | None = None
    chat_retrieval_method: str | None = None
    chat_vector_min_score: float = Field(default=.35, ge=-1, le=1)
    chat_agent_enabled: bool = True
    chat_context_characters: int = Field(default=64000, ge=8000, le=128000)
    chat_turn_token_budget: int = Field(default=64000, ge=8000, le=128000)
    chat_turn_tool_calls: int = Field(default=18, ge=1, le=18)
    chat_turn_model_calls: int = Field(default=10, ge=2, le=12)
    chat_turn_search_calls: int = Field(default=6, ge=0, le=10)
    chat_turn_fetch_calls: int = Field(default=8, ge=0, le=12)
    chat_history_messages: int = Field(default=24, ge=4, le=60)
    chat_history_characters: int = Field(default=12000, ge=1400, le=24000)
    chat_storage_path: str = "runs/chat-sessions.sqlite3"
    chat_access_token: SecretStr | None = None
    chat_guard_path: str | None = None
    chat_rate_limit_per_minute: int = Field(default=6, ge=0)
    chat_daily_turn_limit: int = Field(default=100, ge=0)
    chat_daily_model_calls: int = Field(default=200, ge=0)
    chat_daily_token_limit: int = Field(default=600000, ge=0)


def get_settings() -> Settings:
    """Read once per consumer initialization; tests/deployments can change env."""
    values = {name: os.environ[name.upper()] for name in Settings.model_fields
              if os.environ.get(name.upper()) not in (None, "")}
    return Settings.model_validate(values)


# Existing provider adapters expose these constants as their injection seam.
# Keep that public API while all environment parsing lives here.
_initial = get_settings()
FETCH_TIMEOUT_SECONDS = _initial.fetch_timeout_seconds
MAX_PAGE_BYTES = _initial.max_page_bytes
MAX_PAGE_CHARACTERS = _initial.max_page_characters
MAX_REDIRECTS = _initial.max_redirects
MODEL_API_URL = _initial.model_api_url
MODEL_NAME = _initial.model_name
MODEL_TIMEOUT_SECONDS = _initial.model_timeout_seconds
MODEL_MAX_TOKENS = _initial.model_max_tokens
EMBEDDING_TIMEOUT_SECONDS = _initial.embedding_timeout_seconds


def get_model_api_key() -> str | None:
    """Read the key at call time so tests and deployments can configure it safely."""
    override=current_model_api_key()
    if override is not None:
        return override.get_secret_value()
    settings = get_settings()
    secret = settings.model_api_key or settings.deepseek_api_key
    return secret.get_secret_value() if secret else None

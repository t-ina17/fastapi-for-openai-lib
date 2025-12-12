from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_prefix='APP_', case_sensitive=False)

    # Public API settings
    api_prefix: str = "/v1"
    api_key: Optional[str] = None  # If set, require Bearer token

    # Model/backend settings
    model_id: str = "gpt-oss-20b"
    backend: str = "echo"  # echo | transformers | vllm_proxy

    # vLLM proxy settings (if backend=vllm_proxy)
    vllm_base_url: Optional[str] = None  # e.g., http://localhost:8001/v1
    vllm_api_key: Optional[str] = None

    # Generation defaults
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0


settings = Settings()

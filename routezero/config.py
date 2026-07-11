from pydantic_settings import BaseSettings
from enum import Enum
import functools
import os


class RouteTarget(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class CacheStatus(Enum):
    HIT = "hit"
    MISS = "miss"


class TaskType(Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZE = "summarize"
    NER = "ner"
    DEBUG = "debug"
    REASONING = "reasoning"
    CODEGEN = "codegen"

class Settings(BaseSettings):
    # Cache
    cache_similarity_threshold: float = 0.92
    chromadb_path: str = ".chromadb"
    embedding_model: str = os.environ.get("EMBEDDING_MODEL_PATH", "./model_cache/all-MiniLM-L6-v2")

    # Router — Semantic + Heuristic
    router_local_threshold: float = 0.4
    router_heuristic_weight: float = 0.3

    # Local model (Qwen 3.x MoE via vLLM/ROCm)
    local_model_name: str = "Qwen/Qwen3-30B-A3B"
    local_model_endpoint: str = "http://localhost:8000/v1"
    local_max_tokens: int = 1024
    local_timeout_s: float = 120.0

    # Remote model (Fireworks AI)
    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    allowed_models: str = ""  # comma-separated, read from ALLOWED_MODELS env var
    fireworks_model_id: str = "accounts/fireworks/models/llama-v3p1-405b-instruct"
    remote_timeout_s: float = 60.0

    # Verification
    verification_task_types: list[str] = ["json", "code", "reasoning"]

    # Telemetry
    sqlite_path: str = "metrics.db"

    # HuggingFace
    hf_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@functools.lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # Set HF_TOKEN environment variable so sentence-transformers can use it
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    return settings

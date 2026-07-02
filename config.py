from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None

load_dotenv()

@dataclass
class Config:
    cache_enabled: bool
    local_rocm_url: str
    fireworks_url: str
    openai_api_key: Optional[str]
    route_cost_threshold: float
    cache_similarity_threshold: float
    chroma_collection_name: str
    chroma_persist_directory: str


def load_config() -> Config:
    """Load application configuration from environment variables."""
    return Config(
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        local_rocm_url=os.getenv("LOCAL_ROCM_URL", "http://localhost:8000"),
        fireworks_url=os.getenv("FIREWORKS_URL", "https://api.fireworks.ai/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        route_cost_threshold=float(os.getenv("ROUTE_COST_THRESHOLD", "0.5")),
        cache_similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.8")),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "routezero_cache"),
        chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./.chromadb"),
    )

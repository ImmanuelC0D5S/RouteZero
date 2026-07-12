import os

from routezero.config import Settings, get_settings, RouteTarget, CacheStatus, TaskType

def test_settings_defaults():
    # Temporarily unset env vars that override defaults so we test the actual defaults
    env_keys = [
        "CACHE_SIMILARITY_THRESHOLD", "CHROMADB_PATH", "EMBEDDING_MODEL",
        "ROUTER_LOCAL_THRESHOLD", "ROUTER_HEURISTIC_WEIGHT",
        "LOCAL_MODEL_PATH", "LOCAL_MAX_TOKENS",
        "FIREWORKS_API_KEY", "FIREWORKS_BASE_URL",
        "HF_TOKEN",
    ]
    saved = {k: os.environ.pop(k, None) for k in env_keys}
    try:
        settings = Settings(_env_file=None)  # skip .env file loading
        assert settings.cache_similarity_threshold == 0.92
        assert settings.chromadb_path == ".chromadb"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.router_local_threshold == 0.4
        assert settings.router_heuristic_weight == 0.3
        assert settings.local_model_path == "/app/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        assert settings.fireworks_base_url == "https://api.fireworks.ai/inference/v1"
        assert settings.local_max_tokens == 1024
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

def test_get_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

def test_enums():
    assert RouteTarget.LOCAL.value == "local"
    assert RouteTarget.REMOTE.value == "remote"
    assert CacheStatus.HIT.value == "hit"
    assert CacheStatus.MISS.value == "miss"
    assert TaskType.FACTUAL.value == "factual"
    assert TaskType.CODEGEN.value == "codegen"
    assert len(TaskType) == 8

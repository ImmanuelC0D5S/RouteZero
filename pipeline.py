from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import load_config
from cache import ChromaCache
from llm_clients import LocalQwenClient, RemoteFireworksClient
from router import RouteLLMController, create_router_controller
from verifier import verify_response
from metrics import MetricsTracker


@dataclass
class PipelineResult:
    prompt: str
    response: str
    route: str
    verified: bool
    cache_hit: bool
    metadata: Dict[str, Any]


def run_pipeline(prompt: str) -> dict:
    """Run the adaptive routing pipeline for a single prompt.

    Steps:
      1. Load configuration and initialize supporting systems.
      2. Check the semantic cache for an existing response.
      3. Use RouteLLM to decide whether to call the local ROCm Qwen model or Fireworks AI.
      4. Call the chosen model client.
      5. Verify the candidate response.
      6. If verification fails, fallback to the alternate model.
      7. Store the result in the cache and return the result payload.
    """
    config = load_config()
    metrics = MetricsTracker()
    cache_client = ChromaCache(
        collection_name=config.chroma_collection_name,
        persist_directory=config.chroma_persist_directory,
    )

    # 1. Cache lookup
    cached = None
    if config.cache_enabled:
        cached = cache_client.get(prompt)
    if cached is not None:
        metrics.record_cache_hit()
        return {
            "prompt": prompt,
            "response": cached.get("response", ""),
            "route": cached.get("route", "cache"),
            "verified": True,
            "cache_hit": True,
            "metadata": cached.get("metadata", {}),
        }

    # 2. Route decision
    router = create_router_controller(config)
    route = router.decide_route(prompt)

    # 3. Model invocation
    local_client = LocalQwenClient(endpoint_url=config.local_rocm_url)
    remote_client = RemoteFireworksClient(endpoint_url=config.fireworks_url, api_key=config.openai_api_key)

    if route == "local":
        candidate = local_client.generate(prompt=prompt)
    else:
        candidate = remote_client.generate(prompt=prompt)

    response_text = candidate.get("text", "")

    # 4. Verification
    verified = verify_response(response_text)

    # 5. Fallback if local verification failed
    if not verified and route == "local":
        fallback_candidate = remote_client.generate(prompt=prompt)
        response_text = fallback_candidate.get("text", "")
        verified = verify_response(response_text)
        route = "fallback_remote"

    payload = {
        "prompt": prompt,
        "response": response_text,
        "route": route,
        "verified": verified,
        "cache_hit": False,
        "metadata": {
            "route_decision": route,
            "verified": verified,
        },
    }

    # 6. Cache write
    if config.cache_enabled:
        cache_client.set(prompt, payload)

    metrics.record_route(route_name=route, cost=0.0, tokens=0)
    return payload

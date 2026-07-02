from __future__ import annotations
from typing import List

from pipeline import run_pipeline
from config import load_config
from metrics import MetricsTracker


def run_benchmark(prompts: List[str], cache_enabled: bool) -> dict:
    """Run a benchmark over a list of prompts with cache on or off."""
    config = load_config()
    config.cache_enabled = cache_enabled
    metrics = MetricsTracker()
    results = []

    for prompt in prompts:
        result = run_pipeline(prompt)
        results.append(result)
        if result.get("cache_hit"):
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()
    return {
        "cache_enabled": cache_enabled,
        "prompt_count": len(prompts),
        "results": results,
        "metrics": {
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
        },
    }


def main() -> None:
    prompts = [
        "Summarize the latest AMD ROCm release for a software engineer.",
        "Generate a short Python script that uses OpenAI-compatible inference.",
        "What are the advantages of cost-aware LLM routing?",
    ]

    on_results = run_benchmark(prompts, cache_enabled=True)
    off_results = run_benchmark(prompts, cache_enabled=False)
    print("Benchmark results")
    print("Cache ON:", on_results["metrics"])
    print("Cache OFF:", off_results["metrics"])


if __name__ == "__main__":
    main()

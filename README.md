# 🚦 Adaptive Cost-Aware LLM Router
**Submission for AMD Developer Hackathon Act II — Track 1**

A minimal, token-efficient hybrid routing pipeline that balances cost and quality by intelligently distributing workloads between local and remote Large Language Models.

Instead of hardcoding brittle routing heuristics, this project uses [RouteLLM](https://github.com/lm-sys/RouteLLM) to dynamically route user prompts. Simple queries are handled by a local **Qwen model running on AMD ROCm**, while complex queries are routed to a remote **Fireworks AI** endpoint.

## Project Structure

- `routezero/config.py` — centralized configuration, thresholds, and endpoint settings
- `routezero/pipeline.py` — core orchestration for cache lookup, routing, model call, verification, and fallback
- `routezero/llm_clients.py` — wrappers for local ROCm Qwen and remote Fireworks AI OpenAI-compatible calls
- `routezero/router.py` — RouteLLM controller stub and routing logic
- `routezero/cache.py` — ChromaDB semantic cache setup, get, and set methods
- `routezero/verifier.py` — verification utilities for response consistency and format validation
- `routezero/metrics.py` — metrics tracker and export helpers
- `scripts/benchmark.py` — CLI benchmark script for comparing cache on/off performance
- `requirements.txt` — project dependencies
- `.env.example` — placeholder environment variables for API keys and endpoints

## Getting Started

1. Copy `.env.example` to `.env`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the benchmark script:
   ```bash
   python benchmark.py
   ```

## Notes

- The current codebase is a scaffold with stubbed client and cache behavior.
- Replace placeholder local and remote client logic with real ROCm and Fireworks API calls.
- The RouteLLM controller currently uses a simple prompt-length rule and can be swapped for real routing logic later.
# Project RouteZero

## Stack
- **Language:** Python 3.12+
- **Frontend / UI:** Streamlit (for the interactive demo & cost-savings dashboard)
- **Hardware Target:** AMD Developer Cloud (ROCm), AMD Instinct MI300X (192GB VRAM)
- **Core Router:** RouteLLM (`lm-sys/RouteLLM`) using Matrix Factorization / Cost Thresholding
- **Local Model (Free):** Qwen 3.5 35B-A3B (via local OpenAI-compatible endpoint)
- **Remote Model (Paid):** Fireworks AI API
- **Semantic Cache:** ChromaDB (PersistentClient) + `sentence-transformers`
- **Environment Management:** `python-dotenv`

## Project structure
- `app.py` — Streamlit frontend entrypoint (UI & visualization).
- `routezero/config.py` — Centralized settings, thresholds, cache flags, and API keys.
- `routezero/pipeline.py` — Core orchestration: Cache -> Router -> Model -> Verify -> Fallback.
- `routezero/llm_clients.py` — Wrappers for Local (ROCm Qwen) and Remote (Fireworks) calls.
- `routezero/router.py` — RouteLLM Controller initialization and context-aware routing logic.
- `routezero/cache.py` — ChromaDB semantic cache setup, get, and set methods.
- `routezero/verifier.py` — Verification utilities (self-consistency, format checks) and local-retry logic.
- `routezero/metrics.py` — Tracks cache hits, route decisions, token usage, and cost savings.
- `scripts/benchmark.py` — CLI script to run automated benchmark suites.
- `docs/` — Architecture decisions, benchmark results, and hackathon documentation.

## Commands
- `pip install -r requirements.txt` — Install project dependencies.
- `python scripts/benchmark.py` — Run the CLI benchmark comparing Cache ON vs Cache OFF.
- `streamlit run app.py` — Launch the frontend UI for the demo.

## Code standards
- **Type Hinting:** Enforce strict Python typing (`from typing import ...`) on all function signatures.
- **Dependency Management:** Keep dependencies light (`pip`) to ensure easy setup on AMD Cloud instances.
- **Fail-Safe Routing:** Router must never crash on context limits; always check token estimation before hitting the local MI300X instance.
- **Metrics-Driven:** All pipeline executions must log to the `MetricsTracker` to populate the Streamlit dashboard.
- **API Wrappers:** Abstract all LLM calls behind `llm_clients.py` to allow easy swapping of Fireworks or Qwen.

## AI & Routing Patterns
- **Remote API Firewall:** Treat the remote API as a last resort to minimize cost metrics.
- **Multi-Pass Local Retry:** If local verification fails, use the free local compute to retry with a stricter prompt before falling back to Remote.
- **Pre-Compute Token Compression:** For massive contexts routed to Remote, use the local MI300X to extract/compress the prompt first to save Remote API token costs.
- **Fuzzy Semantic Caching:** Use cosine similarity in ChromaDB (threshold ~0.80) to intercept similar queries, saving 100% of compute cost on cache hits.
- **Context Awareness:** Enforce soft threshold routing for prompts > 80k-100k tokens to prevent local model degradation ("needle in a haystack" failures).

## Workflow
1. For testing routing thresholds, modify `.env` values, do not hardcode.
2. Backend runs `python scripts/benchmark.py` to ensure core metrics haven't regressed.
3. Frontend teammate imports `run_pipeline` from `routezero.pipeline` to wire up the UI.
4. Keep the judges in mind: Every architectural change must clearly serve the goal of "Cost & Compute Awareness."
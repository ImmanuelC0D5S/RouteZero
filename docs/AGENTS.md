# Agents

RouteZero uses two layers of agents:

## OpenCode AI Agents (`.opencode/agents/`)

Custom opencode agents registered for this project. Select from the agent
menu in opencode when working on RouteZero.

- **orchestrator** — Coordinates multi-step development tasks across the
  codebase. Breaks down work, delegates to subagents, and synthesizes
  results.
- **db-architect** — ChromaDB and semantic cache schema design.
- **code-reviewer** — Code review for correctness, type safety, and
  conventions.
- **test-writer** — Test generation with pytest.
- **ai-pipeline-engineer** — Core routing pipeline, model clients, and
  fallback logic.

## Pipeline Agents (`routezero/`)

The production pipeline uses internal agent abstractions within the
`routezero/` package:

- `routezero/router.py` — RouteLLM-based controller that decides local vs.
  remote based on token estimation and cost thresholds.
- `routezero/llm_clients.py` — Model clients for local (Qwen on ROCm) and
  remote (Fireworks AI) inference.
- `routezero/cache.py` — ChromaDB semantic cache wrapper.
- `routezero/verifier.py` — Response validation and quality checks.
- `routezero/pipeline.py` — Orchestrates the full inference pipeline.
- `routezero/conversation.py` — Windowed session history management.
- `routezero/metrics.py` — Performance and cost tracking.

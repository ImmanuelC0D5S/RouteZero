# Agents

This document describes the agent abstractions used by RouteZero.

- `router.py` decides whether to use local or remote models.
- `local.py` wraps ROCm Qwen inference.
- `remote.py` wraps Fireworks/OpenAI-compatible inference.
- `verifier.py` validates candidate outputs.
- `memory.py` handles memory retrieval and retrieval augmentation.

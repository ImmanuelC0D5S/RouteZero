from __future__ import annotations
from typing import Dict, Any


class LocalAgent:
    def __init__(self, endpoint_url: str) -> None:
        self.endpoint_url = endpoint_url

    def infer(self, prompt: str) -> Dict[str, Any]:
        """Perform a local ROCm Qwen inference call."""
        return {"text": "[local placeholder]"}

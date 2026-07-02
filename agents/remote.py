from __future__ import annotations
from typing import Dict, Any, Optional


class RemoteAgent:
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def infer(self, prompt: str) -> Dict[str, Any]:
        """Perform a remote Fireworks AI OpenAI-compatible inference call."""
        return {"text": "[remote placeholder]"}

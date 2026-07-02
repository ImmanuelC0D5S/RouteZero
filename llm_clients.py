from __future__ import annotations
from typing import Any, Dict, Optional


class OpenAICompatibleClient:
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate a text response from the OpenAI-compatible endpoint.

        This is a placeholder implementation. Swap in a real HTTP client
        or SDK call for ROCm Qwen and Fireworks AI later.
        """
        return {
            "text": "",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "route": "unknown",
        }


class LocalQwenClient(OpenAICompatibleClient):
    def __init__(self, endpoint_url: str) -> None:
        super().__init__(endpoint_url=endpoint_url, api_key=None)

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        # Placeholder for local ROCm Qwen inference.
        return {
            "text": "[local qwen placeholder response]",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "route": "local",
        }


class RemoteFireworksClient(OpenAICompatibleClient):
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None) -> None:
        super().__init__(endpoint_url=endpoint_url, api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        # Placeholder for Fireworks AI OpenAI-compatible call.
        return {
            "text": "[remote fireworks placeholder response]",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "route": "remote",
        }

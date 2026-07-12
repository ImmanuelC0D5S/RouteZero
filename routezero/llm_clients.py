from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

from routezero.config import Settings


class LLMClient(Protocol):
    """Protocol for all LLM clients."""
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str


class LocalQwenClient:
    """Client for local Qwen2.5 GGUF model running via llama.cpp on CPU."""

    def __init__(self, settings: Settings) -> None:
        self.model_path = settings.local_model_path
        self.max_tokens = settings.local_max_tokens
        # lazy import so the module can be imported without llama-cpp-python
        from llama_cpp import Llama
        self._model = Llama(
            model_path=self.model_path,
            n_ctx=32768,
            n_threads=None,           # auto-detect CPU cores
            verbose=False,
        )

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start = time.monotonic()

        response: dict = await asyncio.to_thread(
            self._model.create_chat_completion,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        logging.debug("RAW LOCAL RESPONSE: %s", response)

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"Local model returned no choices. Full response: {response}"
            )

        choice = choices[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})
        return LLMResponse(
            text=message.get("content") or "",
            tokens_used=usage.get("total_tokens", 0),
            latency_ms=round(latency_ms, 2),
            model_name=self.model_path,
        )

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(
                self._model.create_chat_completion,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False


class FireworksRemoteClient:
    """Client for Fireworks AI remote API.
    
    Reads configuration from environment variables at runtime:
      - FIREWORKS_BASE_URL   (default: https://api.fireworks.ai/inference/v1)
      - FIREWORKS_API_KEY    (required)
      - ALLOWED_MODELS       (comma-separated, first value is the model ID)
    """
    
    def __init__(self, settings: Settings) -> None:
        self.timeout = settings.remote_timeout_s
        base_url = os.environ.get(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1",
        )
        api_key = os.environ["FIREWORKS_API_KEY"]
        allowed = os.environ["ALLOWED_MODELS"]
        self.model = allowed.split(",")[0].strip()
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        last_exc = None
        for attempt in range(3):
            try:
                start = time.monotonic()
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                    **kwargs,
                )
                latency_ms = (time.monotonic() - start) * 1000

                if not response.choices:
                    raise RuntimeError(
                        f"Remote model returned no choices. Full response: {response}"
                    )

                choice = response.choices[0]
                usage = response.usage
                return LLMResponse(
                    text=choice.message.content or "",
                    tokens_used=usage.total_tokens if usage else 0,
                    latency_ms=round(latency_ms, 2),
                    model_name=self.model,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]
    
    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
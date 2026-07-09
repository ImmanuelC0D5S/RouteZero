from __future__ import annotations
import asyncio
import logging
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
    """Client for local Qwen 3.x MoE running on ROCm via vLLM."""
    
    def __init__(self, settings: Settings) -> None:
        self.model = settings.local_model_name
        self.max_tokens = settings.local_max_tokens
        self.timeout = settings.local_timeout_s
        self._client = AsyncOpenAI(
            base_url=settings.local_model_endpoint,
            api_key="not-needed",
        )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **kwargs,
        )
        latency_ms = (time.monotonic() - start) * 1000

        logging.debug("RAW LOCAL RESPONSE: %s", response)

        if not response.choices:
            raise RuntimeError(
                f"Local model returned no choices. Full response: {response}"
            )

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            tokens_used=usage.total_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
            model_name=self.model,
        )
    
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


class FireworksRemoteClient:
    """Client for Fireworks AI remote API."""
    
    def __init__(self, settings: Settings) -> None:
        self.model = settings.fireworks_model_id
        self.timeout = settings.remote_timeout_s
        self._client = AsyncOpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=settings.fireworks_api_key,
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
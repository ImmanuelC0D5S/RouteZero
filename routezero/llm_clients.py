from __future__ import annotations
import asyncio
import logging
import time
import os
from dataclasses import dataclass
from typing import Protocol, List

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

def get_allowed_models(settings: Settings) -> List[str]:
    """Helper to parse the ALLOWED_MODELS env var."""
    raw = os.environ.get("ALLOWED_MODELS", settings.allowed_models)
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        # Fallback to a safe guess if env is missing, but this shouldn't happen in production
        return [settings.fireworks_model_id]
    return models

class LocalQwenClient:
    """
    REDIRECTED FOR PRODUCTION: Uses the 'Fast' Fireworks model 
    instead of local vLLM to save RAM and ensure scores.
    """
    def __init__(self, settings: Settings) -> None:
        models = get_allowed_models(settings)
        # USE THE FIRST MODEL (Usually the smallest/cheapest allowed model)
        self.model = models[0] 
        self.max_tokens = settings.local_max_tokens
        self.timeout = settings.local_timeout_s
        self._client = AsyncOpenAI(
            base_url=settings.fireworks_base_url, # Use dynamic URL
            api_key=settings.fireworks_api_key,
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
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            tokens_used=usage.total_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
            model_name=self.model,
        )

    async def health_check(self) -> bool:
        return True

class FireworksRemoteClient:
    """Client for Fireworks AI remote API using the 'Strong' model."""
    def __init__(self, settings: Settings) -> None:
        models = get_allowed_models(settings)
        # USE THE LAST MODEL (Usually the strongest/largest allowed model)
        self.model = models[-1] 
        self.timeout = settings.remote_timeout_s
        self._client = AsyncOpenAI(
            base_url=settings.fireworks_base_url, # Use dynamic URL
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
        raise last_exc

    async def health_check(self) -> bool:
        return True
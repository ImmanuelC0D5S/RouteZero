from __future__ import annotations
from typing import Dict, Optional

from config import Config


class RouterAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def route(self, prompt: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Decide whether to use the local or remote model for the prompt."""
        return "local"

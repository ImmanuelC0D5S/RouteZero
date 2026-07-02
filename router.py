from __future__ import annotations
from typing import Any, Dict, Optional

from config import Config


class RouteLLMController:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.router_model_url = config.local_rocm_url

    def decide_route(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Return the route decision for a prompt.

        This stub uses a simple threshold-based placeholder instead of a real RouteLLM model.
        """
        # TODO: wire in RouteLLM model inference based on prompt complexity and cost.
        if len(prompt) > 80:
            return "remote"
        return "local"

    def explain_decision(self, prompt: str) -> Dict[str, Any]:
        return {
            "prompt_length": len(prompt),
            "route_cost_threshold": self.config.route_cost_threshold,
        }


def create_router_controller(config: Config) -> RouteLLMController:
    return RouteLLMController(config=config)

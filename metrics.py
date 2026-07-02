from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class MetricsTracker:
    cache_hits: int = 0
    cache_misses: int = 0
    routes: Dict[str, int] = field(default_factory=dict)
    tokens_used: int = 0
    total_cost: float = 0.0

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_route(self, route_name: str, cost: float, tokens: int) -> None:
        self.routes.setdefault(route_name, 0)
        self.routes[route_name] += 1
        self.total_cost += cost
        self.tokens_used += tokens

    def export_json(self, path: str) -> None:
        payload = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "routes": self.routes,
            "tokens_used": self.tokens_used,
            "total_cost": self.total_cost,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def export_csv(self, path: str) -> None:
        rows = [
            ["metric", "value"],
            ["cache_hits", self.cache_hits],
            ["cache_misses", self.cache_misses],
            ["tokens_used", self.tokens_used],
            ["total_cost", self.total_cost],
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)

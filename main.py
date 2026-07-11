"""Batch entry point for Docker container.

Reads /input/tasks.json, runs each task through the RouteZero pipeline,
writes results to /output/results.json, and exits with code 0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

from routezero.config import get_settings
from routezero.pipeline import RouteZeroPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


async def main() -> None:
    settings = get_settings()
    pipeline = RouteZeroPipeline(settings)
    await pipeline.metrics.init_db()
    await pipeline.init()

    input_path = Path("/input/tasks.json")
    output_path = Path("/output/results.json")

    if not input_path.exists():
        logger.info("No /input/tasks.json found — nothing to do.")
        sys.exit(0)

    tasks = json.loads(input_path.read_text())
    logger.info("Loaded %d task(s) from %s", len(tasks), input_path)

    results = []
    for task in tasks:
        task_id = task.get("task_id", str(uuid4()))
        prompt = task.get("prompt", "")
        if not prompt:
            logger.warning("Skipping task %s — empty prompt", task_id)
            continue
        result = await pipeline.run(prompt)
        results.append({
            "task_id": task_id,
            "answer": result.response,
            "route": result.route,
            "verified": result.verified,
            "cache_hit": result.cache_hit,
        })
        logger.info(
            "Processed task %s (route=%s, verified=%s, cache_hit=%s)",
            task_id, result.route, result.verified, result.cache_hit,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %d result(s) to %s", len(results), output_path)


if __name__ == "__main__":
    asyncio.run(main())

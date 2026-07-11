from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from routezero.config import get_settings
from routezero.pipeline import RouteZeroPipeline, PipelineResult


async def run_batch_from_file(pipeline: RouteZeroPipeline, input_path: Path, output_path: Path) -> None:
    tasks = json.loads(input_path.read_text())
    results = []
    for task in tasks:
        task_id = task.get("task_id", str(uuid4()))
        prompt = task.get("prompt", "")
        if not prompt:
            continue
        result: PipelineResult = await pipeline.run(prompt)
        results.append({
            "task_id": task_id,
            "answer": result.response,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {output_path}")


async def run_batch_mode() -> int:
    """Competition entrypoint: process /input/tasks.json, write /output/results.json, exit."""
    settings = get_settings()
    pipeline = RouteZeroPipeline(settings)
    await pipeline.metrics.init_db()
    await pipeline.init()

    input_path = Path("/input/tasks.json")
    output_path = Path("/output/results.json")

    if not input_path.exists():
        print(f"FATAL: {input_path} not found", file=sys.stderr)
        return 1

    try:
        await run_batch_from_file(pipeline, input_path, output_path)
        return 0
    except Exception as e:
        print(f"FATAL ERROR during batch processing: {e}", file=sys.stderr)
        return 1


def run_server_mode():
    """Dev/demo mode only — interactive FastAPI server. NOT used by the
    competition harness. Run explicitly with: python app.py --serve
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    class ChatRequest(BaseModel):
        prompt: str
        conversation_id: str | None = None

    class ChatResponse(BaseModel):
        response: str
        route: str
        verified: bool
        cache_hit: bool
        metadata: dict

    app = FastAPI(title="RouteZero", version="0.1.0")
    pipeline_holder: dict = {}

    @app.on_event("startup")
    async def startup():
        settings = get_settings()
        pipeline_holder["pipeline"] = RouteZeroPipeline(settings)
        await pipeline_holder["pipeline"].metrics.init_db()
        await pipeline_holder["pipeline"].init()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        pipeline = pipeline_holder.get("pipeline")
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        if not req.prompt or not req.prompt.strip():
            raise HTTPException(422, detail="Prompt must not be empty")
        result: PipelineResult = await pipeline.run(req.prompt, req.conversation_id)
        return ChatResponse(
            response=result.response,
            route=result.route,
            verified=result.verified,
            cache_hit=result.cache_hit,
            metadata=result.metadata,
        )

    @app.get("/metrics/dashboard")
    async def dashboard():
        pipeline = pipeline_holder.get("pipeline")
        if pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        return await pipeline.metrics.get_dashboard_snapshot()

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        run_server_mode()
    else:
        # Default: competition batch mode — process tasks, write results, EXIT.
        exit_code = asyncio.run(run_batch_mode())
        sys.exit(exit_code)
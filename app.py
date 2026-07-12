from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from routezero.config import get_settings
from routezero.pipeline import RouteZeroPipeline, PipelineResult
from routezero.metrics import DashboardSnapshot



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
pipeline: RouteZeroPipeline | None = None


@app.on_event("startup")
async def startup():
    global pipeline
    settings = get_settings()
    pipeline = RouteZeroPipeline(settings)
    await pipeline.metrics.init_db()
    # Eagerly load cache model + build router prototypes
    await pipeline.init()
    input_path = Path("/input/tasks.json")
    output_path = Path("/output/results.json")
    if input_path.exists():
        await run_batch_from_file(input_path, output_path)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
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


@app.get("/metrics/dashboard", response_model=DashboardSnapshot)
async def dashboard() -> DashboardSnapshot:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return await pipeline.metrics.get_dashboard_snapshot()


async def run_batch_from_file(input_path: Path, output_path: Path) -> None:
    if pipeline is None:
        return

    tasks = json.loads(input_path.read_text())
    results = []
    for task in tasks:
        task_id = task.get("task_id", str(uuid4()))
        prompt = task.get("prompt", "")
        if not prompt:
            continue
        result = await pipeline.run(prompt)
        results.append({
            "task_id": task_id,
            "answer": result.response,
            "route": result.route,
            "verified": result.verified,
            "cache_hit": result.cache_hit,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from routezero.config import Settings, TaskType, get_settings
from routezero.conversation import ConversationStore, PipelineState
from routezero.metrics import MetricsLogger, RequestLogRecord
from routezero.verifier import verify


@dataclass
class PipelineResult:
    prompt: str
    response: str
    route: str
    verified: bool
    cache_hit: bool
    metadata: dict[str, Any]


class RouteZeroPipeline:
    """RouteZero pipeline with lazy initialization.
    
    Constructor is lightweight — heavy model loading happens in async init().
    Call await pipeline.init() before first use.
    """
    
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._initialized = False
        # Lightweight components (no model loading)
        self.conversation = ConversationStore()
        self.metrics = MetricsLogger(self.settings)
        # Heavy components (loaded in init())
        self.cache = None
        self.router = None
        self.local_client = None
        self.remote_client = None
        self.graph = None
    
    async def init(self) -> None:
        """Load heavyweight components: cache (HF model), router, LLM clients.
        
        Call this once before using the pipeline. Safe to call multiple times.
        """
        if self._initialized:
            return
        
        # Lazy imports — only loaded when init() is called
        from routezero.cache import SemanticCache
        from routezero.router import SemanticHeuristicRouter
        from routezero.llm_clients import LocalQwenClient, FireworksRemoteClient
        
        self.cache = SemanticCache(self.settings)
        self.router = SemanticHeuristicRouter(self.cache.embedder, self.settings)
        self.local_client = LocalQwenClient(self.settings)
        self.remote_client = FireworksRemoteClient(self.settings)
        self.graph = self._build_graph()
        self._initialized = True
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph execution graph."""
        graph = StateGraph(PipelineState)
        graph.add_node("check_semantic_cache", self._node_check_cache)
        graph.add_node("route_decision", self._node_route_decision)
        graph.add_node("call_local_qwen", self._node_call_local)
        graph.add_node("call_remote_api", self._node_call_remote)
        graph.add_node("verify_output", self._node_verify)
        graph.add_node("finalize_response", self._node_finalize)

        graph.set_entry_point("check_semantic_cache")
        graph.add_conditional_edges(
            "check_semantic_cache",
            self.route_on_cache_hit,
            {"hit": "finalize_response", "miss": "route_decision"},
        )
        graph.add_conditional_edges(
            "route_decision",
            self.route_on_target,
            {"local": "call_local_qwen", "remote": "call_remote_api"},
        )
        graph.add_edge("call_local_qwen", "verify_output")
        graph.add_conditional_edges(
            "verify_output",
            self.route_on_verification,
            {"pass": "finalize_response", "fail": "call_remote_api"},
        )
        graph.add_edge("call_remote_api", "finalize_response")
        graph.set_finish_point("finalize_response")
        return graph.compile()
    
    # ── Node methods ─────────────────────────────────────────────────────
    
    async def _node_check_cache(self, state: PipelineState) -> dict:
        prompt = state["user_prompt"]
        result = self.cache.query(prompt)  # type: ignore[union-attr]
        if result is not None:
            return {
                "cache_hit": True,
                "model_response": result.response,
                "task_type": result.task_type,
                "routing_target": "cache",
            }
        return {"cache_hit": False}
    
    async def _node_route_decision(self, state: PipelineState) -> dict:
        decision = self.router.decide(state["user_prompt"])  # type: ignore[union-attr]
        return {
            "routing_target": decision.target.value,
            "task_type": decision.task_type.value,
        }
    
    async def _node_call_local(self, state: PipelineState) -> dict:
        start = time.monotonic()
        response = await asyncio.wait_for(
            self.local_client.generate(state["user_prompt"]),  # type: ignore[union-attr]
            timeout=self.settings.local_timeout_s,
        )
        latency = (time.monotonic() - start) * 1000
        return {
            "model_response": response.text,
            "execution_latency_ms": latency,
            "tokens_used": response.tokens_used,
            "model_name": response.model_name,
        }
    
    async def _node_call_remote(self, state: PipelineState) -> dict:
        start = time.monotonic()
        response = await self.remote_client.generate(state["user_prompt"])  # type: ignore[union-attr]
        latency = (time.monotonic() - start) * 1000
        return {
            "model_response": response.text,
            "execution_latency_ms": latency,
            "routing_target": "remote",
            "tokens_used": response.tokens_used,
            "model_name": response.model_name,
        }
    
    async def _node_verify(self, state: PipelineState) -> dict:
        task_type = TaskType(state["task_type"])
        result = verify(state["model_response"], task_type)
        return {"verification_passed": result.passed}
    
    async def _node_finalize(self, state: PipelineState) -> dict:
        record = RequestLogRecord(
            request_id=state.get("conversation_id", ""),
            cache_status="HIT" if state["cache_hit"] else "MISS",
            selected_route=state["routing_target"].upper(),
            task_type=state["task_type"],
            saved_tokens=0,
            latency_ms=state.get("execution_latency_ms", 0.0),
            verification_passed=state.get("verification_passed"),
            timestamp=datetime.now(timezone.utc),
        )
        await self.metrics.log_request(record)

        # Populate cache on cache miss for future lookups
        if not state["cache_hit"] and state.get("model_response"):
            await asyncio.to_thread(
                self.cache.insert,
                state["user_prompt"],
                state["model_response"],
                {"task_type": state["task_type"]},
            )

        return {}
    
    # ── Conditional edge predicates ──────────────────────────────────────
    
    @staticmethod
    def route_on_cache_hit(state: PipelineState) -> str:
        return "hit" if state["cache_hit"] else "miss"
    
    @staticmethod
    def route_on_target(state: PipelineState) -> str:
        return state["routing_target"]
    
    @staticmethod
    def route_on_verification(state: PipelineState) -> str:
        return "pass" if state["verification_passed"] else "fail"
    
    # ── Entry point ──────────────────────────────────────────────────────
    
    async def run(
        self, prompt: str, conversation_id: str | None = None
    ) -> PipelineResult:
        if not self._initialized:
            await self.init()
        
        if conversation_id is not None:
            full_prompt = self.conversation.build_contextual_prompt(
                conversation_id, prompt
            )
        else:
            full_prompt = prompt
        
        initial: PipelineState = {
            "user_prompt": full_prompt,
            "prompt_embedding": [],
            "cache_hit": False,
            "routing_target": "",
            "task_type": "",
            "model_response": "",
            "verification_passed": False,
            "execution_latency_ms": 0.0,
            "tokens_used": 0,
            "model_name": "",
            "conversation_id": conversation_id or "",
            "turn_index": 0,
        }
        
        start_time = time.monotonic()
        final_state = await self.graph.ainvoke(initial)  # type: ignore[union-attr]
        total_latency = (time.monotonic() - start_time) * 1000
        
        metadata: dict[str, Any] = {
            "routing_target": final_state.get("routing_target", ""),
            "task_type": final_state.get("task_type", ""),
            "execution_latency_ms": final_state.get("execution_latency_ms", 0.0),
            "total_pipeline_latency_ms": round(total_latency, 2),
            "tokens_used": final_state.get("tokens_used", 0),
            "model_name": final_state.get("model_name", ""),
        }
        
        if conversation_id is not None:
            self.conversation.append_turn(conversation_id, "user", prompt)
            self.conversation.append_turn(
                conversation_id, "assistant", final_state["model_response"]
            )
        
        return PipelineResult(
            prompt=prompt,
            response=final_state["model_response"],
            route=final_state.get("routing_target", "cache"),
            verified=final_state.get("verification_passed", False),
            cache_hit=final_state.get("cache_hit", False),
            metadata=metadata,
        )

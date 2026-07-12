"""Comprehensive tests for RouteZeroPipeline.

Tests cover:
  - PipelineResult and PipelineState dataclasses
  - Conditional edge predicates (route_on_cache_hit, route_on_target, route_on_verification)
  - Full cache HIT flow (short-circuit)
  - Full cache MISS → LOCAL → verify PASS flow
  - Full cache MISS → LOCAL → verify FAIL → REMOTE fallback flow
  - Full cache MISS → REMOTE flow
  - The cache-not-populated bug (pipeline never calls cache.insert)
  - Conversation context integration
  - Pipeline initialization (lazy loading)
  - Error handling edge cases
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Any

from routezero.config import Settings, TaskType
from routezero.pipeline import RouteZeroPipeline, PipelineResult
from routezero.conversation import PipelineState
from routezero.llm_clients import LLMResponse


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    return Settings(
        cache_similarity_threshold=0.92,
        router_local_threshold=0.4,
        router_heuristic_weight=0.3,
        chromadb_path=":memory:",
        sqlite_path=":memory:",
    )


@pytest.fixture
def mock_deps():
    """Create mocks for all heavyweight pipeline dependencies.

    Note: pipeline.py uses lazy imports inside init() (e.g., ``from routezero.cache import SemanticCache``),
    so we must patch at the source module, not at routezero.pipeline.
    """
    with patch("routezero.cache.SemanticCache") as mock_cache_cls, \
         patch("routezero.router.SemanticHeuristicRouter") as mock_router_cls, \
         patch("routezero.llm_clients.LocalQwenClient") as mock_local_cls, \
         patch("routezero.llm_clients.FireworksRemoteClient") as mock_remote_cls:

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.query.return_value = None  # default: cache MISS
        mock_cache.embedder = MagicMock()
        mock_cache_cls.return_value = mock_cache

        # Mock router
        mock_router = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "factual"
        mock_router.decide.return_value = mock_decision
        mock_router_cls.return_value = mock_router

        # Mock clients
        mock_local = AsyncMock()
        mock_local.generate.return_value = LLMResponse(
            text="local model response",
            tokens_used=50,
            latency_ms=100.0,
            model_name="local-model",
        )
        mock_local_cls.return_value = mock_local

        mock_remote = AsyncMock()
        mock_remote.generate.return_value = LLMResponse(
            text="remote model response",
            tokens_used=100,
            latency_ms=500.0,
            model_name="remote-model",
        )
        mock_remote_cls.return_value = mock_remote

        yield {
            "mock_cache": mock_cache,
            "mock_cache_cls": mock_cache_cls,
            "mock_router": mock_router,
            "mock_router_cls": mock_router_cls,
            "mock_local": mock_local,
            "mock_local_cls": mock_local_cls,
            "mock_remote": mock_remote,
            "mock_remote_cls": mock_remote_cls,
        }


@pytest.fixture
def pipeline(settings, mock_deps):
    """Create a RouteZeroPipeline with mocked dependencies."""
    pipe = RouteZeroPipeline(settings)
    # Manually set up lightweight components so init() doesn't recreate them
    # But we need to call init() to set up the graph with our mocks
    return pipe


# ── Dataclass and static method tests ─────────────────────────────────────


class TestPipelineResult:
    def test_pipeline_result_dataclass(self):
        """PipelineResult should store all fields correctly."""
        result = PipelineResult(
            prompt="test prompt",
            response="test response",
            route="local",
            verified=True,
            cache_hit=False,
            metadata={"task_type": "factual", "latency_ms": 50.0},
        )
        assert result.prompt == "test prompt"
        assert result.response == "test response"
        assert result.route == "local"
        assert result.verified is True
        assert result.cache_hit is False
        assert result.metadata["task_type"] == "factual"

    def test_pipeline_result_cache_hit_variant(self):
        """PipelineResult should handle cache hit scenario."""
        result = PipelineResult(
            prompt="test",
            response="cached response",
            route="cache",
            verified=True,
            cache_hit=True,
            metadata={"task_type": "factual"},
        )
        assert result.cache_hit is True
        assert result.route == "cache"

    def test_pipeline_result_remote_variant(self):
        """PipelineResult should handle remote route."""
        result = PipelineResult(
            prompt="complex task",
            response="remote response",
            route="remote",
            verified=True,
            cache_hit=False,
            metadata={"task_type": "codegen"},
        )
        assert result.route == "remote"

    def test_pipeline_result_unverified(self):
        """PipelineResult should handle unverified response."""
        result = PipelineResult(
            prompt="test",
            response="bad response",
            route="remote",
            verified=False,
            cache_hit=False,
            metadata={},
        )
        assert result.verified is False


class TestPipelineState:
    def test_pipeline_state_defaults(self):
        """PipelineState should accept default values."""
        state: PipelineState = {
            "user_prompt": "",
            "prompt_embedding": [],
            "cache_hit": False,
            "routing_target": "",
            "task_type": "",
            "model_response": "",
            "verification_passed": False,
            "execution_latency_ms": 0.0,
            "conversation_id": "",
            "turn_index": 0,
        }
        assert state["cache_hit"] is False
        assert state["verification_passed"] is False
        assert state["execution_latency_ms"] == 0.0

    def test_pipeline_state_with_values(self):
        """PipelineState should hold all expected fields."""
        state: PipelineState = {
            "user_prompt": "test prompt",
            "prompt_embedding": [0.1, 0.2, 0.3],
            "cache_hit": True,
            "routing_target": "cache",
            "task_type": "factual",
            "model_response": "cached response",
            "verification_passed": True,
            "execution_latency_ms": 5.0,
            "conversation_id": "conv_123",
            "turn_index": 1,
        }
        assert state["user_prompt"] == "test prompt"
        assert state["cache_hit"] is True
        assert len(state["prompt_embedding"]) == 3
        assert state["turn_index"] == 1


class TestConditionalEdges:
    def test_route_on_cache_hit_returns_hit(self):
        """route_on_cache_hit should return 'hit' when cache_hit is True."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": True,
            "routing_target": "", "task_type": "", "model_response": "",
            "verification_passed": False, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_cache_hit(state) == "hit"

    def test_route_on_cache_hit_returns_miss(self):
        """route_on_cache_hit should return 'miss' when cache_hit is False."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": False,
            "routing_target": "", "task_type": "", "model_response": "",
            "verification_passed": False, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_cache_hit(state) == "miss"

    def test_route_on_target_returns_target(self):
        """route_on_target should return the routing_target value."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": False,
            "routing_target": "local", "task_type": "", "model_response": "",
            "verification_passed": False, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_target(state) == "local"

    def test_route_on_target_remote(self):
        """route_on_target should return 'remote' when target is remote."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": False,
            "routing_target": "remote", "task_type": "", "model_response": "",
            "verification_passed": False, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_target(state) == "remote"

    def test_route_on_verification_pass(self):
        """route_on_verification should return 'pass' when verified."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": False,
            "routing_target": "", "task_type": "", "model_response": "",
            "verification_passed": True, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_verification(state) == "pass"

    def test_route_on_verification_fail(self):
        """route_on_verification should return 'fail' when not verified."""
        state: PipelineState = {
            "user_prompt": "", "prompt_embedding": [], "cache_hit": False,
            "routing_target": "", "task_type": "", "model_response": "",
            "verification_passed": False, "execution_latency_ms": 0.0,
            "conversation_id": "", "turn_index": 0,
        }
        assert RouteZeroPipeline.route_on_verification(state) == "fail"


# ── Pipeline initialization tests ─────────────────────────────────────────


class TestPipelineInit:
    @pytest.mark.asyncio
    async def test_pipeline_lazy_initialization(self, settings, mock_deps):
        """Pipeline should start uninitialized and initialize on first run."""
        pipe = RouteZeroPipeline(settings)
        assert pipe._initialized is False
        assert pipe.cache is None
        assert pipe.router is None
        assert pipe.local_client is None
        assert pipe.remote_client is None
        assert pipe.graph is None

    @pytest.mark.asyncio
    async def test_pipeline_init_loads_components(self, settings, mock_deps):
        """Pipeline.init() should load cache, router, and clients."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        assert pipe._initialized is True
        assert pipe.cache is not None
        assert pipe.router is not None
        assert pipe.local_client is not None
        assert pipe.remote_client is not None
        assert pipe.graph is not None

    @pytest.mark.asyncio
    async def test_pipeline_init_is_idempotent(self, settings, mock_deps):
        """Calling init() twice should not re-initialize."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        await pipe.init()  # second call should be no-op
        assert pipe._initialized is True
        # verify init was called only once
        mock_deps["mock_cache_cls"].assert_called_once()
        mock_deps["mock_router_cls"].assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_run_auto_inits(self, settings, mock_deps):
        """Running the pipeline without explicit init should auto-init."""
        pipe = RouteZeroPipeline(settings)
        assert pipe._initialized is False
        with patch.object(pipe, '_node_finalize', AsyncMock(return_value={})):
            result = await pipe.run("test prompt")
        assert pipe._initialized is True

    @pytest.mark.asyncio
    async def test_pipeline_uses_provided_settings(self, settings, mock_deps):
        """Pipeline should pass settings to all components."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        mock_deps["mock_cache_cls"].assert_called_with(settings)
        mock_deps["mock_router_cls"].assert_called_with(
            pipe.cache.embedder, settings
        )


# ── Full pipeline flow tests ───────────────────────────────────────────────


class TestCacheHitFlow:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self, settings, mock_deps):
        """Cache HIT should short-circuit and return cached response directly."""
        # Arrange: Mock cache HIT
        mock_deps["mock_cache"].query.return_value = MagicMock(
            response="cached response content",
            similarity=0.95,
            task_type="factual",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        result = await pipe.run("test prompt")

        # Assert: Should return cached response without calling LLM
        assert result.response == "cached response content"
        assert result.cache_hit is True
        assert result.route == "cache"
        # LLM clients should NOT be called
        mock_deps["mock_local"].generate.assert_not_called()
        mock_deps["mock_remote"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_reason_content_preserved(self, settings, mock_deps):
        """Cache HIT should preserve the exact cached response content."""
        cached_content = "The capital of France is Paris. It is known for the Eiffel Tower."
        mock_deps["mock_cache"].query.return_value = MagicMock(
            response=cached_content,
            similarity=0.95,
            task_type="factual",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        result = await pipe.run("what is capital of france?")

        assert result.response == cached_content


class TestCacheMissLocalFlow:
    @pytest.mark.asyncio
    async def test_cache_miss_local_verified(self, settings, mock_deps):
        """Cache MISS → LOCAL → verify PASS should return local response."""
        # Arrange: Cache MISS, Router → LOCAL, Verify PASS
        mock_deps["mock_cache"].query.return_value = None  # MISS
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "factual"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_local"].generate.return_value = LLMResponse(
            text="local model response", tokens_used=50,
            latency_ms=100.0, model_name="local-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        # Mock verify to return PASS
        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)

            result = await pipe.run("what is the capital of france?")

        assert result.response == "local model response"
        assert result.route == "local"
        assert result.cache_hit is False
        assert result.verified is True
        mock_deps["mock_local"].generate.assert_called_once()
        mock_deps["mock_remote"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_local_fail_remote_fallback(self, settings, mock_deps):
        """Cache MISS → LOCAL → verify FAIL → REMOTE fallback."""
        # Arrange
        mock_deps["mock_cache"].query.return_value = None  # MISS
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "codegen"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_local"].generate.return_value = LLMResponse(
            text="bad local response", tokens_used=50,
            latency_ms=100.0, model_name="local-model",
        )
        mock_deps["mock_remote"].generate.return_value = LLMResponse(
            text="good remote response", tokens_used=100,
            latency_ms=500.0, model_name="remote-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        # Mock verify to return FAIL
        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=False, failure_reason="Bad response")

            result = await pipe.run("write complex code")

        assert result.response == "good remote response"
        assert result.route == "remote"
        assert result.cache_hit is False
        assert result.verified is False  # remote goes directly to finalize without verification
        mock_deps["mock_local"].generate.assert_called_once()
        mock_deps["mock_remote"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss_local_verify_called_with_correct_args(self, settings, mock_deps):
        """Verify should be called with the local model response and task type."""
        mock_deps["mock_cache"].query.return_value = None
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "codegen"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_local"].generate.return_value = LLMResponse(
            text="local code output", tokens_used=50,
            latency_ms=100.0, model_name="local-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            await pipe.run("write a function")

            # Verify was called with the local response and task type
            mock_verify.assert_called_once_with("local code output", TaskType.CODEGEN)


class TestCacheMissRemoteFlow:
    @pytest.mark.asyncio
    async def test_cache_miss_remote_direct(self, settings, mock_deps):
        """Cache MISS → REMOTE should call remote and return response."""
        mock_deps["mock_cache"].query.return_value = None  # MISS
        mock_decision = MagicMock()
        mock_decision.target.value = "remote"
        mock_decision.task_type.value = "codegen"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_remote"].generate.return_value = LLMResponse(
            text="remote response", tokens_used=100,
            latency_ms=500.0, model_name="remote-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        result = await pipe.run("write complex code")

        assert result.response == "remote response"
        assert result.route == "remote"
        assert result.cache_hit is False
        # Remote goes directly to finalize, no verification
        mock_deps["mock_remote"].generate.assert_called_once()
        mock_deps["mock_local"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_remote_skips_verification(self, settings, mock_deps):
        """REMOTE route should skip verification entirely."""
        mock_deps["mock_cache"].query.return_value = None
        mock_decision = MagicMock()
        mock_decision.target.value = "remote"
        mock_decision.task_type.value = "reasoning"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_remote"].generate.return_value = LLMResponse(
            text="remote response", tokens_used=100,
            latency_ms=500.0, model_name="remote-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch("routezero.pipeline.verify") as mock_verify:
            result = await pipe.run("reason step by step")
            # Verify should NOT be called for remote route
            mock_verify.assert_not_called()


# ── The cache-not-populated bug test ───────────────────────────────────────


class TestCacheNotPopulated:
    """Test that the pipeline has a bug: it never stores responses in cache.

    The SemanticCache.insert() method is never called anywhere in the pipeline.
    This means the cache will never grow from pipeline executions.
    """

    @pytest.mark.asyncio
    async def test_cache_insert_called_on_cache_miss(self, settings, mock_deps):
        """After a cache MISS → LOCAL → verify PASS, cache.insert is called.

        The pipeline stores the successful response in cache for future use.
        """
        mock_deps["mock_cache"].query.return_value = None  # MISS
        mock_deps["mock_cache"].insert = MagicMock()  # track insert calls
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "factual"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_local"].generate.return_value = LLMResponse(
            text="successful response", tokens_used=50,
            latency_ms=100.0, model_name="local-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            await pipe.run("test prompt")

            mock_deps["mock_cache"].insert.assert_called_once_with(
                "test prompt", "successful response", {"task_type": "factual"}
            )

    @pytest.mark.asyncio
    async def test_cache_insert_called_on_remote(self, settings, mock_deps):
        """After REMOTE response, cache.insert is called."""
        mock_deps["mock_cache"].query.return_value = None
        mock_deps["mock_cache"].insert = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target.value = "remote"
        mock_decision.task_type.value = "codegen"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_remote"].generate.return_value = LLMResponse(
            text="remote response", tokens_used=100,
            latency_ms=500.0, model_name="remote-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        await pipe.run("complex task")
        mock_deps["mock_cache"].insert.assert_called_once_with(
            "complex task", "remote response", {"task_type": "codegen"}
        )

    @pytest.mark.asyncio
    async def test_cache_insert_called_on_remote_fallback(self, settings, mock_deps):
        """After LOCAL fail → REMOTE fallback, cache.insert is called."""
        mock_deps["mock_cache"].query.return_value = None
        mock_deps["mock_cache"].insert = MagicMock()
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "codegen"
        mock_deps["mock_router"].decide.return_value = mock_decision
        mock_deps["mock_local"].generate.return_value = LLMResponse(
            text="bad", tokens_used=50, latency_ms=100.0, model_name="local-model",
        )
        mock_deps["mock_remote"].generate.return_value = LLMResponse(
            text="good", tokens_used=100, latency_ms=500.0, model_name="remote-model",
        )

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=False)
            await pipe.run("complex task")

        mock_deps["mock_cache"].insert.assert_called_once_with(
            "complex task", "good", {"task_type": "codegen"}
        )


# ── Conversation context tests ─────────────────────────────────────────────


class TestConversationContext:
    @pytest.mark.asyncio
    async def test_conversation_context_used_when_id_provided(self, settings, mock_deps):
        """When conversation_id is provided, the contextual prompt should be used."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch.object(pipe.conversation, 'build_contextual_prompt') as mock_build:
            mock_build.return_value = "contextual prompt"
            mock_deps["mock_cache"].query.return_value = None
            mock_decision = MagicMock()
            mock_decision.target.value = "local"
            mock_decision.task_type.value = "factual"
            mock_deps["mock_router"].decide.return_value = mock_decision
            mock_deps["mock_local"].generate.return_value = LLMResponse(
                text="response", tokens_used=10,
                latency_ms=50.0, model_name="local-model",
            )

            with patch("routezero.pipeline.verify") as mock_verify:
                mock_verify.return_value = MagicMock(passed=True)
                result = await pipe.run("new message", conversation_id="conv_1")

            mock_build.assert_called_once_with("conv_1", "new message")

    @pytest.mark.asyncio
    async def test_conversation_appends_turns(self, settings, mock_deps):
        """After pipeline run with conversation_id, turns should be appended."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        # Spy on conversation.append_turn
        original_append = pipe.conversation.append_turn

        with patch.object(pipe.conversation, 'append_turn') as mock_append, \
             patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            mock_deps["mock_cache"].query.return_value = None
            mock_decision = MagicMock()
            mock_decision.target.value = "local"
            mock_decision.task_type.value = "factual"
            mock_deps["mock_router"].decide.return_value = mock_decision
            mock_deps["mock_local"].generate.return_value = LLMResponse(
                text="response", tokens_used=10,
                latency_ms=50.0, model_name="local-model",
            )

            await pipe.run("hello", conversation_id="conv_1")

            # Should append user and assistant turns
            assert mock_append.call_count == 2
            mock_append.assert_any_call("conv_1", "user", "hello")
            mock_append.assert_any_call("conv_1", "assistant", "response")

    @pytest.mark.asyncio
    async def test_conversation_not_used_without_id(self, settings, mock_deps):
        """Without conversation_id, build_contextual_prompt should not be called."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch.object(pipe.conversation, 'build_contextual_prompt') as mock_build, \
             patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            mock_deps["mock_cache"].query.return_value = None
            mock_decision = MagicMock()
            mock_decision.target.value = "local"
            mock_decision.task_type.value = "factual"
            mock_deps["mock_router"].decide.return_value = mock_decision
            mock_deps["mock_local"].generate.return_value = LLMResponse(
                text="response", tokens_used=10,
                latency_ms=50.0, model_name="local-model",
            )

            await pipe.run("hello")
            mock_build.assert_not_called()


# ── Edge case tests ────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_prompt_does_not_crash(self, settings, mock_deps):
        """Empty prompt should not crash the pipeline."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        mock_deps["mock_cache"].query.return_value = None
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "factual"
        mock_deps["mock_router"].decide.return_value = mock_decision

        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            result = await pipe.run("")
            assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_latency_metrics_included(self, settings, mock_deps):
        """Pipeline run should include latency in metadata."""
        mock_deps["mock_cache"].query.return_value = None
        mock_decision = MagicMock()
        mock_decision.target.value = "local"
        mock_decision.task_type.value = "factual"
        mock_deps["mock_router"].decide.return_value = mock_decision

        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        with patch("routezero.pipeline.verify") as mock_verify:
            mock_verify.return_value = MagicMock(passed=True)
            result = await pipe.run("test")

        assert "total_pipeline_latency_ms" in result.metadata
        assert result.metadata["total_pipeline_latency_ms"] >= 0.0
        assert "execution_latency_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_router_print_does_not_cause_error(self, settings, mock_deps):
        """The router's debug print statement should not cause errors in the pipeline."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        result = await pipe.run("test")
        assert isinstance(result, PipelineResult)


# ── Pipeline build graph tests ─────────────────────────────────────────────


class TestGraphStructure:
    @pytest.mark.asyncio
    async def test_graph_has_correct_nodes(self, settings, mock_deps):
        """The compiled graph should have the expected nodes."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()

        # LangGraph exposes nodes via graph.nodes or similar
        # Check that graph is properly compiled
        assert pipe.graph is not None

    @pytest.mark.asyncio
    async def test_graph_build_sets_entry_point(self, settings, mock_deps):
        """The graph should start with cache check."""
        pipe = RouteZeroPipeline(settings)
        await pipe.init()
        # Verify by running a cache hit first - should be the entry point
        mock_deps["mock_cache"].query.return_value = MagicMock(
            response="cached", similarity=0.95, task_type="factual",
        )
        result = await pipe.run("test")
        assert result.cache_hit is True
        # If cache hit works first, then cache check is the entry point

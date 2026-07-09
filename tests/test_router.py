"""Comprehensive tests for SemanticHeuristicRouter.

Tests cover:
  - Prototype initialization for all 8 task types
  - Task classification for all task types
  - Heuristic signal detection (code, numerics, constraints, output demands)
  - Edge cases: empty prompt, very long prompt, short-circuit logic ordering
  - Routing decisions for LOCAL vs REMOTE targets
  - Threshold boundary testing
  - The short conversational prompt short-circuit behavior
  - Confidence capping bug (can exceed 1.0)
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from routezero.config import Settings, TaskType, RouteTarget
from routezero.router import SemanticHeuristicRouter, RoutingDecision


# ── Mock embedders ─────────────────────────────────────────────────────────


class MockEmbedder:
    """Returns deterministic embeddings for basic testing.

    Uses seeded random based on text content so same text = same embedding.
    Does NOT perform semantic classification — use mock for that.
    """
    def encode(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(384)


class KeywordAwareEmbedder:
    """Returns embeddings that roughly correlate with certain prompt keywords.

    Prompts containing code-related keywords get embeddings closer to
    CODEGEN/DEBUG prototypes. Factual/simple prompts get embeddings in
    a different region. This allows basic routing tests without mocking
    classify_task.
    """
    def __init__(self) -> None:
        np.random.seed(0)
        self._code_region = np.random.randn(384) * 2
        self._factual_region = np.random.randn(384) * 2
        self._math_region = np.random.randn(384) * 2

    def encode(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        code_keywords = ["write", "function", "implement", "code", "debug", "bug"]
        math_keywords = ["calculate", "compute", "number", "math", "solve", "equation"]

        code_score = sum(1 for kw in code_keywords if kw in text_lower)
        math_score = sum(1 for kw in math_keywords if kw in text_lower)

        if code_score > math_score:
            return self._code_region + np.random.randn(384) * 0.1
        elif math_score > 0:
            return self._math_region + np.random.randn(384) * 0.1
        else:
            return self._factual_region + np.random.randn(384) * 0.1


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    return Settings(
        router_local_threshold=0.4,
        router_heuristic_weight=0.3,
    )


@pytest.fixture
def router(settings):
    embedder = MockEmbedder()
    return SemanticHeuristicRouter(embedder, settings)


@pytest.fixture
def keyword_router(settings):
    embedder = KeywordAwareEmbedder()
    return SemanticHeuristicRouter(embedder, settings)


# ── Prototype tests ────────────────────────────────────────────────────────


class TestPrototypes:
    def test_all_eight_categories_have_prototypes(self, router):
        """All 8 TaskType categories should have a 384-dim prototype vector."""
        assert len(router.prototypes) == 8
        for task_type in TaskType:
            assert task_type in router.prototypes, f"Missing prototype for {task_type}"
            assert router.prototypes[task_type].shape == (384,)

    def test_prototypes_are_distinct(self, router):
        """Each prototype should be a distinct vector (not all the same)."""
        vectors = list(router.prototypes.values())
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                if np.array_equal(vectors[i], vectors[j]):
                    pytest.fail(f"Prototypes {i} and {j} are identical")

    def test_router_initialization_with_custom_settings(self):
        """Router should accept custom settings and pass them through."""
        s = Settings(router_local_threshold=0.7, router_heuristic_weight=0.5)
        r = SemanticHeuristicRouter(MockEmbedder(), s)
        assert r.settings.router_local_threshold == 0.7
        assert r.settings.router_heuristic_weight == 0.5

    def test_prototypes_are_built_from_category_examples(self):
        """Prototypes should be computed from CATEGORY_EXAMPLES."""
        from routezero.router import CATEGORY_EXAMPLES
        embedder = MagicMock()
        total_examples = sum(len(v) for v in CATEGORY_EXAMPLES.values())
        embedder.encode.side_effect = [
            np.full(384, i * 0.1) for i in range(total_examples)
        ]
        r = SemanticHeuristicRouter(embedder, Settings())
        assert len(r.prototypes) == len(TaskType)


# ── Classification tests ───────────────────────────────────────────────────


class TestClassifyTask:
    def test_returns_valid_task_type(self, router):
        """classify_task should always return a (TaskType, float, float) tuple."""
        result = router.classify_task("test prompt")
        task_type, best_sim, gap = result
        assert isinstance(task_type, TaskType)
        assert task_type in TaskType
        assert isinstance(best_sim, float)
        assert isinstance(gap, float)

    def test_classify_empty_string(self, router):
        """Empty string should not crash and return a valid tuple."""
        result = router.classify_task("")
        task_type, _, _ = result
        assert isinstance(task_type, TaskType)

    def test_classify_very_long_prompt(self, router):
        """Very long prompts should not crash."""
        long_prompt = "test " * 10_000
        result = router.classify_task(long_prompt)
        task_type, _, _ = result
        assert isinstance(task_type, TaskType)

    def test_classify_special_characters(self, router):
        """Prompts with special characters should not crash."""
        result = router.classify_task("!@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`")
        task_type, _, _ = result
        assert isinstance(task_type, TaskType)

    def test_classify_deterministic(self, router):
        """Same input should give same classification when embedder is deterministic."""
        t1 = router.classify_task("same text")
        t2 = router.classify_task("same text")
        assert t1 == t2


# ── Heuristic signal tests ─────────────────────────────────────────────────


class TestHeuristicSignals:
    def test_simple_greeting_has_zero_score(self, router):
        """Simple greeting should have no heuristic signals."""
        assert router.heuristic_signals("hello") == 0.0

    def test_short_conversational_prompt_has_zero_score(self, router):
        """Short chit-chat should have no heuristic signals."""
        assert router.heuristic_signals("how are you") == 0.0
        assert router.heuristic_signals("what's up") == 0.0
        assert router.heuristic_signals("ok thanks") == 0.0

    def test_code_block_detected(self, router):
        """Code blocks with backticks should boost score."""
        score = router.heuristic_signals("Write a function in ```python```")
        # "Write a function" matches output demand (+0.3) + `````` code block (+0.3) = 0.6
        assert score >= 0.6

    def test_indented_code_detected(self, router):
        """Indented lines (simulating code) should boost score."""
        score = router.heuristic_signals("def foo():\n    return bar")
        assert score >= 0.2

    def test_high_numeric_density(self, router):
        """Prompts with many digits should boost score."""
        score = router.heuristic_signals("calculate 123.45 * 678.90 + 111.22")
        assert score >= 0.10

    def test_low_numeric_density(self, router):
        """Few digits in a long prompt should NOT trigger numeric density."""
        score = router.heuristic_signals("tell me one thing about the number 5 please")
        assert score == 0.0

    def test_constraint_pattern_detected(self, router):
        """Constraint phrases should boost score."""
        score = router.heuristic_signals("must satisfy all conditions step by step")
        assert score >= 0.6

    def test_output_demand_pattern_detected(self, router):
        """Output demand phrases should boost score."""
        score = router.heuristic_signals("write a function that returns JSON")
        assert score >= 0.3

    def test_all_signals_stack_up_to_cap(self, router):
        """Multiple signals should stack (capped at 1.0)."""
        prompt = """```python
        def solve():
            pass
        ```
        Must satisfy all conditions step by step.
        Write a function that returns JSON.
        Calculate 123.45 * 678.90
        """
        score = router.heuristic_signals(prompt)
        assert score > 0.8
        assert score <= 1.0

    def test_empty_prompt(self, router):
        """Empty prompt should return 0.0."""
        assert router.heuristic_signals("") == 0.0

    def test_whitespace_prompt(self, router):
        """Whitespace-only prompt should return 0.0."""
        assert router.heuristic_signals("   \n  \t  ") == 0.0


# ── Routing decision tests ─────────────────────────────────────────────────


class TestDecide:
    def test_returns_routing_decision(self, router):
        """decide should always return a RoutingDecision."""
        decision = router.decide("what is the capital of france")
        assert isinstance(decision, RoutingDecision)

    def test_routing_decision_has_all_fields(self, router):
        """RoutingDecision should have target, task_type, confidence, reason."""
        decision = router.decide("what is the capital of france")
        assert decision.target in (RouteTarget.LOCAL, RouteTarget.REMOTE)
        assert isinstance(decision.task_type, TaskType)
        assert isinstance(decision.confidence, float)
        assert isinstance(decision.reason, str)

    def test_factual_routes_local(self, keyword_router):
        """Factual questions should route LOCAL (below threshold)."""
        decision = keyword_router.decide("what is the capital of france please tell me")
        assert decision.target == RouteTarget.LOCAL

    def test_codegen_routes_remote(self, keyword_router):
        """Code generation tasks should route REMOTE (long prompt avoids short-circuit)."""
        with patch.object(keyword_router, 'classify_task', return_value=(TaskType.CODEGEN, 0.8, 0.15)):
            decision = keyword_router.decide("please write a function that implements fibonacci")
            assert decision.target == RouteTarget.REMOTE

    def test_reasoning_routes_remote(self, keyword_router):
        """Reasoning tasks should route REMOTE."""
        with patch.object(keyword_router, 'classify_task', return_value=(TaskType.REASONING, 0.8, 0.15)):
            decision = keyword_router.decide(
                "explain the dynamic programming approach at each recursive step "
                "and show why it works step by step"
            )
            assert decision.target == RouteTarget.REMOTE

    def test_debug_routes_remote(self, keyword_router):
        """Debug tasks should route REMOTE."""
        decision = keyword_router.decide(
            "can you please find the bug in this code and fix it for me"
        )
        assert decision.target == RouteTarget.REMOTE

    def test_math_routes_local(self, keyword_router):
        """Math tasks (without heavy heuristics) should route LOCAL."""
        decision = keyword_router.decide("what is one hundred forty four divided by twelve")
        assert decision.target == RouteTarget.LOCAL

    def test_sentiment_routes_local(self, router):
        """Sentiment tasks should route LOCAL."""
        with patch.object(router, 'classify_task', return_value=(TaskType.SENTIMENT, 0.8, 0.15)):
            decision = router.decide("classify the sentiment of this text as positive or negative")
            assert decision.target == RouteTarget.LOCAL

    def test_summarize_routes_local(self, router):
        """Summarization tasks should route LOCAL."""
        decision = router.decide("summarize the following text in one sentence for me")
        assert decision.target == RouteTarget.LOCAL

    def test_ner_routes_local(self, router):
        """NER tasks should route LOCAL."""
        decision = router.decide("extract all person and location entities from this text")
        assert decision.target == RouteTarget.LOCAL

    def test_heuristic_boost_can_push_local_to_remote(self, keyword_router):
        """A factual task with code blocks should get enough heuristic boost."""
        decision = keyword_router.decide(
            "explain how this function works: def foo():\n"
            "    pass\n"
            "Must satisfy all conditions and include a full explanation."
        )
        assert decision.confidence >= keyword_router.settings.router_local_threshold

    def test_empty_prompt_routes_local(self, router):
        """Empty prompt should route LOCAL (short conversational prompt)."""
        decision = router.decide("")
        assert decision.target == RouteTarget.LOCAL

    def test_very_long_prompt_does_not_crash(self, router):
        """Very long prompts should not crash the router."""
        decision = router.decide("test " * 10_000)
        assert isinstance(decision, RoutingDecision)

    def test_math_with_high_numerics_gets_heuristic_boost(self, keyword_router):
        """Math task with high numeric density should have confidence > 0."""
        decision = keyword_router.decide(
            "calculate 123.45 * 678.90 + 111.22 - 333.44 / 555.66"
        )
        assert decision.confidence > 0.0

    def test_confidence_capped_at_one(self, router):
        """Confidence is capped at 1.0 for CODEGEN tasks with heuristics.

        The final_score = min(1.0, base_score + heuristic_boost) ensures
        confidence never exceeds 1.0.
        """
        router.classify_task = MagicMock(return_value=(TaskType.CODEGEN, 0.8, 0.15))
        decision = router.decide("write a function that also returns JSON")
        assert decision.confidence == 1.0


# ── Short-circuit edge case tests ──────────────────────────────────────────


class TestShortCircuit:
    def test_short_conversational_routes_local(self, router):
        """Prompts under 10 words without output demands should route LOCAL."""
        decision = router.decide("hi how are you")
        assert decision.target == RouteTarget.LOCAL
        assert "short conversational prompt" in decision.reason

    def test_short_prompt_task_type_overridden_to_factual(self, router):
        """Short prompts have their task_type overridden to FACTUAL."""
        decision = router.decide("hello world")
        assert decision.task_type == TaskType.FACTUAL
        assert "short conversational prompt" in decision.reason

    def test_short_but_with_output_demand_bypasses_short_circuit(self, router):
        """Short prompts WITH output demands should NOT take the short-circuit."""
        decision = router.decide("write code")
        assert "short conversational prompt" not in decision.reason

    def test_short_debug_prompt_bypasses_short_circuit(self, router):
        """Short prompts with 'debug this' should bypass short-circuit."""
        decision = router.decide("debug this")
        assert "short conversational prompt" not in decision.reason

    def test_short_codegen_prompt_bypasses_short_circuit(self, router):
        """Short prompts with 'implement' should bypass short-circuit."""
        decision = router.decide("implement sort")
        assert "short conversational prompt" not in decision.reason

    def test_short_prompt_confidence_is_zero(self, router):
        """Short-circuited prompts should have confidence=0.0."""
        decision = router.decide("howdy")
        assert decision.confidence == 0.0

    def test_10_word_prompt_does_not_short_circuit(self, router):
        """Prompts with >=10 words should use full routing logic."""
        prompt = "this is a ten word prompt that will not short"
        words = prompt.split()
        assert len(words) == 10, f"Expected 10 words, got {len(words)}: {words}"
        decision = router.decide(prompt)
        assert "short conversational prompt" not in decision.reason

    def test_9_word_prompt_short_circuits(self, router):
        """Prompts with <10 words (no output demands) should short-circuit."""
        prompt = "this prompt has nine words that should short circuit"
        words = prompt.split()
        assert len(words) == 9, f"Expected 9 words, got {len(words)}"
        decision = router.decide(prompt)
        assert "short conversational prompt" in decision.reason


# ── Integration-style tests ────────────────────────────────────────────────


class TestRouterIntegration:
    def test_router_with_different_thresholds_does_not_crash(self):
        """Adjusting the local threshold should work without errors."""
        low = Settings(router_local_threshold=0.1, router_heuristic_weight=0.3)
        high = Settings(router_local_threshold=0.9, router_heuristic_weight=0.3)
        r_low = SemanticHeuristicRouter(MockEmbedder(), low)
        r_high = SemanticHeuristicRouter(MockEmbedder(), high)
        prompt = "explain what this code does with proper formatting"
        d_low = r_low.decide(prompt)
        d_high = r_high.decide(prompt)
        assert isinstance(d_low, RoutingDecision)
        assert isinstance(d_high, RoutingDecision)

    def test_heuristic_weight_zero_means_only_semantic_matters(self, settings):
        """Zero heuristic weight means only semantic classification counts."""
        settings.router_heuristic_weight = 0.0
        router = SemanticHeuristicRouter(MockEmbedder(), settings)
        with patch.object(router, 'classify_task', return_value=(TaskType.CODEGEN, 0.8, 0.15)):
            decision = router.decide("write a function with code")
            assert decision.confidence == 1.0

    def test_reason_contains_task_type_and_target(self, router):
        """The reason string should contain useful routing information."""
        decision = router.decide(
            "what is the capital of france please tell me the answer"
        )
        assert (decision.task_type.value in decision.reason.lower()
                or decision.target.value in decision.reason.lower())

    def test_short_circuit_reason_mentions_short(self, router):
        """Short-circuited decisions should mention 'short' in reason."""
        decision = router.decide("hi")
        assert "short" in decision.reason


# ── Edge case: Settings beyond normal ranges ──────────────────────────────


class TestEdgeCases:
    def test_threshold_zero_means_all_remote(self):
        """Threshold of 0 means non-short-circuited prompts route REMOTE."""
        s = Settings(router_local_threshold=0.0, router_heuristic_weight=0.3)
        r = SemanticHeuristicRouter(MockEmbedder(), s)
        with patch.object(r, 'classify_task', return_value=(TaskType.CODEGEN, 0.8, 0.15)):
            decision = r.decide("what is the capital of france please tell me the answer today")
            assert decision.target == RouteTarget.REMOTE

    def test_threshold_one_keeps_factual_local(self):
        """Threshold of 1 means factual routes LOCAL (final_score < 1)."""
        s = Settings(router_local_threshold=1.0, router_heuristic_weight=0.3)
        r = SemanticHeuristicRouter(MockEmbedder(), s)
        decision = r.decide("what is the capital of france please tell me")
        assert decision.target == RouteTarget.LOCAL

    def test_heuristic_weight_above_one_produces_valid_decision(self):
        """Heuristic weight > 1 should still produce a valid routing decision."""
        s = Settings(router_local_threshold=0.4, router_heuristic_weight=2.0)
        r = SemanticHeuristicRouter(MockEmbedder(), s)
        decision = r.decide("write a python function with code please")
        assert isinstance(decision, RoutingDecision)
        assert 0.0 <= decision.confidence

    def test_same_prompt_same_behavior(self, keyword_router):
        """Repeated calls with same prompt should give same routing behavior."""
        prompt = "please explain how transformers work in machine learning"
        d1 = keyword_router.decide(prompt)
        d2 = keyword_router.decide(prompt)
        assert d1.target == d2.target
        assert d1.task_type == d2.task_type
        assert d1.confidence == d2.confidence

    def test_threshold_boundary_routing(self):
        """Test routing decisions at various threshold values."""
        s = Settings(router_local_threshold=0.5, router_heuristic_weight=0.3)
        r = SemanticHeuristicRouter(MockEmbedder(), s)

        # Use 10+ word prompts to avoid short-circuit
        prompt = "this is a test prompt that has enough words to avoid short circuit"

        with patch.object(r, 'classify_task', return_value=(TaskType.FACTUAL, 0.8, 0.15)):
            d = r.decide(prompt)
            assert d.target == RouteTarget.LOCAL

        with patch.object(r, 'classify_task', return_value=(TaskType.CODEGEN, 0.8, 0.15)):
            d = r.decide(prompt)
            assert d.target == RouteTarget.REMOTE

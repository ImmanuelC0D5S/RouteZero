from __future__ import annotations

import logging
import re

import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from routezero.config import Settings, TaskType, RouteTarget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category prototype examples — 12-15 per category, diverse phrasing
# Covers different difficulty levels and surface forms of each task type
# ---------------------------------------------------------------------------

CATEGORY_EXAMPLES: dict[TaskType, list[str]] = {
    TaskType.FACTUAL: [
        "explain how photosynthesis works",
        "what is a transformer model in machine learning",
        "define recursion in computer science",
        "what is the capital of France",
        "how does the internet work",
        "what does API stand for",
        "describe the water cycle",
        "what is the difference between RAM and ROM",
        "explain what a neural network is",
        "what is quantum computing",
        "how does GPS work",
        "what is the definition of machine learning",
        "describe how vaccines work",
        "what is blockchain technology",
        "explain the concept of gravity",
        "when should I use a hash map in programming",
        "teach me how to use vectors in C++",
        "explain when and where to use binary search trees",
        "how do I know when to use dynamic programming",
        "what is the difference between a stack and a queue",
    ],
    TaskType.MATH: [
        "calculate 15% of 340",
        "project revenue growing at 8% annually over 5 years",
        "solve for x in 3x + 7 = 22",
        "what is 144 divided by 12",
        "compute compound interest on 10000 at 5% for 3 years",
        "find the average of 23 45 67 89 12",
        "what is the square root of 256",
        "calculate the area of a circle with radius 7",
        "if a train travels 60 mph for 2.5 hours how far does it go",
        "what is 35 percent of 980",
        "solve this word problem: 3 workers finish a job in 8 days how long for 6 workers",
        "convert 75 fahrenheit to celsius",
        "what is 2 to the power of 10",
        "find the perimeter of a rectangle 12 by 8",
        "calculate the median of 5 3 8 1 9 2 7",
    ],
    TaskType.SENTIMENT: [
        "classify the sentiment of this text as positive negative or neutral",
        "is this product review positive or negative",
        "what is the emotional tone of this passage",
        "label the sentiment and justify the classification",
        "determine if this customer feedback is positive neutral or negative",
        "analyse the sentiment expressed in this tweet",
        "is this comment expressing approval or disapproval",
        "identify whether this review is favorable or unfavorable",
        "what emotion does this text convey",
        "rate the sentiment of this paragraph on a scale",
        "does this sentence express a positive or negative opinion",
        "classify the mood of this message",
        "is the author happy or unhappy based on this text",
        "identify the overall sentiment polarity of this review",
    ],
    TaskType.SUMMARIZE: [
        "summarize the following text in one sentence",
        "condense this passage to 50 words",
        "give me a tldr of this article",
        "provide a brief summary of this document",
        "summarize the key points of this text",
        "write a one paragraph summary of the following",
        "condense this passage to a specific format or length constraint",
        "give me the main takeaways from this article in bullet points",
        "shorten this text while preserving the key information",
        "what is the main idea of this passage",
        "summarize this in 3 sentences or fewer",
        "extract the most important information from this text",
        "give a concise overview of the following content",
        "reduce this article to its core argument",
    ],
    TaskType.NER: [
        "extract all person and location entities from this text",
        "identify all organizations mentioned in this passage",
        "find and label all named entities including person org location and date",
        "extract person organization location and date from this document",
        "identify all proper nouns and classify them by type",
        "list all the people places and companies mentioned",
        "tag all named entities in this text",
        "find all mentions of dates times and locations",
        "extract entity names and categorize them",
        "identify all the named entities and their types",
        "pull out all person names from this paragraph",
        "find every organization mentioned and label it",
        "extract all geographic locations from this text",
        "identify and classify every named entity in this passage",
    ],
    TaskType.DEBUG: [
        "find the bug in this code snippet and provide a fix",
        "why does this function return None unexpectedly",
        "identify the error in this Python code and correct it",
        "this code is throwing an index out of range error fix it",
        "what is wrong with this implementation",
        "debug this function and provide the corrected implementation",
        "why is this returning the wrong value",
        "this loop is running infinitely find the issue",
        "my code crashes with a key error find why",
        "identify and fix the logic error in this function",
        "this recursive function hits maximum recursion depth fix it",
        "why is this SQL query returning incorrect results",
        "find the off by one error in this code",
        "this async function is deadlocking explain why and fix it",
        "identify the memory leak in this code",
    ],
    TaskType.REASONING: [
        "if A is greater than B and B is greater than C what can we deduce about A and C",
        "all conditions must be satisfied determine which option is valid",
        "solve this logic puzzle where every clue must hold simultaneously",
        "deduce the correct answer from these constraints step by step",
        "given the following premises what conclusion can be drawn",
        "reason through this problem using deductive logic",
        "if all X are Y and some Y are Z what follows",
        "prove that this solution satisfies all the given requirements",
        "determine which statement must be true given these conditions",
        "work through this constraint satisfaction problem",
        "given these clues identify the only possible arrangement",
        "explain step by step why this argument is valid or invalid",
        "using logical deduction find the answer to this puzzle",
        "what can be inferred from these statements using first order logic",
        "identify the flaw in this logical argument",
    ],
    TaskType.CODEGEN: [
        "write a Python function that finds the maximum path sum in a binary tree",
        "implement a recursive algorithm using dynamic programming with memoization",
        "write a function that handles edge cases and returns the correct output",
        "implement a class with methods for insertion deletion and search",
        "create a script that reads a CSV file and outputs summary statistics",
        "write a function to perform binary search on a sorted array",
        "implement a graph traversal algorithm using breadth first search",
        "write correct well structured functions from this specification",
        "create a REST API endpoint that validates input and returns JSON",
        "implement a sorting algorithm from scratch with time complexity analysis",
        "write a Python decorator that adds retry logic with exponential backoff",
        "implement a thread safe queue data structure",
        "write a function that parses and validates an email address",
        "create a class that implements the observer design pattern",
        "write a generator function that lazily evaluates a sequence",
    ],
}

# Task types that should ALWAYS go remote — never downgrade to local
REMOTE_TASK_TYPES: set[TaskType] = {TaskType.REASONING, TaskType.CODEGEN, TaskType.DEBUG}

# Task types that should ALWAYS stay local — never upgrade to remote
LOCAL_TASK_TYPES: set[TaskType] = {
    TaskType.FACTUAL, TaskType.MATH, TaskType.SENTIMENT,
    TaskType.SUMMARIZE, TaskType.NER,
}

# Minimum similarity gap between top-2 classifications to trust the result
# If gap < this, classification is ambiguous → escalate to REMOTE
_CONFIDENCE_GAP_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Heuristic patterns — weighted by signal strength
# ---------------------------------------------------------------------------

# (pattern, score) — stronger signals score higher
_CONSTRAINT_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"must satisfy", re.IGNORECASE), 0.25),
    (re.compile(r"all conditions", re.IGNORECASE), 0.25),
    (re.compile(r"ensure that", re.IGNORECASE), 0.20),
    (re.compile(r"must handle", re.IGNORECASE), 0.20),
    (re.compile(r"step by step", re.IGNORECASE), 0.15),
    (re.compile(r"include a full explanation", re.IGNORECASE), 0.20),
    (re.compile(r"prove that", re.IGNORECASE), 0.25),
    (re.compile(r"deduce", re.IGNORECASE), 0.20),
]

_OUTPUT_DEMAND_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"write a function", re.IGNORECASE), 0.35),
    (re.compile(r"write a python", re.IGNORECASE), 0.35),
    (re.compile(r"\bimplement\b", re.IGNORECASE), 0.30),
    (re.compile(r"create a (script|class|module|api)", re.IGNORECASE), 0.30),
    (re.compile(r"write code", re.IGNORECASE), 0.30),
    (re.compile(r"find the bug", re.IGNORECASE), 0.35),
    (re.compile(r"debug this", re.IGNORECASE), 0.35),
    (re.compile(r"return JSON", re.IGNORECASE), 0.20),
    (re.compile(r"fix (this|the) (code|function|error|bug)", re.IGNORECASE), 0.35),
]

_TEACHING_PATTERNS: list[re.Pattern] = [
    re.compile(r"teach me", re.IGNORECASE),
    re.compile(r"when (should|do) I use", re.IGNORECASE),
    re.compile(r"how (do I|should I|can I) (use|learn|understand)", re.IGNORECASE),
    re.compile(r"explain (when|how|where) to use", re.IGNORECASE),
]

# Math signal patterns — if detected, skip conversational short-circuit
# These catch prompts like "solve for x: 2x+15=35" that look short but are math tasks
_MATH_SIGNAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"solve for", re.IGNORECASE),
    re.compile(r"calculate", re.IGNORECASE),
    re.compile(r"\bequation\b", re.IGNORECASE),
    re.compile(r"\bcompute\b", re.IGNORECASE),
    re.compile(r"\bformula\b", re.IGNORECASE),
    re.compile(r"\bpercentage\b", re.IGNORECASE),
    re.compile(r"\bderivative\b", re.IGNORECASE),
    re.compile(r"\bintegral\b", re.IGNORECASE),
    re.compile(r"\bexponential\b", re.IGNORECASE),
    re.compile(r"\blogarithm\b", re.IGNORECASE),
]



@dataclass
class RoutingDecision:
    target: RouteTarget
    task_type: TaskType
    confidence: float
    reason: str


class SemanticHeuristicRouter:
    def __init__(self, embedder: SentenceTransformer, settings: Settings) -> None:
        self.embedder = embedder
        self.settings = settings
        self.prototypes: dict[TaskType, np.ndarray] = {}
        self._build_prototypes()

    def _build_prototypes(self) -> None:
        """Pre-compute category centroids from examples at startup."""
        for task_type, examples in CATEGORY_EXAMPLES.items():
            embeddings = np.array([self.embedder.encode(ex) for ex in examples])
            centroid = embeddings.mean(axis=0)
            # L2-normalize for pure cosine distance computation later
            norm = np.linalg.norm(centroid)
            self.prototypes[task_type] = centroid / (norm + 1e-12)
        logger.debug("Router prototypes built for %d categories", len(self.prototypes))

    def _cosine_similarities(self, prompt: str) -> dict[TaskType, float]:
        """Return cosine similarity of prompt against all category centroids."""
        vec = self.embedder.encode(prompt)
        norm = np.linalg.norm(vec)
        vec = vec / (norm + 1e-12)
        return {
            task_type: float(np.dot(vec, centroid))
            for task_type, centroid in self.prototypes.items()
        }

    def classify_task(self, prompt: str) -> tuple[TaskType, float, float]:
        """
        Returns (best_task_type, best_similarity, confidence_gap).
        confidence_gap = difference between top-1 and top-2 similarity scores.
        A small gap means ambiguous classification.
        """
        sims = self._cosine_similarities(prompt)
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best_type, best_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        gap = best_sim - second_sim
        return best_type, best_sim, gap

    def heuristic_signals(self, prompt: str) -> float:
        """
        Returns a weighted heuristic complexity score in [0.0, 1.0].
        Higher = more complex = stronger push toward REMOTE.
        """
        score: float = 0.0

        # Code block detection
        if "```" in prompt:
            score += 0.35
        elif any(line.startswith("    ") for line in prompt.splitlines()):
            score += 0.20

        # Numeric density (math-heavy but not necessarily complex)
        digit_count = sum(c.isdigit() for c in prompt)
        numeric_density = digit_count / max(len(prompt), 1)
        if numeric_density > 0.05:
            score += 0.10

        # Prompt length modulation — longer prompts are generally harder
        word_count = len(prompt.split())
        if word_count > 80:
            score += 0.20
        elif word_count > 40:
            score += 0.10

        # Weighted constraint patterns
        for pat, weight in _CONSTRAINT_PATTERNS:
            if pat.search(prompt):
                score += weight

        # Weighted output demand patterns
        for pat, weight in _OUTPUT_DEMAND_PATTERNS:
            if pat.search(prompt):
                score += weight

        return max(0.0, min(1.0, score))

    def decide(self, prompt: str) -> RoutingDecision:
        # ----------------------------------------------------------------
        # 1. Short-circuit: short conversational prompts → always LOCAL
        # ----------------------------------------------------------------
        word_count = len(prompt.split())
        has_output_demand = any(pat.search(prompt) for pat, _ in _OUTPUT_DEMAND_PATTERNS)
        has_math_signal = any(pat.search(prompt) for pat in _MATH_SIGNAL_PATTERNS)

        if word_count < 10 and not has_output_demand and not has_math_signal:
            logger.debug("ROUTER short-circuit: conversational prompt → LOCAL")
            return RoutingDecision(
                target=RouteTarget.LOCAL,
                task_type=TaskType.FACTUAL,
                confidence=0.0,
                reason="short conversational prompt → LOCAL",
            )

        # ----------------------------------------------------------------
        # 2. Semantic classification with confidence gap check
        # ----------------------------------------------------------------
        task_type, best_sim, gap = self.classify_task(prompt)

        # ----------------------------------------------------------------
        # 3. LOCAL task type guard — these never go remote
        # ----------------------------------------------------------------
        if task_type in LOCAL_TASK_TYPES:
            heuristic_boost = self.heuristic_signals(prompt) * self.settings.router_heuristic_weight
            final_score = min(1.0, heuristic_boost)  # base_score = 0, no remote escalation

            logger.debug(
                "ROUTER — task:%s sim:%.3f gap:%.3f boost:%.3f final:%.3f → LOCAL (locked)",
                task_type, best_sim, gap, heuristic_boost, final_score,
            )
            return RoutingDecision(
                target=RouteTarget.LOCAL,
                task_type=task_type,
                confidence=final_score,
                reason=f"{task_type.value} task (local-locked) → LOCAL",
            )

        # ----------------------------------------------------------------
        # 4. REMOTE task type — check ambiguity and heuristics
        # ----------------------------------------------------------------
        # Ambiguous classification → escalate to REMOTE (safe default)
        if gap < _CONFIDENCE_GAP_THRESHOLD:
            logger.debug(
                "ROUTER — task:%s sim:%.3f gap:%.3f AMBIGUOUS → REMOTE",
                task_type, best_sim, gap,
            )
            return RoutingDecision(
                target=RouteTarget.REMOTE,
                task_type=task_type,
                confidence=best_sim,
                reason=f"ambiguous classification (gap={gap:.3f}) → REMOTE",
            )

        heuristic_boost = self.heuristic_signals(prompt) * self.settings.router_heuristic_weight
        final_score = min(1.0, 1.0 + heuristic_boost)  # base_score = 1.0 for REMOTE types

        target = (
            RouteTarget.LOCAL
            if final_score < self.settings.router_local_threshold
            else RouteTarget.REMOTE
        )

        parts = [f"{task_type.value} task (sim={best_sim:.3f}, gap={gap:.3f})"]
        if heuristic_boost > 0:
            parts.append(f"heuristic boost={heuristic_boost:.3f}")
        parts.append(f"→ {target.value}")

        logger.debug(
            "ROUTER — task:%s sim:%.3f gap:%.3f boost:%.3f final:%.3f threshold:%.2f → %s",
            task_type, best_sim, gap, heuristic_boost, final_score,
            self.settings.router_local_threshold, target.value,
        )

        return RoutingDecision(
            target=target,
            task_type=task_type,
            confidence=final_score,
            reason=", ".join(parts),
        )
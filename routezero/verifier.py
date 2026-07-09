from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from routezero.config import TaskType


@dataclass
class VerificationResult:
    passed: bool
    failure_reason: str | None


_CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_STEP_MARKERS = re.compile(
    r"\b(Step|Therefore|Because|First|Then|Finally)\b", re.IGNORECASE
)
_REFUSAL_PATTERNS = [
    re.compile(r"I cannot", re.IGNORECASE),
    re.compile(r"I['\u2019]m unable", re.IGNORECASE),
    re.compile(r"sorry,?\s+but", re.IGNORECASE),
]

# ── Task-type keyword maps (lightweight, no embedder) ──────────────────────

_CODE_KEYWORDS = [
    "write", "implement", "code", "function", "class", "method",
    "program", "script", "def ", "import ", "return ", "lambda",
    "async ", "await ", "yield ", "decorator", "generator",
]
_DEBUG_KEYWORDS = [
    "debug", "fix", "bug", "error", "issue", "crash", "traceback",
    "exception", "stack trace", "not working", "broken",
]
_MATH_KEYWORDS = [
    "calculate", "compute", "solve", "equation", "formula",
    "derivative", "integral", "sum", "product", "matrix",
    "vector", "probability", "statistics",
]
_SENTIMENT_KEYWORDS = [
    "sentiment", "positive", "negative", "neutral", "opinion",
    "feeling", "tone", "attitude", "review", "feedback",
]


def detect_task_type(prompt: str) -> TaskType:
    text = prompt.strip().lower()

    for kw in _CODE_KEYWORDS:
        if kw in text:
            return TaskType.CODEGEN

    for kw in _DEBUG_KEYWORDS:
        if kw in text:
            return TaskType.DEBUG

    for kw in _MATH_KEYWORDS:
        if kw in text:
            return TaskType.MATH

    for kw in _SENTIMENT_KEYWORDS:
        if kw in text:
            return TaskType.SENTIMENT

    return TaskType.FACTUAL


# ── Individual verifiers ──────────────────────────────────────────────────


def verify_json(response: str) -> VerificationResult:
    try:
        json.loads(response)
    except json.JSONDecodeError:
        return VerificationResult(False, "Invalid JSON")
    return VerificationResult(True, None)


def verify_code(response: str) -> VerificationResult:
    match = _CODE_BLOCK_PATTERN.search(response)
    code = match.group(1) if match else response.strip()

    try:
        ast.parse(code)
    except SyntaxError:
        return VerificationResult(False, "Syntax error in code block")
    return VerificationResult(True, None)


def verify_reasoning(response: str) -> VerificationResult:
    markers = _STEP_MARKERS.findall(response)
    if len(markers) < 2:
        return VerificationResult(False, "Missing chain-of-thought reasoning")
    return VerificationResult(True, None)


def verify_generic(response: str) -> VerificationResult:
    stripped = response.strip()
    if not stripped:
        return VerificationResult(False, "Response is empty")
    if len(stripped) <= 10:
        return VerificationResult(False, "Response too short")
    for pat in _REFUSAL_PATTERNS:
        if pat.search(stripped):
            return VerificationResult(False, "Response contains refusal")
    return VerificationResult(True, None)


# ── Dispatcher ────────────────────────────────────────────────────────────


def verify(response: str, task_type: TaskType) -> VerificationResult:
    result: VerificationResult | None = None

    if task_type in (TaskType.CODEGEN, TaskType.DEBUG):
        result = verify_code(response)
    elif task_type == TaskType.REASONING:
        result = verify_reasoning(response)
    elif task_type in (
        TaskType.FACTUAL,
        TaskType.MATH,
        TaskType.SENTIMENT,
        TaskType.SUMMARIZE,
        TaskType.NER,
    ):
        result = verify_generic(response)

    if result is not None and not result.passed:
        return result

    stripped = response.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        json_result = verify_json(response)
        if not json_result.passed:
            return json_result

    return result if result is not None else VerificationResult(True, None)

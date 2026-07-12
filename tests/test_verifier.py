import pytest
from routezero.config import TaskType
from routezero.verifier import (
    verify, verify_json, verify_code, verify_reasoning, verify_generic,
    detect_task_type, VerificationResult
)

def test_verify_json_valid():
    result = verify_json('{"key": "value"}')
    assert result.passed is True

def test_verify_json_invalid():
    result = verify_json("{invalid json}")
    assert result.passed is False
    assert "Invalid JSON" in result.failure_reason

def test_verify_code_python():
    result = verify_code("```python\nx = 1\nprint(x)\n```")
    assert result.passed is True

def test_verify_code_syntax_error():
    result = verify_code("```python\nx = \n```")
    assert result.passed is False

def test_verify_reasoning_with_steps():
    result = verify_reasoning("First, we calculate X. Then we apply Y. Therefore Z.")
    assert result.passed is True

def test_verify_reasoning_no_steps():
    result = verify_reasoning("The answer is 42.")
    assert result.passed is False

def test_verify_generic_valid():
    result = verify_generic("This is a valid response with enough content.")
    assert result.passed is True

def test_verify_generic_empty():
    result = verify_generic("")
    assert result.passed is False

def test_verify_generic_refusal():
    result = verify_generic("I cannot answer that question.")
    assert result.passed is False

def test_verify_dispatcher_codegen():
    result = verify("```python\ndef foo(): pass\n```", TaskType.CODEGEN)
    assert result.passed is True

def test_verify_dispatcher_reasoning():
    result = verify("Step 1: Do X. Step 2: Do Y.", TaskType.REASONING)
    assert result.passed is True

def test_detect_task_type_code():
    assert detect_task_type("write a function that does X") == TaskType.CODEGEN

def test_detect_task_type_math():
    assert detect_task_type("calculate the sum of 5 and 3") == TaskType.MATH

def test_detect_task_type_default():
    assert detect_task_type("what is the weather today") == TaskType.FACTUAL

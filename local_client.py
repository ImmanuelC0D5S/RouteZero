"""
local_client.py — CPU-based local inference via llama.cpp (GGUF, quantized).

Per competition rules: local model inference is permitted, counts toward
accuracy, but records ZERO tokens toward the score. Correctly-answered
local tasks are the most token-efficient outcome possible.

Sized for the 4GB RAM / 2 vCPU grading environment: Qwen2.5-1.5B-Instruct,
4-bit quantized (~1.1GB on disk, comfortably fits alongside agent code and
the embedding model).
"""

import time

_llm = None  # lazy-loaded singleton — only load the model if actually used


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(
            model_path="./model_cache/local_llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            n_threads=2,
            verbose=False,
        )
    return _llm


def call_local(system_prompt: str, user_prompt: str, max_tokens: int) -> str | None:
    """
    Runs inference locally via llama.cpp.

    Returns the generated text, or None if local inference fails for any
    reason (model load error, generation error) — callers should treat
    None as "fall back to Fireworks", never let a local failure crash
    the whole pipeline.
    """
    try:
        llm = _get_llm()
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Local inference failed, will fall back to Fireworks: {e}")
        return None


if __name__ == "__main__":
    print("Testing local model...")
    start = time.monotonic()
    result = call_local("Answer concisely.", "What is 2 + 2?", 50)
    elapsed = time.monotonic() - start
    print(f"Result: {result}")
    print(f"Took {elapsed:.2f}s")
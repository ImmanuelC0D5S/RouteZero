"""
local_eval.py — compares your results.json against expected answers
for a hand-built local test set, so you get an approximate accuracy
signal before submitting. THIS IS THE INTIAL TEMP FILE EDIT LATER
"""

import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

def run_eval(results_path: str, expected_path: str):
    results = {r["task_id"]: r["answer"] for r in load_json(results_path)}
    expected = load_json(expected_path)  # [{"task_id": ..., "expected": ...}, ...]

    correct = 0
    for item in expected:
        tid = item["task_id"]
        actual = results.get(tid, "")
        # crude check for now — replace with something smarter per category
        # (exact match, keyword containment, or an LLM-judge call)
        is_correct = item["expected"].strip().lower() in actual.strip().lower()
        status = "✓" if is_correct else "✗"
        print(f"{tid:<8} {status}  expected~='{item['expected'][:40]}'  got='{actual[:40]}'")
        correct += is_correct

    print(f"\nRough accuracy: {correct}/{len(expected)} ({100*correct/len(expected):.1f}%)")

if __name__ == "__main__":
    run_eval("/output/results.json", "expected_answers.json")
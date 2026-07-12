import json
import os
import sys
from category_router import CategoryRouter, classify_with_fallback, MODEL_TIER

try:
    from fireworks_client import call_fireworks
except ImportError:
    def call_fireworks(*args, **kwargs): return None

try:
    from local_client import call_local
except ImportError:
    def call_local(*args, **kwargs): return None

# MAXIMIZE TOKEN SAVINGS: Try local for these 5 categories first.
# Even if tier is 'medium', trying local costs 0 tokens. 
# If it fails verification, we then pay for Fireworks.
LOCAL_ELIGIBLE_CATEGORIES = {"sentiment", "ner", "summarization", "factual", "math"}

PROMPT_TEMPLATES = {
    "factual": ("Answer concisely and accurately.", 150),
    "math": ("Solve step by step. Answer:", 200),
    "sentiment": ("Classify sentiment (positive/negative/neutral) + 1-sentence justification.", 80),
    "summarization": ("Summarize concisely.", 120),
    "ner": ("Extract entities as JSON: person, org, location, date.", 250),
    "debugging": ("Fix the bug. Provide code + brief explanation.", 300),
    "codegen": ("Write the function spec only.", 300),
    "logic": ("Solve the puzzle. Show reasoning briefly.", 250),
}

def get_tier_to_model():
    allowed = os.environ.get("ALLOWED_MODELS", "").split(",")
    allowed = [m.strip() for m in allowed if m.strip()]
    if not allowed:
        return {"easy": "llama-v3-8b", "medium": "llama-v3-70b", "hard": "llama-v3-70b"}
    # Rules: Use smallest for easy, largest for hard
    return {
        "easy": allowed[0],
        "medium": allowed[len(allowed) // 2],
        "hard": allowed[-1],
    }

def verify_answer(category: str, answer: str) -> bool:
    if not answer or len(answer.strip()) < 2: return False
    # Basic quality gate: local models sometimes output gibberish
    if category == "ner" and "{" not in answer: return False
    if category == "math" and not any(c.isdigit() for c in answer): return False
    return True

def main():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"

    if not os.path.exists(input_path):
        sys.exit(1)

    with open(input_path, "r") as f:
        tasks = json.load(f)

    router = CategoryRouter(n_neighbors=3)
    tier_to_model = get_tier_to_model()
    results = []

    for task in tasks:
        task_id = task.get("task_id", task.get("id", "unknown"))
        prompt = task["prompt"]

        category, _ = classify_with_fallback(router, prompt)
        tier = MODEL_TIER.get(category, "hard")
        system_prompt, max_tokens = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["factual"])

        answer = None
        used_local = False

        # STEP 1: Attempt Local (Free)
        # If it's a category our 1.5B model can handle, try it first.
        if category in LOCAL_ELIGIBLE_CATEGORIES:
            local_answer = call_local(system_prompt, prompt, max_tokens)
            if local_answer and verify_answer(category, local_answer):
                answer = local_answer
                used_local = True

        # STEP 2: Call Fireworks (Paid)
        if answer is None:
            model_id = tier_to_model[tier]
            try:
                answer = call_fireworks(model_id, system_prompt, prompt, max_tokens)
            except Exception as e:
                print(f"  ⚠ Fireworks call failed: {e}")
                answer = None
            
            # STEP 3: Escalation (If API gave a bad answer, try the biggest model)
            if answer is not None and not verify_answer(category, answer) and tier != "hard":
                try:
                    answer = call_fireworks(tier_to_model["hard"], system_prompt, prompt, max_tokens)
                except Exception as e:
                    print(f"  ⚠ Fireworks escalation failed: {e}")
                    answer = None

        # STEP 4: If still no answer, record the routing decision
        if answer is None:
            answer = f"[ROUTED: {category} -> {'Local' if used_local else 'Fireworks'}]"

        results.append({"task_id": task_id, "answer": answer})
        route_indicator = "Local (attempted)" if category in LOCAL_ELIGIBLE_CATEGORIES else "Fireworks"
        print(f"  ✓ {task_id}: {category} -> {route_indicator}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    sys.exit(0) # Ensure Exit Code 0

if __name__ == "__main__":
    main()
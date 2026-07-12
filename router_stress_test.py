# -*- coding: utf-8 -*-
"""
router_stress_test.py
Tests the routing decision for "solve for x: 2x+15=35" on both routing systems.
"""

import logging
import sys

print("=" * 70)
print("ROUTING STRESS TEST: \"solve for x: 2x+15=35\"")
print("=" * 70)

# ---------------------------------------------------------------------------
# TEST 1: category_router.py (used by main.py)
# ---------------------------------------------------------------------------
print("\n" + "#" * 70)
print("# TEST 1: category_router.py (used by main.py)")
print("#" * 70)

try:
    from category_router import CategoryRouter, classify_with_fallback, MODEL_TIER

    # LOCAL_ELIGIBLE_CATEGORIES from main.py
    LOCAL_ELIGIBLE_CATEGORIES = {"sentiment", "ner", "summarization", "factual"}

    print("\n[1a] Creating CategoryRouter (n_neighbors=3)...")
    router = CategoryRouter(n_neighbors=3, model_name="all-MiniLM-L6-v2")

    prompt = "solve for x: 2x+15=35"
    print(f"\n[1b] classify_with_fallback(router, {prompt!r})...")
    category, source = classify_with_fallback(router, prompt)
    tier = MODEL_TIER.get(category, "unknown")

    print(f"\n--- RESULTS (category_router) ---")
    print(f"  Prompt:     {prompt!r}")
    print(f"  Category:   {category}")
    print(f"  Source:     {source}")
    print(f"  Tier:       {tier}")

    if category in LOCAL_ELIGIBLE_CATEGORIES:
        print(f"  main.py →   LOCAL (category '{category}' IS in LOCAL_ELIGIBLE_CATEGORIES)")
    else:
        print(f"  main.py →   REMOTE (category '{category}' is NOT in LOCAL_ELIGIBLE_CATEGORIES)")

except Exception as e:
    print(f"\nERROR in Test 1: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 2: routezero/router.py (SemanticHeuristicRouter)
# ---------------------------------------------------------------------------
print("\n" + "#" * 70)
print("# TEST 2: routezero/router.py (SemanticHeuristicRouter)")
print("#" * 70)

try:
    from sentence_transformers import SentenceTransformer
    from routezero.config import Settings, RouteTarget, TaskType
    from routezero.router import SemanticHeuristicRouter

    # Enable debug logging to see router internals
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    print("\n[2a] Creating SentenceTransformer embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n[2b] Creating Settings()...")
    settings = Settings()

    print(f"  router_local_threshold:    {settings.router_local_threshold}")
    print(f"  router_heuristic_weight:   {settings.router_heuristic_weight}")

    print("\n[2c] Creating SemanticHeuristicRouter...")
    router2 = SemanticHeuristicRouter(embedder, settings)

    prompt = "solve for x: 2x+15=35"
    print(f"\n[2d] router2.decide({prompt!r})...\n")
    decision = router2.decide(prompt)

    print(f"\n--- RESULTS (SemanticHeuristicRouter) ---")
    print(f"  Prompt:     {prompt!r}")
    print(f"  Target:     {decision.target.value}")
    print(f"  Task Type:  {decision.task_type.value}")
    print(f"  Confidence: {decision.confidence}")
    print(f"  Reason:     {decision.reason}")

except Exception as e:
    print(f"\nERROR in Test 2: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

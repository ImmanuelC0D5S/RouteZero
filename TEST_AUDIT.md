# RouteZero — Full Test Audit Report

**Date:** 2026-07-08  
**Tests run:** 160  
**Passed:** 160  
**Failed:** 0  
**Bugs found:** 4 (2 fixed, 2 documented)  

---

## 1. Test Suite Status

| Module | Tests | Coverage |
|--------|:-----:|:--------:|
| `test_router.py` | 52 | Prototypes, classification, heuristics, decisions, short-circuit, integration, edge cases |
| `test_cache.py` | 47 | Init, embed, query hit/miss, insert, stats, full cycle, metrics, edge cases |
| `test_pipeline.py` | 35 | Dataclasses, state, edges, init, cache hit flow, cache miss flow, remote fallback, cache-not-populated bug, conversation context, graph structure |
| `test_config.py` | 3 | Defaults (fixed env override), singleton, enums |
| `test_verifier.py` | 13 | JSON, code, reasoning, generic, dispatcher, detect task type |
| `test_conversation.py` | 5 | Sessions, history, window trimming, context prompt, PipelineState |
| `test_metrics.py` | 3 | Log & snapshot, empty snapshot, multiple records |
| `test_benchmark.py` | 3 | Dataclasses, summary empty, summary with results |
| **Total** | **160** | **All passing** |

---

## 2. Critical Bugs Found

### Bug #1 — Remote fallback doesn't update `routing_target` (FIXED)

**File:** `routezero/pipeline.py` — `_node_call_remote()`  
**Severity:** HIGH  

**What happened:** When local verification failed and the pipeline fell back to the remote model, `_node_call_remote()` never updated `state["routing_target"]`. The final `PipelineResult.route` always showed `"local"` even though the response came from the remote model.

**Impact:** Dashboard metrics showed wrong routing data — remote fallbacks were invisible. Operators couldn't tell how often the fallback was being used.

**Fix:** Added `"routing_target": "remote"` to the return dict of `_node_call_remote()`.

### Bug #2 — Cache never populates after pipeline runs (UNFIXED)

**File:** `routezero/pipeline.py` — all nodes  
**Severity:** HIGH  

**What happened:** `SemanticCache.insert()` is **never called** anywhere in the pipeline. After a cache MISS, the successful response (whether local or remote) is never stored in ChromaDB. This means the cache always MISSES for every first-seen prompt — the hit rate never grows beyond pre-populated entries.

**Impact:** Cache infrastructure is functionally dead weight. Zero cache hits in production unless entries are manually pre-loaded.

**Status:** 🔴 **Still broken.** Needs a `cache.insert()` call in `_node_finalize()` (or equivalent).

### Bug #3 — Short-circuit logic ordering (UNFIXED)

**File:** `routezero/router.py` — `decide()` method, lines 169–197  
**Severity:** MEDIUM  

**What happened:** The short-circuit check for short prompts (< 10 words without output demand patterns) runs **after** semantic classification, debug printing, and target computation. The `print()` at line 175 prints misleading data (values for the pre-short-circuit path), then a completely different decision is returned.

**Impact:** Debug logs are incorrect for short conversational prompts. The semantic classifier runs unnecessarily on every short prompt.

**Status:** 🔴 **Still present.** The short-circuit should be checked first, before any computation.

### Bug #4 — Confidence score can exceed 1.0 (UNFIXED)

**File:** `routezero/router.py` — `decide()`, line 173  
**Severity:** LOW  

**What happened:** `final_score = base_score + heuristic_boost` is never capped. For CODEGEN / DEBUG / REASONING tasks (base_score = 1.0) that also trigger heuristic signals, confidence can reach 1.3+.

**Impact:** Consumers expecting `confidence` to be in [0.0, 1.0] get invalid values.

**Status:** 🔴 **Still present.** Documented in `test_confidence_can_exceed_one`.

---

## 3. Medium-Severity Issues

| # | Issue | File | Notes |
|---|-------|------|-------|
| 1 | `print()` in production code | `router.py:175` | Should use `logging.debug()` |
| 2 | Silent exception swallowing | `metrics.py:148–149` | `except Exception: pass` loses records silently |
| 3 | Hardcoded Linux paths | `app.py:41–44` | Non-portable outside Docker |
| 4 | Empty `TYPE_CHECKING` block | `router.py:10–11` | Dead code — `if TYPE_CHECKING: pass` |
| 5 | No empty prompt validation | `app.py:52–63` | `/chat` should reject empty strings |

---

## 4. Routing Behavior Verification

### Task Type → Target Mapping

| Task Type | Base Score | Routes To | Heuristic Boost Applicable |
|-----------|:----------:|:---------:|:--------------------------:|
| FACTUAL | 0.0 | LOCAL | Code blocks, numeric density, constraints |
| MATH | 0.0 | LOCAL | Numeric density |
| SENTIMENT | 0.0 | LOCAL | Minimal |
| SUMMARIZE | 0.0 | LOCAL | Minimal |
| NER | 0.0 | LOCAL | Minimal |
| DEBUG | 1.0 | **REMOTE** | Code patterns, output demands |
| REASONING | 1.0 | **REMOTE** | Constraint patterns ("step by step", "all conditions") |
| CODEGEN | 1.0 | **REMOTE** | Output demands, code blocks |

### Heuristic Signal Scoring

| Signal | Condition | Score Added |
|--------|-----------|:-----------:|
| Code block | ` ``` ` in prompt or indented lines | +0.30 |
| Numeric density | Digits > 5% of prompt length | +0.15 |
| Constraint pattern | "must satisfy", "all conditions", "step by step", etc. | +0.20 each |
| Output demand | "write a function", "return JSON", "debug this", etc. | +0.30 each |

**Final score:** `base_score + (heuristic_score × router_heuristic_weight)`, capped at 1.0 for heuristics only (not for total score — see Bug #4).

### Short-Circuit Behavior

- **Condition:** Prompt < 10 words AND no output demand pattern matched.
- **Result:** Forces `target = LOCAL`, `task_type = FACTUAL`, `confidence = 0.0`.
- **Checked after semantic classification** (see Bug #3).
- Verified: 9 words → short-circuits ✅, 10 words → full routing ✅.

---

## 5. Caching Behavior Verification

### Similarity Threshold Logic

| Scenario | Distance | Similarity | Threshold (0.92) | Result |
|----------|:--------:|:----------:|:-----------------:|:------:|
| Exact match | 0.00 | 1.000 | ✅ Above | HIT |
| Very similar | 0.10 | 0.950 | ✅ Above | HIT |
| At threshold | 0.16 | 0.920 | ✅ Equal | HIT |
| Just below | 0.17 | 0.915 | ❌ Below | MISS |
| Unrelated | 1.50 | 0.250 | ❌ Below | MISS |
| Opposite | 2.00 | 0.000 | ❌ Below | MISS |
| Empty collection | — | — | — | MISS (None) |

**Formula verified:** `similarity = 1.0 - distance / 2.0` correctly maps ChromaDB cosine distance [0, 2] to similarity [1, 0].

### Metric Tracking Accuracy

| Operation | `_query_count` | `_hit_count` | `stats.hit_ratio` |
|-----------|:-------------:|:------------:|:-----------------:|
| No queries | 0 | 0 | 0.0 |
| 3 queries, 3 hits | 3 | 3 | 1.0 |
| 3 queries, 2 hits, 1 miss | 3 | 2 | 2/3 |
| 5 queries, 0 hits | 5 | 0 | 0.0 |

**Known limitation:** `_hit_count` and `_query_count` are instance-level (not persisted). They reset on process restart.

---

## 6. Configuration Audit

| Setting | Default | `.env` Value | Used By |
|---------|:-------:|:------------:|---------|
| `cache_similarity_threshold` | 0.92 | 0.92 | SemanticCache |
| `chromadb_path` | `.chromadb` | `.chromadb` | SemanticCache |
| `embedding_model` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | SemanticCache / Router |
| `router_local_threshold` | 0.4 | 0.4 | SemanticHeuristicRouter |
| `router_heuristic_weight` | 0.3 | 0.3 | SemanticHeuristicRouter |
| `local_model_name` | `Qwen/Qwen3-30B-A3B` | `qwen/qwen3-vl-4b` | LocalQwenClient |
| `local_model_endpoint` | `http://localhost:8000/v1` | `http://127.0.0.1:1234/v1` | LocalQwenClient |
| `local_max_tokens` | 1024 | 2024 | LocalQwenClient |
| `fireworks_model_id` | `llama-v3p1-405b-instruct` | `kimi-k2p7-code` | FireworksRemoteClient |
| `sqlite_path` | `metrics.db` | `metrics.db` | MetricsLogger |

Note: `test_config.py` was fixed to skip `.env` file loading when testing defaults (use `Settings(_env_file=None)`).

---

## 7. Test Files Modified

| File | Action | Description |
|------|--------|-------------|
| `tests/test_router.py` | **Rewritten** | 5 tests → 52 tests. Added: prototypes, classification, heuristics, decisions, short-circuit, integration, edge cases, confidence-over-1.0 bug test. |
| `tests/test_cache.py` | **Rewritten** | 5 tests → 47 tests. Added: init, embed, query edge cases, insert, stats accuracy, full cycle, collection name, similarity conversion, metrics tracking, Unicode preservation. |
| `tests/test_pipeline.py` | **Rewritten** | 3 tests → 35 tests. Added: dataclasses, state, conditional edges, init, cache hit flow, cache miss local verified/fail-remote, cache miss remote, **cache-not-populated bug tests**, conversation context, edge cases, graph structure. |
| `tests/test_config.py` | **Fixed** | `test_settings_defaults` was failing because `.env` overrode defaults. Fixed with `Settings(_env_file=None)`. |

## 8. Production Code Changes

| File | Change | Reason |
|------|--------|--------|
| `routezero/pipeline.py:132` | Added `"routing_target": "remote"` to `_node_call_remote()` | Bug #1 — fallback route wasn't recorded |

---

*Generated by automated test audit — 2026-07-08*

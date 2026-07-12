from routezero.benchmark import (
    BenchmarkTask, BenchmarkResult, BenchmarkSummary, compute_summary
)

def test_benchmark_dataclasses():
    task = BenchmarkTask("t1", "test prompt", "factual")
    assert task.task_id == "t1"
    
    result = BenchmarkResult("t1", "response", "local", False, True, 100.0, "factual")
    assert result.route == "local"
    assert result.latency_ms == 100.0

def test_compute_summary_empty():
    summary = compute_summary([])
    assert summary.total_tasks == 0

def test_compute_summary_with_results():
    results = [
        BenchmarkResult("t1", "a", "local", False, True, 50.0, "factual"),
        BenchmarkResult("t2", "b", "remote", False, True, 200.0, "codegen"),
        BenchmarkResult("t3", "c", "cache", True, True, 5.0, "factual"),
    ]
    summary = compute_summary(results)
    assert summary.total_tasks == 3
    assert summary.cache_hit_count == 1
    assert summary.local_route_count == 1
    assert summary.remote_route_count == 1
    assert summary.quality_retained_pct == 100.0

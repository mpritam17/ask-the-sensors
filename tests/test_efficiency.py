from ats.efficiency import benchmark_callable


def test_benchmark_enforces_run_count_and_returns_percentiles():
    result = benchmark_callable(lambda: sum(range(20)), warmup=1, runs=30)
    assert result["runs"] == 30
    assert result["median_latency_ms"] >= 0
    assert result["p95_latency_ms"] >= result["median_latency_ms"]

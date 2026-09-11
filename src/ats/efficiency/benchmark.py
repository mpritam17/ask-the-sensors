"""Repeatable warmed-up latency and peak-memory measurements."""
from __future__ import annotations

import platform
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict

import numpy as np


def benchmark_callable(
    function: Callable[[], object], *, warmup: int = 5, runs: int = 30
) -> Dict[str, float | int]:
    if warmup < 0 or runs < 30:
        raise ValueError("efficiency results require at least 30 measured runs")
    for _ in range(warmup):
        function()
    tracemalloc.start()
    timings = []
    for _ in range(runs):
        started = perf_counter()
        function()
        timings.append((perf_counter() - started) * 1000.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "runs": int(runs),
        "warmup_runs": int(warmup),
        "median_latency_ms": float(statistics.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "peak_python_memory_mb": float(peak / (1024 ** 2)),
    }


def estimator_parameter_count(estimator) -> int:
    if hasattr(estimator, "named_steps"):
        estimator = estimator.named_steps.get("model", estimator)
    if hasattr(estimator, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in estimator.estimators_))
    total = 0
    for name in ("coef_", "intercept_"):
        value = getattr(estimator, name, None)
        if value is not None:
            total += int(np.asarray(value).size)
    return total


def artifact_metadata(path: Path, estimator) -> Dict[str, object]:
    path = Path(path)
    return {
        "model_disk_bytes": int(path.stat().st_size),
        "model_parameter_or_tree_node_count": estimator_parameter_count(estimator),
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

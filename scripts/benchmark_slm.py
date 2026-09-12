#!/usr/bin/env python3
"""Download/load Qwen and benchmark validated Task 4 inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import resolve  # noqa: E402
from ats.efficiency import benchmark_callable  # noqa: E402
from ats.qa import answer_open_world  # noqa: E402
from ats.slm import GroundedQwen, GroundingError  # noqa: E402
from ats.timeline import ActivityInterval  # noqa: E402


def cache_size(model_name: str) -> int | None:
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name:
                return int(repo.size_on_disk)
    except Exception:  # cache accounting is supplementary, not required to run
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="artifacts/slm_efficiency.json")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    import psutil
    import torch

    timeline = [
        ActivityInterval(
            "sitting", 0.0, 12.0, 0.91, "accelerometer",
            ["Acc X", "Acc Y", "Acc Z"], {"acc_rms_g": 1.01},
        ),
        ActivityInterval(
            "running", 16.0, 24.0, 0.88, "both", ["All"],
            {"acc_rms_g": 1.82, "gyro_rms_rad_s": 0.71},
        ),
    ]
    question = "Was there any strenuous activity?"
    runner = GroundedQwen(args.model)
    load_started = perf_counter()
    runner._load()  # isolate one-time loading from warmed inference latency
    load_seconds = perf_counter() - load_started
    model = runner._model

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    validation_rejections = 0
    last_rejection = None

    def validated_task4():
        nonlocal validation_rejections, last_rejection
        try:
            return runner.answer(question, timeline)
        except GroundingError as exc:
            # This is the production safety contract: rejected generation can
            # never reach output, and the deterministic grounded rule answers.
            validation_rejections += 1
            last_rejection = str(exc)
            return answer_open_world(question, timeline)

    first = validated_task4()
    benchmark = benchmark_callable(
        validated_task4, runs=args.runs
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    process = psutil.Process()
    payload = {
        "dataset_kind": "real_model_runtime",
        "model": args.model,
        "load_seconds": float(load_seconds),
        "parameter_count": int(parameter_count),
        "cache_bytes": cache_size(args.model),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if torch.cuda.is_available() else None
        ),
        "process_rss_mb_after_benchmark": float(process.memory_info().rss / (1024 ** 2)),
        "validated_answer": {
            "answer": first.answer,
            "activity_event": first.activity_event,
            "timestamps": first.timestamps,
            "modality": first.modality,
            "channels": first.channels,
            "explanation": first.explanation,
        },
        "validation": {
            "model_calls": int(args.runs + 6),
            "rejected_generations": int(validation_rejections),
            "fallback_used": bool(validation_rejections),
            "last_rejection": last_rejection,
        },
        "inference": benchmark,
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

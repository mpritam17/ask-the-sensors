#!/usr/bin/env python3
"""Benchmark recognition and deterministic QA with at least 30 warmed runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import resolve  # noqa: E402
from ats.efficiency import artifact_metadata, benchmark_callable  # noqa: E402
from ats.features import extract_features, select_engineered_features  # noqa: E402
from ats.qa import answer_question  # noqa: E402
from ats.recognition import load_bundle, predict_probabilities  # noqa: E402
from ats.timeline import build_timeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recording-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    model_path = Path(args.model)
    bundle = load_bundle(model_path)
    with np.load(args.recording_npz, allow_pickle=True) as data:
        valid = np.asarray(data["valid"], bool)
        windows = np.asarray(data["windows"])[valid][:256]
        starts = np.asarray(data["t_start"], float)[valid][:256]
        ends = np.asarray(data["t_end"], float)[valid][:256]
    features = select_engineered_features(
        extract_features(windows), bundle["feature_names"]
    )

    recognition = benchmark_callable(
        lambda: predict_probabilities(bundle, features), runs=args.runs
    )
    probabilities = predict_probabilities(bundle, features)
    timeline = build_timeline(
        probabilities,
        starts,
        ends,
        bundle["classes"],
        available_modalities=bundle.get("available_modalities"),
    )
    questions = [
        "What was the main activity?",
        "How long did the person walk?",
        "How many times did the person run?",
        "When did bicycling begin?",
    ]
    deterministic_qa = benchmark_callable(
        lambda: [answer_question(question, timeline) for question in questions],
        runs=args.runs,
    )
    payload = {
        "dataset_kind": bundle["dataset_kind"],
        "artifact": artifact_metadata(model_path, bundle["estimator"]),
        "recognition_batch": {**recognition, "windows_per_run": int(len(windows))},
        "tasks_1_3_batch": {**deterministic_qa, "questions_per_run": len(questions)},
        "task_4": {"status": "not measured; optional model dependencies/weights required"},
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    if bundle["dataset_kind"] != "real":
        print("*** SYNTHETIC INPUT BENCHMARK - NOT A REAL-DATA ACCURACY RESULT ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

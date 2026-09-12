#!/usr/bin/env python3
"""Evaluate recognition, deterministic QA, grounding, strictness, and robustness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.eval.metrics import interval_iou, normalize_answer  # noqa: E402
from ats.features import extract_features, select_engineered_features  # noqa: E402
from ats.qa import answer_question  # noqa: E402
from ats.recognition import load_bundle, predict_probabilities  # noqa: E402
from ats.timeline import build_timeline  # noqa: E402


def load_users(processed: Path, users, cap_per_class=None, seed=0):
    rng = np.random.default_rng(seed)
    for uuid in users:
        with np.load(processed / f"{uuid}.npz", allow_pickle=True) as data:
            valid = np.asarray(data["valid"], bool)
            labels = np.asarray(data["y"], int)[valid]
            selected = np.arange(len(labels))
            if cap_per_class is not None:
                chosen = []
                for label in np.unique(labels):
                    indices = np.flatnonzero(labels == label)
                    if len(indices) > cap_per_class:
                        indices = rng.choice(indices, cap_per_class, replace=False)
                    chosen.extend(indices.tolist())
                selected = np.asarray(sorted(chosen), dtype=int)
            yield {
                "uuid": uuid,
                "windows": np.asarray(data["windows"])[valid][selected],
                "y": labels[selected],
                "t_start": np.asarray(data["t_start"], float)[valid][selected],
                "t_end": np.asarray(data["t_end"], float)[valid][selected],
                "available_modalities": (
                    [str(item) for item in data["modalities"]]
                    if "modalities" in data.files
                    else ["accelerometer", "gyroscope"]
                ),
            }


def timeline_from_labels(block, classes):
    probabilities = np.eye(len(classes), dtype=float)[block["y"]]
    return build_timeline(
        probabilities, block["t_start"], block["t_end"], classes,
        smoothing_windows=1, confidence_threshold=0.5,
        available_modalities=block["available_modalities"],
    )


def _features_for_bundle(windows, bundle):
    return select_engineered_features(
        extract_features(windows), bundle["feature_names"]
    )


def timeline_from_model(block, bundle, model_cfg):
    probabilities = predict_probabilities(
        bundle, _features_for_bundle(block["windows"], bundle)
    )
    cfg = model_cfg["timeline"]
    return build_timeline(
        probabilities, block["t_start"], block["t_end"], bundle["classes"],
        smoothing_windows=int(cfg["smoothing_windows"]),
        confidence_threshold=float(model_cfg["recognition"]["confidence_threshold"]),
        max_gap_seconds=float(cfg["max_gap_seconds"]),
        minimum_interval_seconds=float(cfg["minimum_interval_seconds"]),
        available_modalities=bundle.get("available_modalities"),
    )


def _score_answer(question_type, expected, predicted):
    if question_type in {"duration", "onset"}:
        try:
            return abs(float(expected.answer.split()[0]) - float(predicted.answer.split()[0])) <= 2.56
        except (ValueError, IndexError):
            return False
    return normalize_answer(expected.answer) == normalize_answer(predicted.answer)


def qa_queries(true_timeline):
    activities = sorted({item.activity for item in true_timeline})
    primary = activities[0]
    second = activities[1] if len(activities) > 1 else primary
    left, right = primary.replace("_", " "), second.replace("_", " ")
    templates = {
        "identification": [
            "What was the main activity?",
            "Which activity occupied the most time?",
            "What activity was the person doing for longest?",
        ],
        "verification": [
            f"Did the person perform {left}?",
            f"Was the person {left}?",
            f"Has the person done {left}?",
        ],
        "duration": [
            f"How long did {left} last?",
            f"What was the duration of {left}?",
            f"What was the total time spent {left}?",
        ],
        "count": [
            f"How many times did {left} occur?",
            f"How often did the person do {left}?",
            f"Count the {left} intervals.",
        ],
        "onset": [
            f"When did {left} first begin?",
            f"At what time did {left} start?",
            f"What was the onset of {left}?",
        ],
        "comparison": [
            f"Was {left} longer than {right}?",
            f"Did they spend more time {left} than {right}?",
            f"Compare the duration of {left} and {right}.",
        ],
        "grounding": [
            f"Did the person perform {left}?",
            f"Was the person {left}?",
            f"Has the person done {left}?",
        ],
        "open_world": [
            "Was there any strenuous activity?",
            "Did the person do anything vigorous?",
            "Was intense activity present?",
        ],
    }
    return [
        (question_type, question)
        for question_type, questions in templates.items()
        for question in questions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recognition-results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    processed = Path(args.processed)
    split_payload = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    splits = split_payload["splits"]
    bundle = load_bundle(Path(args.model))
    dataset_kind = str(bundle["dataset_kind"])
    classes = list(bundle["classes"])
    model_cfg = load_config("model")

    cap = int(load_config("labels")["imbalance"]["max_windows_per_class_per_user"])
    seed = int(model_cfg["recognition"]["random_seed"])
    test_blocks = list(
        load_users(
            processed, splits["recognition_test"], cap_per_class=cap, seed=seed + 2
        )
    )
    test_y = np.concatenate([block["y"] for block in test_blocks])
    test_windows = np.concatenate([block["windows"] for block in test_blocks])
    test_pred = np.argmax(
        predict_probabilities(bundle, _features_for_bundle(test_windows, bundle)), axis=1
    )
    matrix = confusion_matrix(test_y, test_pred, labels=np.arange(len(classes)))

    qa_scores = {name: [] for name in (
        "identification", "verification", "duration", "count", "onset", "comparison", "grounding", "open_world"
    )}
    ious = []
    for block in load_users(processed, splits["qa"]):
        true_timeline = timeline_from_labels(block, classes)
        predicted_timeline = timeline_from_model(block, bundle, model_cfg)
        for question_type, question in qa_queries(true_timeline):
            expected = answer_question(question, true_timeline)
            predicted = answer_question(question, predicted_timeline)
            correct = _score_answer(question_type, expected, predicted)
            if question_type == "grounding":
                correct = bool(
                    correct and expected.timestamps and predicted.timestamps
                    and interval_iou(expected.timestamps[0], predicted.timestamps[0]) >= 0.5
                    and expected.modality == predicted.modality
                    and set(expected.channels) == set(predicted.channels)
                )
            qa_scores[question_type].append(float(correct))
        for truth in true_timeline:
            candidates = [item for item in predicted_timeline if item.activity == truth.activity]
            ious.append(max((interval_iou((truth.start, truth.end), (item.start, item.end)) for item in candidates), default=0.0))

    recognition_payload = json.loads(Path(args.recognition_results).read_text(encoding="utf-8"))
    overhead = []
    for name, values in recognition_payload["candidate_validation"].items():
        overhead.append({
            "model": name,
            "latency_ms": values["median_latency_ms_per_window"],
            "accuracy": values["accuracy"],
            "serialized_bytes": values["serialized_bytes"],
        })

    rates = [0, 5, 10, 20, 30]
    robustness = []
    for rate in rates:
        scores = []
        for repeat in range(3):
            corrupted = test_windows.copy()
            rng = np.random.default_rng(seed + repeat)
            dropped = rng.random(corrupted.shape[:2]) < rate / 100.0
            replacement = np.median(corrupted, axis=1, keepdims=True)
            corrupted[dropped] = np.broadcast_to(replacement, corrupted.shape)[dropped]
            predicted = np.argmax(
                predict_probabilities(bundle, _features_for_bundle(corrupted, bundle)),
                axis=1,
            )
            scores.append(float(np.mean(predicted == test_y)))
        robustness.append(float(np.mean(scores)))

    thresholds = [round(value, 1) for value in np.arange(0.1, 1.0, 0.1)]
    payload = {
        "dataset_kind": dataset_kind,
        "scope": str(bundle.get("feature_source", "unknown")),
        "available_modalities": list(
            bundle.get("available_modalities", ["accelerometer", "gyroscope"])
        ),
        "accuracy_by_question_type": {key: float(np.mean(values)) for key, values in qa_scores.items()},
        "question_count_by_type": {key: len(values) for key, values in qa_scores.items()},
        "confusion_matrix": {"classes": classes, "matrix": matrix.tolist()},
        "strictness": {"threshold": thresholds, "accuracy": [float(np.mean(np.asarray(ious) >= value)) for value in thresholds]},
        "overhead": overhead,
        "robustness": {"dropped_percent": rates, "accuracy": robustness},
        "provenance": {
            "model_config_sha256": bundle["config_sha256"],
            "split_manifest_sha256": split_payload["sha256"],
            "qa_users": list(splits["qa"]),
            "robustness_repeats": 3,
        },
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    if dataset_kind != "real":
        print("*** SYNTHETIC EVALUATION - NOT A REAL-DATA RESULT ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

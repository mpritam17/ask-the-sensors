#!/usr/bin/env python3
"""Evaluate recognition, deterministic QA, grounding, strictness, and robustness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

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


def load_qa_recordings(processed: Path, users, *, recordings_per_user=5, examples_per_recording=12):
    """Yield short, transition-containing recordings from held-out QA users.

    ExtraSensory users span days of sparse 20-second sessions. Treating an
    entire user as one input recording would make duration/onset errors depend
    on multi-day wall-clock gaps. We instead form deterministic snippets around
    label transitions and rebase each snippet to its own second zero.
    """
    for uuid in users:
        with np.load(processed / f"{uuid}.npz", allow_pickle=True) as data:
            valid = np.asarray(data["valid"], bool)
            timestamps = np.asarray(data["example_ts"], int)[valid]
            labels = np.asarray(data["y"], int)[valid]
            windows = np.asarray(data["windows"])[valid]
            starts = np.asarray(data["t_start"], float)[valid]
            ends = np.asarray(data["t_end"], float)[valid]
            modalities = (
                [str(item) for item in data["modalities"]]
                if "modalities" in data.files
                else ["accelerometer", "gyroscope"]
            )
        unique_timestamps, first_indices = np.unique(timestamps, return_index=True)
        if len(unique_timestamps) == 0:
            continue
        session_labels = labels[first_indices]
        transitions = (np.flatnonzero(np.diff(session_labels) != 0) + 1).tolist()
        if len(transitions) > recordings_per_user:
            transition_positions = np.linspace(
                0, len(transitions) - 1, num=recordings_per_user, dtype=int
            )
            transitions = [transitions[index] for index in transition_positions]
        evenly_spaced = np.linspace(
            0, max(len(unique_timestamps) - examples_per_recording, 0),
            num=recordings_per_user,
            dtype=int,
        ).tolist()
        half = examples_per_recording // 2
        candidate_starts = [
            min(max(index - half, 0), max(len(unique_timestamps) - examples_per_recording, 0))
            for index in transitions
        ] + evenly_spaced
        selected_starts = []
        for candidate in candidate_starts:
            if candidate in selected_starts:
                continue
            selected_starts.append(candidate)
            if len(selected_starts) == recordings_per_user:
                break
        for recording_index, first in enumerate(selected_starts):
            selected_timestamps = unique_timestamps[first:first + examples_per_recording]
            selected = np.flatnonzero(np.isin(timestamps, selected_timestamps))
            if len(selected) == 0:
                continue
            origin = float(starts[selected[0]])
            yield {
                "uuid": uuid,
                "recording_index": recording_index,
                "windows": windows[selected],
                "y": labels[selected],
                "t_start": starts[selected] - origin,
                "t_end": ends[selected] - origin,
                "available_modalities": modalities,
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


def _evidence_iou(expected, predicted) -> float:
    """Match every expected evidence interval to its best predicted overlap."""
    if not expected.timestamps:
        return 1.0 if not predicted.timestamps else 0.0
    if not predicted.timestamps:
        return 0.0
    return float(np.mean([
        max(interval_iou(truth, candidate) for candidate in predicted.timestamps)
        for truth in expected.timestamps
    ]))


def _score_grounded(expected, predicted, answer_correct: bool) -> bool:
    if not answer_correct:
        return False
    if not expected.timestamps:
        return not predicted.timestamps
    if not predicted.timestamps:
        return False
    if expected.modality != predicted.modality:
        return False
    if set(expected.channels) != set(predicted.channels):
        return False
    expected_supported = all(
        max(interval_iou(truth, candidate) for candidate in predicted.timestamps) >= 0.5
        for truth in expected.timestamps
    )
    predicted_supported = all(
        max(interval_iou(candidate, truth) for truth in expected.timestamps) >= 0.5
        for candidate in predicted.timestamps
    )
    return expected_supported and predicted_supported


def qa_queries(true_timeline, all_classes=None):
    activities = sorted({item.activity for item in true_timeline})
    primary = activities[0]
    second = activities[1] if len(activities) > 1 else primary
    absent = next(
        (activity for activity in (all_classes or []) if activity not in activities),
        primary,
    )
    left, right = primary.replace("_", " "), second.replace("_", " ")
    absent_text = absent.replace("_", " ")
    templates = {
        "identification": [
            "What was the main activity?",
            "Which activity occupied the most time?",
            "What activity was the person doing for longest?",
        ],
        "verification": [
            f"Did the person perform {left}?",
            f"Was the person {left}?",
            f"Did the person perform {absent_text}?",
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
    parser.add_argument("--efficiency", default=None)
    parser.add_argument("--slm-efficiency", default=None)
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
    grounded_scores = {name: [] for name in qa_scores}
    categorical_pairs = {
        name: {"expected": [], "predicted": []}
        for name in ("identification", "comparison", "open_world")
    }
    verification_pairs = []
    open_world_rubric_scores = []
    numeric_errors = {"duration": [], "onset": []}
    numeric_attempts = {"duration": 0, "onset": 0}
    evidence_ious = []
    qa_blocks = list(load_qa_recordings(processed, splits["qa"]))
    for block in qa_blocks:
        true_timeline = timeline_from_labels(block, classes)
        predicted_timeline = timeline_from_model(block, bundle, model_cfg)
        for question_type, question in qa_queries(true_timeline, classes):
            expected = answer_question(question, true_timeline)
            predicted = answer_question(question, predicted_timeline)
            if question_type in numeric_errors:
                numeric_attempts[question_type] += 1
                try:
                    numeric_errors[question_type].append(
                        abs(float(expected.answer.split()[0]) - float(predicted.answer.split()[0]))
                    )
                except (ValueError, IndexError):
                    pass
            correct = _score_answer(question_type, expected, predicted)
            qa_scores[question_type].append(float(correct))
            grounded_scores[question_type].append(
                float(_score_grounded(expected, predicted, bool(correct)))
            )
            evidence_ious.append(_evidence_iou(expected, predicted))
            if question_type in categorical_pairs:
                categorical_pairs[question_type]["expected"].append(
                    normalize_answer(expected.answer)
                )
                categorical_pairs[question_type]["predicted"].append(
                    normalize_answer(predicted.answer)
                )
            if question_type == "verification":
                verification_pairs.append((
                    normalize_answer(expected.answer) == "yes",
                    normalize_answer(predicted.answer) == "yes",
                ))
            if question_type == "open_world":
                explanation = predicted.explanation.lower()
                rubric_checks = [
                    bool(predicted.answer.strip() and predicted.explanation.strip()),
                    bool(correct),
                    _evidence_iou(expected, predicted) >= 0.5,
                    (
                        expected.modality == predicted.modality
                        and set(expected.channels) == set(predicted.channels)
                    ),
                    any(
                        token in explanation
                        for token in ("confidence", "acc_", "gyro_", "interval", "running")
                    ),
                ]
                open_world_rubric_scores.append(max(1, sum(rubric_checks)))

    recognition_payload = json.loads(Path(args.recognition_results).read_text(encoding="utf-8"))
    plain_by_type = {key: float(np.mean(values)) for key, values in qa_scores.items()}
    grounded_by_type = {
        key: float(np.mean(values)) for key, values in grounded_scores.items()
    }
    overall_accuracy = float(np.mean(list(plain_by_type.values())))
    overall_grounded_accuracy = float(np.mean(list(grounded_by_type.values())))

    overhead = []
    overhead_metric = "recognition_validation_accuracy"
    if args.efficiency:
        efficiency = json.loads(Path(args.efficiency).read_text(encoding="utf-8"))
        deterministic = efficiency["tasks_1_3_batch"]
        deterministic_latency = float(deterministic["median_latency_ms"]) / float(
            deterministic["questions_per_run"]
        )
        model_bytes = int(efficiency["artifact"]["model_disk_bytes"])
        overhead.append({
            "model": "deterministic",
            "latency_ms": deterministic_latency,
            "accuracy": overall_accuracy,
            "serialized_bytes": model_bytes,
        })
        if args.slm_efficiency:
            slm = json.loads(Path(args.slm_efficiency).read_text(encoding="utf-8"))
            task4_latency = float(slm["inference"]["median_latency_ms"])
            type_count = len(qa_scores)
            workload_latency = (
                deterministic_latency * (type_count - 1) + task4_latency
            ) / type_count
            overhead.append({
                "model": "Qwen guarded",
                "latency_ms": workload_latency,
                "accuracy": overall_accuracy,
                "serialized_bytes": model_bytes + int(slm.get("cache_bytes") or 0),
            })
        overhead_metric = "overall_qa_macro_accuracy"
    else:
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
    numeric_tolerances = [0.0, 2.56, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    verification_counts = {
        "tp": sum(expected and predicted for expected, predicted in verification_pairs),
        "fp": sum((not expected) and predicted for expected, predicted in verification_pairs),
        "tn": sum((not expected) and (not predicted) for expected, predicted in verification_pairs),
        "fn": sum(expected and (not predicted) for expected, predicted in verification_pairs),
    }
    tp, fp = verification_counts["tp"], verification_counts["fp"]
    tn, fn = verification_counts["tn"], verification_counts["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    payload = {
        "dataset_kind": dataset_kind,
        "scope": str(bundle.get("feature_source", "unknown")),
        "available_modalities": list(
            bundle.get("available_modalities", ["accelerometer", "gyroscope"])
        ),
        "accuracy_by_question_type": plain_by_type,
        "grounded_accuracy_by_question_type": grounded_by_type,
        "overall_accuracy_macro": overall_accuracy,
        "overall_grounded_accuracy_macro": overall_grounded_accuracy,
        "question_count_by_type": {key: len(values) for key, values in qa_scores.items()},
        "categorical_macro_f1": {
            key: float(f1_score(
                values["expected"], values["predicted"], average="macro", zero_division=0
            ))
            for key, values in categorical_pairs.items()
        },
        "binary_verification": {
            **verification_counts,
            "precision_positive": float(precision),
            "recall_positive": float(recall),
            "f1_positive": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "specificity": float(tn / max(tn + fp, 1)),
        },
        "open_world_rubric": {
            "scale": "1-5",
            "mean": float(np.mean(open_world_rubric_scores)),
            "n": len(open_world_rubric_scores),
            "grader": "deterministic fixed rubric (not human-rated)",
            "criteria": [
                "nonempty answer and explanation",
                "broad behavior category correct",
                "cited evidence IoU at least 0.5",
                "modality and channels match reference",
                "explanation cites an interval, confidence, or measured feature",
            ],
        },
        "numeric_mae_seconds_answered": {
            key: (float(np.mean(values)) if values else None)
            for key, values in numeric_errors.items()
        },
        "numeric_answer_coverage": {
            key: len(numeric_errors[key]) / max(numeric_attempts[key], 1)
            for key in numeric_errors
        },
        "confusion_matrix": {"classes": classes, "matrix": matrix.tolist()},
        "strictness": {
            "threshold": thresholds,
            "accuracy": [
                float(np.mean(np.asarray(evidence_ious) >= value)) for value in thresholds
            ],
            "numeric_tolerance_seconds": numeric_tolerances,
            "duration_accuracy": [
                float(np.sum(np.asarray(numeric_errors["duration"]) <= value) / max(numeric_attempts["duration"], 1))
                for value in numeric_tolerances
            ],
            "onset_accuracy": [
                float(np.sum(np.asarray(numeric_errors["onset"]) <= value) / max(numeric_attempts["onset"], 1))
                for value in numeric_tolerances
            ],
        },
        "overhead": overhead,
        "overhead_metric": overhead_metric,
        "robustness": {"dropped_percent": rates, "accuracy": robustness},
        "provenance": {
            "model_config_sha256": bundle["config_sha256"],
            "split_manifest_sha256": split_payload["sha256"],
            "qa_users": list(splits["qa"]),
            "qa_recordings": len(qa_blocks),
            "qa_recordings_per_user_target": 5,
            "qa_examples_per_recording_target": 12,
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

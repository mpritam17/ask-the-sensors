import json

import numpy as np

from ats.eval import (
    exact_match,
    generate_required_figures,
    grounding_correct,
    interval_iou,
    mean_absolute_error,
    temporal_precision_recall_f1,
    tolerance_accuracy,
)


def test_core_qa_metrics():
    assert exact_match(["Yes", " Walking "], ["yes", "walking"]) == 1.0
    assert mean_absolute_error([1, 3], [2, 5]) == 1.5
    assert tolerance_accuracy([10, 20], [11, 25], absolute=2, relative=0.1) == 0.5
    assert interval_iou((0, 10), (5, 15)) == 1 / 3
    temporal = temporal_precision_recall_f1([(0, 10)], [(1, 9)], iou_threshold=0.5)
    assert temporal == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert grounding_correct(
        answer_correct=True,
        expected_interval=(0, 10), predicted_interval=(1, 9),
        expected_modality="both", predicted_modality="both",
        expected_channels=["All"], predicted_channels=["All"],
    )


def test_five_figures_are_generated_and_nonempty(tmp_path):
    payload = {
        "dataset_kind": "synthetic",
        "accuracy_by_question_type": {"id": 0.8, "duration": 0.6},
        "confusion_matrix": {"classes": ["a", "b"], "matrix": [[4, 1], [2, 3]]},
        "strictness": {"threshold": [0.1, 0.5, 0.9], "accuracy": [0.9, 0.7, 0.3]},
        "overhead": [
            {"model": "logistic", "latency_ms": 1.0, "accuracy": 0.6},
            {"model": "compact RF", "latency_ms": 2.0, "accuracy": 0.75},
            {"model": "full RF", "latency_ms": 4.0, "accuracy": 0.8},
        ],
        "robustness": {"dropped_percent": [0, 10, 30], "accuracy": [0.8, 0.7, 0.5]},
    }
    paths = generate_required_figures(payload, tmp_path)
    assert len(paths) == 5
    assert all(path.stat().st_size > 1000 for path in paths)

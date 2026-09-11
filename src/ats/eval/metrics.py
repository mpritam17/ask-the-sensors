"""Metrics for categorical, numeric, temporal, and grounded QA."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def normalize_answer(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def exact_match(expected: Iterable[object], predicted: Iterable[object]) -> float:
    pairs = list(zip(expected, predicted))
    if not pairs:
        return 0.0
    return float(np.mean([normalize_answer(a) == normalize_answer(b) for a, b in pairs]))


def mean_absolute_error(expected: Sequence[float], predicted: Sequence[float]) -> float:
    if len(expected) == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(expected, float) - np.asarray(predicted, float))))


def tolerance_accuracy(
    expected: Sequence[float],
    predicted: Sequence[float],
    *,
    absolute: float = 2.0,
    relative: float = 0.10,
) -> float:
    if len(expected) == 0:
        return 0.0
    expected_arr, predicted_arr = np.asarray(expected, float), np.asarray(predicted, float)
    allowed = np.maximum(float(absolute), np.abs(expected_arr) * float(relative))
    return float(np.mean(np.abs(expected_arr - predicted_arr) <= allowed))


def interval_iou(expected: tuple[float, float], predicted: tuple[float, float]) -> float:
    left = max(float(expected[0]), float(predicted[0]))
    right = min(float(expected[1]), float(predicted[1]))
    intersection = max(0.0, right - left)
    union = max(float(expected[1]), float(predicted[1])) - min(float(expected[0]), float(predicted[0]))
    return intersection / union if union > 0 else float(expected == predicted)


def temporal_precision_recall_f1(
    expected: Sequence[tuple[float, float]],
    predicted: Sequence[tuple[float, float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    unmatched = set(range(len(expected)))
    true_positive = 0
    for candidate in predicted:
        scored = [(interval_iou(expected[index], candidate), index) for index in unmatched]
        if scored:
            score, index = max(scored)
            if score >= iou_threshold:
                unmatched.remove(index)
                true_positive += 1
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grounding_correct(
    *,
    answer_correct: bool,
    expected_interval: tuple[float, float],
    predicted_interval: tuple[float, float],
    expected_modality: str,
    predicted_modality: str,
    expected_channels: Iterable[str],
    predicted_channels: Iterable[str],
    iou_threshold: float = 0.5,
) -> bool:
    return bool(
        answer_correct
        and interval_iou(expected_interval, predicted_interval) >= iou_threshold
        and normalize_answer(expected_modality) == normalize_answer(predicted_modality)
        and set(expected_channels) == set(predicted_channels)
    )

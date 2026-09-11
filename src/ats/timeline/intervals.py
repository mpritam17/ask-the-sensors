"""Convert window probabilities into gap-aware, evidence-carrying intervals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ActivityInterval:
    activity: str
    start: float
    end: float
    confidence: float
    modality: str
    channels: List[str]
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end - self.start))

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["duration"] = self.duration
        return payload


def evidence_source(activity: str) -> Tuple[str, List[str]]:
    if activity in {"lying_down", "sitting", "standing_in_place"}:
        return "accelerometer", ["Acc X", "Acc Y", "Acc Z"]
    if activity == "standing_and_moving":
        return "both", ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    return "both", ["All"]


def smooth_probabilities(probabilities: np.ndarray, width: int = 3) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (windows, classes)")
    if width <= 1 or len(probabilities) == 0:
        return probabilities.copy()
    if width % 2 == 0:
        raise ValueError("smoothing width must be odd")
    radius = width // 2
    padded = np.pad(probabilities, ((radius, radius), (0, 0)), mode="edge")
    return np.vstack([
        padded[i:i + width].mean(axis=0) for i in range(len(probabilities))
    ])


def build_timeline(
    probabilities: np.ndarray,
    t_start: Sequence[float],
    t_end: Sequence[float],
    classes: Sequence[str],
    *,
    valid: Sequence[bool] | None = None,
    feature_summaries: Sequence[Dict[str, float]] | None = None,
    confidence_threshold: float = 0.35,
    smoothing_windows: int = 3,
    max_gap_seconds: float = 1.5,
    minimum_interval_seconds: float = 1.0,
) -> List[ActivityInterval]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    starts = np.asarray(t_start, dtype=np.float64)
    ends = np.asarray(t_end, dtype=np.float64)
    if probabilities.shape[0] != len(starts) or len(starts) != len(ends):
        raise ValueError("probabilities and timestamps must have equal lengths")
    if probabilities.shape[1] != len(classes):
        raise ValueError("probability columns must match classes")
    valid_mask = np.ones(len(starts), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if len(valid_mask) != len(starts):
        raise ValueError("valid mask must match timestamps")
    summaries = feature_summaries or [{} for _ in starts]
    if len(summaries) != len(starts):
        raise ValueError("feature summaries must match timestamps")

    smoothed = smooth_probabilities(probabilities, smoothing_windows)
    labels = np.argmax(smoothed, axis=1)
    confidence = np.max(smoothed, axis=1)
    supported = valid_mask & (confidence >= confidence_threshold)

    intervals: List[ActivityInterval] = []
    contributors: List[List[Dict[str, float]]] = []
    for index in np.flatnonzero(supported):
        activity = str(classes[int(labels[index])])
        start, end, conf = float(starts[index]), float(ends[index]), float(confidence[index])
        modality, channels = evidence_source(activity)
        if intervals and intervals[-1].activity == activity and start <= intervals[-1].end + max_gap_seconds:
            old_duration = intervals[-1].duration
            new_duration = max(end, intervals[-1].end) - intervals[-1].start
            intervals[-1].end = max(intervals[-1].end, end)
            intervals[-1].confidence = (
                intervals[-1].confidence * old_duration + conf * max(end - start, 1e-9)
            ) / max(new_duration, 1e-9)
            contributors[-1].append(dict(summaries[index]))
        else:
            intervals.append(ActivityInterval(activity, start, end, conf, modality, channels))
            contributors.append([dict(summaries[index])])

    kept: List[ActivityInterval] = []
    for interval, blocks in zip(intervals, contributors):
        if interval.duration < minimum_interval_seconds:
            continue
        keys = sorted(set().union(*(block.keys() for block in blocks))) if blocks else []
        interval.features = {
            key: float(np.mean([b[key] for b in blocks if key in b])) for key in keys
        }
        kept.append(interval)
    return kept


def intervals_for(intervals: Iterable[ActivityInterval], activity: str) -> List[ActivityInterval]:
    return [interval for interval in intervals if interval.activity == activity]

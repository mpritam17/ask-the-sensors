import numpy as np

from ats.timeline import build_timeline


def test_timeline_smooths_merges_and_respects_gaps():
    classes = ["sitting", "walking"]
    probabilities = np.array([
        [0.9, 0.1], [0.85, 0.15], [0.1, 0.9], [0.8, 0.2],
        [0.1, 0.9], [0.1, 0.9],
    ])
    starts = np.array([0.0, 1.28, 2.56, 3.84, 20.0, 21.28])
    ends = starts + 2.56
    timeline = build_timeline(
        probabilities, starts, ends, classes,
        smoothing_windows=3, max_gap_seconds=1.5
    )
    assert [item.activity for item in timeline] == ["sitting", "walking", "walking"]
    assert timeline[0].start == 0.0
    assert timeline[-1].start == 20.0
    assert timeline[1].end < timeline[-1].start


def test_low_confidence_and_invalid_windows_are_rejected():
    timeline = build_timeline(
        np.array([[0.34, 0.33, 0.33], [0.9, 0.05, 0.05]]),
        [0.0, 1.0], [2.0, 3.0], ["a", "b", "c"], valid=[True, False],
        smoothing_windows=1, confidence_threshold=0.35,
    )
    assert timeline == []


def test_accelerometer_only_timeline_never_claims_gyroscope_evidence():
    timeline = build_timeline(
        np.array([[0.05, 0.95]]),
        [0.0],
        [2.56],
        ["sitting", "walking"],
        smoothing_windows=1,
        available_modalities=["accelerometer"],
    )
    assert len(timeline) == 1
    assert timeline[0].modality == "accelerometer"
    assert timeline[0].channels == ["Acc X", "Acc Y", "Acc Z"]

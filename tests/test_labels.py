"""Unit tests for the 7-class label mapping and its filters."""
import numpy as np
import pandas as pd
import pytest

from ats.config import load_config
from ats.data.labels import build_label_table, label_summary, normalise_column


@pytest.fixture
def cfg():
    return load_config("labels")


def _write(tmp_path, name, frame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_normalise_column_strips_prefix_and_punctuation():
    assert normalise_column("label:FIX_walking") == "FIX_WALKING"
    assert normalise_column("label:DRIVE_-_I_M_THE_DRIVER") == "DRIVE_I_M_THE_DRIVER"
    assert normalise_column(" LYING_DOWN ") == "LYING_DOWN"


def test_all_seven_classes_are_recoverable_from_original_labels(tmp_path, cfg):
    """The point of using the original labels: both standing classes survive."""
    ts = np.arange(1_444_000_000, 1_444_000_000 + 7 * 60, 60)
    frame = pd.DataFrame({"timestamp": ts})
    cols = ["label:LYING_DOWN", "label:SITTING", "label:STANDING_IN_PLACE",
            "label:STANDING_AND_MOVING", "label:WALKING", "label:RUNNING",
            "label:BICYCLING"]
    for i, col in enumerate(cols):
        frame[col] = [1 if j == i else 0 for j in range(7)]
    table = build_label_table(_write(tmp_path, "orig.csv", frame), None, cfg)
    assert table["kept"].all()
    assert set(table["label"]) == set(cfg["classes"])


def test_examples_without_a_main_activity_are_dropped(tmp_path, cfg):
    frame = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "label:SITTING": [1, 0, 1],
        "label:WALKING": [0, 0, 0],
    })
    table = build_label_table(_write(tmp_path, "orig.csv", frame), None, cfg)
    assert table["kept"].tolist() == [True, False, True]
    assert table.loc[1, "drop_reason"] == "no_main_activity"


def test_ambiguous_multi_activity_examples_are_dropped_not_arbitrated(tmp_path, cfg):
    frame = pd.DataFrame({
        "timestamp": [1, 2],
        "label:SITTING": [1, 1],
        "label:WALKING": [0, 1],       # contradictory self-report
    })
    table = build_label_table(_write(tmp_path, "orig.csv", frame), None, cfg)
    assert table.loc[0, "kept"]
    assert not table.loc[1, "kept"]
    assert table.loc[1, "drop_reason"] == "multiple_main_activities"


def test_cleaned_labels_veto_contradictions_but_tolerate_nan(tmp_path, cfg):
    orig = pd.DataFrame({
        "timestamp": [10, 20, 30],
        "label:WALKING": [1, 1, 1],
    })
    cleaned = pd.DataFrame({
        "timestamp": [10, 20, 30],
        "label:FIX_walking": [1.0, 0.0, np.nan],   # agree / contradict / unknown
    })
    table = build_label_table(_write(tmp_path, "orig.csv", orig),
                              _write(tmp_path, "clean.csv", cleaned), cfg)
    reasons = dict(zip(table["timestamp"], table["drop_reason"]))
    assert table.set_index("timestamp").loc[10, "kept"]
    assert not table.set_index("timestamp").loc[20, "kept"]
    assert reasons[20] == "cleaned_label_contradiction"
    assert table.set_index("timestamp").loc[30, "kept"]   # NaN is tolerated


def test_missing_class_columns_do_not_crash(tmp_path, cfg):
    """Rare classes are genuinely absent for many users."""
    frame = pd.DataFrame({"timestamp": [1, 2], "label:SITTING": [1, 1]})
    table = build_label_table(_write(tmp_path, "orig.csv", frame), None, cfg)
    assert table["kept"].all()
    assert set(table["label"]) == {"sitting"}


def test_label_summary_reports_counts_and_drops(tmp_path, cfg):
    frame = pd.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "label:SITTING": [1, 1, 0, 1],
        "label:WALKING": [0, 0, 0, 1],
    })
    table = build_label_table(_write(tmp_path, "orig.csv", frame), None, cfg)
    summary = label_summary(table)
    assert summary.loc["sitting", "kept_examples"] == 2
    assert summary.loc["no_main_activity", "dropped_examples"] == 1

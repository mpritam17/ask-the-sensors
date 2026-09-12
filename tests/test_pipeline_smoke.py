"""End-to-end smoke test on synthetic data: raw files -> windowed arrays.

Guards the contract every later phase depends on: shapes, channel order,
the seconds-from-start time base, and that no NaN reaches the model.
"""
import numpy as np

from ats.config import load_config
from ats.data.build_dataset import build_user
from ats.data.synthetic import write_synthetic_user


def test_build_user_end_to_end(tmp_path):
    uuid = "00000000-0000-4000-8000-000000000000"
    schedule = [("sitting", 4), ("walking", 3), ("running", 2), ("lying_down", 3)]
    truth = write_synthetic_user(tmp_path, uuid, schedule, seed=1,
                                 missing_example_prob=0.0)

    arrays, report = build_user(
        tmp_path, uuid,
        tmp_path / "original_labels" / f"{uuid}.original_labels.csv",
        tmp_path / "features_labels" / f"{uuid}.features_labels.csv",
    )
    assert arrays is not None, report.to_dict()

    windows = arrays["windows"]
    assert windows.ndim == 3 and windows.shape[1:] == (64, 6)
    assert np.isfinite(windows).all()
    assert len(windows) == len(arrays["y"]) == len(arrays["t_start"])

    # Time base: seconds from the first retained example, strictly increasing.
    assert arrays["t_start"][0] == 0.0
    assert np.all(np.diff(arrays["t_start"]) > 0)
    assert int(arrays["t0_unix"]) == int(truth["timestamp"].min())

    # Channel order: accelerometer carries ~1 g, gyroscope is centred on zero.
    acc_magnitude = np.linalg.norm(windows[:, :, 0:3], axis=2).mean()
    gyro_magnitude = np.linalg.norm(windows[:, :, 3:6], axis=2).mean()
    assert 0.8 < acc_magnitude < 1.4
    assert gyro_magnitude < 0.8

    # Every scheduled class is present and separable in gross energy.
    classes = list(arrays["classes"])
    labels = arrays["y"][arrays["valid"]]
    assert set(np.unique(labels)) == {classes.index(c) for c, _ in schedule}

    def energy(name):
        idx = arrays["y"] == classes.index(name)
        return float(np.var(windows[idx][:, :, 0:3]))

    assert energy("running") > energy("walking") > energy("sitting")
    assert report.n_examples_used == len(truth)


def test_time_origin_is_first_usable_example(tmp_path):
    uuid = "00000000-0000-4000-8000-000000000001"
    truth = write_synthetic_user(
        tmp_path, uuid, [("sitting", 3)], seed=9, missing_example_prob=0.0
    )
    first_ts = int(truth["timestamp"].iloc[0])
    for sensor_dir in ("raw_acc", "proc_gyro"):
        first_file = next((tmp_path / sensor_dir / uuid).glob(f"{first_ts}.*"))
        first_file.unlink()

    arrays, report = build_user(
        tmp_path,
        uuid,
        tmp_path / "original_labels" / f"{uuid}.original_labels.csv",
        tmp_path / "features_labels" / f"{uuid}.features_labels.csv",
    )
    assert arrays is not None, report.to_dict()
    assert arrays["t_start"][0] == 0.0
    assert int(arrays["t0_unix"]) == int(truth["timestamp"].iloc[1])
    assert report.dropped_no_sensor == 1


def test_accelerometer_only_fallback_records_and_enforces_provenance(tmp_path):
    uuid = "00000000-0000-4000-8000-000000000002"
    write_synthetic_user(
        tmp_path, uuid, [("walking", 2)], seed=10, missing_example_prob=0.0
    )
    for path in (tmp_path / "proc_gyro" / uuid).iterdir():
        path.unlink()

    arrays, report = build_user(
        tmp_path,
        uuid,
        tmp_path / "original_labels" / f"{uuid}.original_labels.csv",
        tmp_path / "features_labels" / f"{uuid}.features_labels.csv",
        modalities="acc",
    )
    assert arrays is not None, report.to_dict()
    assert list(arrays["modalities"]) == ["accelerometer"]
    assert np.all(arrays["windows"][:, :, 3:] == 0)
    assert report.modalities == "acc"

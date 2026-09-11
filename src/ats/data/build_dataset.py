"""Turn raw ExtraSensory sessions into windowed, labelled arrays.

Output per user: data/processed/<uuid>.npz containing

    windows    (N, 64, 6) float32   Acc X/Y/Z, Gyro X/Y/Z at 25 Hz
    y          (N,)       int8      class index, -1 for unlabelled
    t_start    (N,)       float64   seconds from the start of the recording
    t_end      (N,)       float64
    example_ts (N,)       int64     the parent example's Unix timestamp
    valid      (N,)       bool
    classes    (7,)       str
    t0_unix    scalar     int64     Unix time of second 0 of this recording

TIME BASE (stated once, used everywhere, per the challenge brief):
    all reported times are SECONDS FROM THE START OF THE RECORDING, where
    second 0 is the timestamp of the first retained example. t0_unix is kept
    only so an answer can be translated back to clock time on request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ats.config import Config, load_config, resolve
from ats.data import raw_io
from ats.data.labels import build_label_table
from ats.data.resample import ResampledStream, align_streams, resample_to_grid
from ats.data.windowing import WindowSet, concat_windowsets, make_windows

# ExtraSensory phone accelerometer is reported in g. Some Android devices
# report m/s^2. Detect and rescale rather than silently training on a mixture.
GRAVITY_MS2 = 9.80665
_MS2_DETECTION_THRESHOLD = 5.0


@dataclass
class BuildReport:
    uuid: str
    n_examples_seen: int = 0
    n_examples_labelled: int = 0
    n_examples_used: int = 0
    n_windows: int = 0
    n_windows_valid: int = 0
    dropped_no_sensor: int = 0
    dropped_low_coverage: int = 0
    unit_rescaled: bool = False
    class_counts: Dict[str, int] = None

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


def _rescale_units(acc: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Convert accelerometer values to g if they look like m/s^2."""
    magnitude = np.linalg.norm(acc, axis=1)
    median = float(np.median(magnitude)) if magnitude.size else 0.0
    if median > _MS2_DETECTION_THRESHOLD:
        return acc / GRAVITY_MS2, True
    return acc, False


def build_user(
    raw_root: Path,
    uuid: str,
    original_labels_path: Path,
    cleaned_labels_path: Optional[Path],
    data_cfg: Optional[Config] = None,
    label_cfg: Optional[Config] = None,
    acc_index: Optional[Dict[int, Path]] = None,
    gyro_index: Optional[Dict[int, Path]] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], BuildReport]:
    """Build the windowed array set for one user."""
    data_cfg = data_cfg or load_config("data")
    label_cfg = label_cfg or load_config("labels")
    classes: List[str] = list(label_cfg["classes"])
    class_index = {c: i for i, c in enumerate(classes)}

    fs = float(data_cfg["sampling"]["target_hz"])
    example_seconds = float(data_cfg["sampling"]["example_seconds"])
    max_gap = float(data_cfg["sampling"]["max_interp_gap_seconds"])
    min_cov = float(data_cfg["sampling"]["min_example_coverage"])
    win = data_cfg["windowing"]

    report = BuildReport(uuid=uuid, class_counts={c: 0 for c in classes})

    labels = build_label_table(original_labels_path, cleaned_labels_path, label_cfg)
    kept = labels[labels["kept"]]
    label_by_ts: Dict[int, str] = dict(zip(kept["timestamp"], kept["label"]))
    report.n_examples_labelled = len(label_by_ts)
    if not label_by_ts:
        return None, report

    window_sets: List[WindowSet] = []
    y_blocks: List[np.ndarray] = []
    ts_blocks: List[np.ndarray] = []
    t0_unix: Optional[int] = None

    for ts, acc_ex, gyro_ex in raw_io.iter_examples(raw_root, uuid,
                                                    timestamps=list(label_by_ts),
                                                    acc_index=acc_index,
                                                    gyro_index=gyro_index):
        report.n_examples_seen += 1
        if acc_ex is None or gyro_ex is None:
            report.dropped_no_sensor += 1
            continue

        acc_values, rescaled = _rescale_units(acc_ex.x)
        report.unit_rescaled = report.unit_rescaled or rescaled

        acc_stream: ResampledStream = resample_to_grid(
            acc_ex.t, acc_values, fs=fs, duration=example_seconds, t0=0.0,
            max_gap=max_gap)
        gyro_stream: ResampledStream = resample_to_grid(
            gyro_ex.t, gyro_ex.x, fs=fs, duration=example_seconds, t0=0.0,
            max_gap=max_gap)

        if min(acc_stream.coverage, gyro_stream.coverage) < min_cov:
            report.dropped_low_coverage += 1
            continue

        values, valid, _ = align_streams(acc_stream, gyro_stream)
        ws = make_windows(
            values, valid, t0=0.0, fs=fs,
            window_samples=int(win["window_samples"]),
            hop_samples=int(win["hop_samples"]),
            min_validity=float(win["min_window_validity"]),
            meta={"example_ts": ts},
        )
        if len(ws) == 0:
            continue

        # The origin is the first example that actually survives sensor,
        # coverage, and window checks - never merely the earliest label row.
        if t0_unix is None:
            t0_unix = int(ts)
        session_t0 = float(ts - t0_unix)
        ws.t_start += session_t0
        ws.t_end += session_t0

        label = label_by_ts[ts]
        window_sets.append(ws)
        y_blocks.append(np.full(len(ws), class_index[label], dtype=np.int8))
        ts_blocks.append(np.full(len(ws), ts, dtype=np.int64))
        report.n_examples_used += 1
        report.class_counts[label] += int(ws.valid.sum())

    if not window_sets or t0_unix is None:
        return None, report

    merged = concat_windowsets(window_sets)
    arrays = {
        "windows": merged.values.astype(np.float32),
        "y": np.concatenate(y_blocks),
        "t_start": merged.t_start,
        "t_end": merged.t_end,
        "example_ts": np.concatenate(ts_blocks),
        "valid": merged.valid,
        "validity_fraction": merged.validity_fraction,
        "classes": np.array(classes, dtype=object),
        "t0_unix": np.int64(t0_unix),
        "uuid": np.array(uuid, dtype=object),
    }
    report.n_windows = int(len(merged))
    report.n_windows_valid = int(merged.valid.sum())
    return arrays, report


def save_user(arrays: Dict[str, np.ndarray], out_dir: Path, uuid: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{uuid}.npz"
    np.savez_compressed(path, **arrays)
    return path


def write_manifest(reports: List[BuildReport], out_dir: Path) -> Path:
    """Per-user build statistics — the report quotes these, not estimates."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "build_manifest.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([r.to_dict() for r in reports], handle, indent=2)
    return path


def manifest_summary(reports: List[BuildReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        row = {"uuid": r.uuid, "examples_used": r.n_examples_used,
               "windows_valid": r.n_windows_valid}
        row.update(r.class_counts or {})
        rows.append(row)
    return pd.DataFrame(rows)

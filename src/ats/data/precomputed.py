"""Join official ExtraSensory sensor features to original seven-class labels."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from ats.config import Config, load_config
from ats.data.labels import build_label_table


def sensor_feature_columns(columns: Sequence[str], modalities: str = "both") -> List[str]:
    """Select only phone accelerometer/gyroscope features, never label columns."""
    selected = []
    for column in columns:
        lowered = str(column).lower()
        if lowered == "timestamp" or "label:" in lowered:
            continue
        is_acc = "raw_acc" in lowered
        is_gyro = "proc_gyro" in lowered or "raw_gyro" in lowered
        if modalities == "acc" and is_acc:
            selected.append(str(column))
        elif modalities == "gyro" and is_gyro:
            selected.append(str(column))
        elif modalities == "both" and (is_acc or is_gyro):
            selected.append(str(column))
    return selected


def feature_provenance_from_columns(columns: Sequence[str]):
    output = []
    for column in columns:
        lowered = column.lower()
        modality = "accelerometer" if "raw_acc" in lowered else "gyroscope"
        output.append({"feature": column, "modality": modality, "channels": ["All"]})
    return output


def load_precomputed_user(
    features_path: Path,
    original_labels_path: Path,
    *,
    feature_columns: Sequence[str] | None = None,
    modalities: str = "both",
    label_cfg: Config | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return official feature rows aligned to unambiguous original labels."""
    label_cfg = label_cfg or load_config("labels")
    labels = build_label_table(original_labels_path, features_path, label_cfg)
    labels = labels.loc[labels["kept"], ["timestamp", "label"]]
    frame = pd.read_csv(features_path)
    if "timestamp" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    columns = list(feature_columns or sensor_feature_columns(frame.columns, modalities))
    if not columns:
        raise ValueError(f"no {modalities} sensor feature columns found in {features_path}")
    merged = labels.merge(frame[["timestamp", *columns]], on="timestamp", how="inner")
    class_index = {name: index for index, name in enumerate(label_cfg["classes"])}
    x = merged[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
    x[~np.isfinite(x)] = np.nan
    y = merged["label"].map(class_index).to_numpy(dtype=np.int64)
    return x, y, columns

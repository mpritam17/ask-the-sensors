"""Mapping ExtraSensory self-reported labels onto the 7 challenge classes.

See configs/labels.yaml for the full rationale. In short: the challenge's seven
classes are exactly ExtraSensory's mutually-exclusive "main activity" category,
which survives intact only in the *original* (uncleaned) label release — the
cleaned release merges "standing in place" and "standing and moving" into a
single OR_standing label. We therefore read the original labels and use the
cleaned labels as a contradiction filter.

Source: ExtraSensory dataset, http://extrasensory.ucsd.edu/ (Vaizman et al.,
2017). Cited at point of use.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ats.config import Config, load_config


def normalise_column(name: str) -> str:
    """Strip cleaned/original prefixes and normalize punctuation/case."""
    name = re.sub(
        r"^(?:original_)?label:", "", str(name).strip(), flags=re.IGNORECASE
    )
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").upper()


def read_user_csv(path: Path) -> pd.DataFrame:
    """Read a per-user csv or csv.gz with normalised column names."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame.columns = [c if c == "timestamp" else normalise_column(c)
                     for c in (str(x).strip() for x in frame.columns)]
    if "timestamp" not in frame.columns:
        # First column is the timestamp in every ExtraSensory per-user file.
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    return frame


def build_label_table(
    original_labels_path: Path,
    cleaned_labels_path: Optional[Path] = None,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """Produce a per-example label table for one user.

    Returns a DataFrame with columns:
        timestamp : int64 Unix seconds (the example identifier)
        label     : one of the 7 class slugs
        n_main    : how many main-activity labels the user reported
        kept      : bool — survived the filters
        drop_reason : '' when kept, otherwise why it was dropped
    """
    cfg = cfg or load_config("labels")
    classes: List[str] = list(cfg["classes"])
    original_cols: Dict[str, str] = {
        k: normalise_column(v) for k, v in cfg["original_label_columns"].items()
    }
    filters = cfg.get("filters", {}) or {}

    original = read_user_csv(original_labels_path)

    present = {c: original_cols[c] for c in classes if original_cols[c] in original.columns}
    missing = [c for c in classes if c not in present]
    if missing:
        # Not fatal: rare classes are genuinely absent for many users.
        for cls in missing:
            original[original_cols[cls]] = 0.0
            present[cls] = original_cols[cls]

    indicator = original[[present[c] for c in classes]].to_numpy(dtype=np.float64)
    indicator = np.nan_to_num(indicator, nan=0.0)
    n_main = (indicator > 0.5).sum(axis=1)

    out = pd.DataFrame({
        "timestamp": original["timestamp"].to_numpy(np.int64),
        "n_main": n_main.astype(np.int16),
    })
    first_positive = np.argmax(indicator > 0.5, axis=1)
    out["label"] = [classes[i] if n > 0 else "" for i, n in zip(first_positive, n_main)]
    out["kept"] = True
    out["drop_reason"] = ""

    def _drop(mask: np.ndarray, reason: str) -> None:
        mask = mask & out["kept"].to_numpy()
        out.loc[mask, "kept"] = False
        out.loc[mask, "drop_reason"] = reason

    if filters.get("require_main_activity", True):
        _drop(n_main == 0, "no_main_activity")
    if filters.get("drop_multi_main_activity", True):
        _drop(n_main > 1, "multiple_main_activities")

    if filters.get("apply_cleaned_consistency_filter", True) and cleaned_labels_path:
        out = _apply_cleaned_filter(out, Path(cleaned_labels_path), cfg)

    return out


def _apply_cleaned_filter(
    table: pd.DataFrame, cleaned_path: Path, cfg: Config
) -> pd.DataFrame:
    """Drop examples whose cleaned label directly contradicts the original one.

    A cleaned value of 0 is an explicit "this was not happening". NaN means the
    cleaning procedure could not determine it and is tolerated — dropping those
    would discard most of the dataset.
    """
    cleaned_cols = {k: normalise_column(v)
                    for k, v in cfg["cleaned_consistency_columns"].items()}
    cleaned = read_user_csv(cleaned_path)
    keep_cols = ["timestamp"] + sorted({c for c in cleaned_cols.values()
                                        if c in cleaned.columns})
    cleaned = cleaned[keep_cols]

    merged = table.merge(cleaned, on="timestamp", how="left", suffixes=("", "_clean"))
    contradicts = np.zeros(len(merged), dtype=bool)
    for cls, col in cleaned_cols.items():
        if col not in merged.columns:
            continue
        is_class = (merged["label"] == cls).to_numpy()
        values = merged[col].to_numpy(dtype=np.float64)
        contradicts |= is_class & (values == 0.0)

    contradicts &= merged["kept"].to_numpy()
    merged.loc[contradicts, "kept"] = False
    merged.loc[contradicts, "drop_reason"] = "cleaned_label_contradiction"
    return merged[["timestamp", "n_main", "label", "kept", "drop_reason"]]


def label_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Counts per class plus drop reasons — printed by the build script so the
    report can quote real numbers instead of estimates."""
    kept = table[table["kept"]]
    counts = kept["label"].value_counts().rename("kept_examples")
    drops = (table.loc[~table["kept"], "drop_reason"]
             .value_counts().rename("dropped_examples"))
    return pd.concat([counts, drops], axis=1).fillna(0).astype(int)

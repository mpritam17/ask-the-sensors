"""Segmenting a resampled recording into overlapping analysis windows.

Default: 64 samples at 25 Hz (2.56 s) with a 32-sample hop (50% overlap).

Why 2.56 s:
  * a power of two, so the FFT in the feature extractor needs no zero-padding
    and the frequency bins land on exact multiples of 25/64 = 0.39 Hz;
  * long enough to contain 2-4 gait cycles at normal walking cadence
    (~1.6-2.2 Hz), which is what makes the dominant-frequency feature separable
    between walking and running;
  * short enough that a labelled 20 s session yields 14 windows and that the
    quantisation error on an interval boundary stays at ~1.3 s, comfortably
    inside the temporal tolerances used for scoring.

Why 50% overlap:
  * doubles the number of training windows from an imbalanced dataset;
  * halves the worst-case boundary error at activity transitions.
  Cost: adjacent windows are correlated, so the train/test split must be by
  user (configs/data.yaml: split.strategy = by_user), never by window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WindowSet:
    """Windows extracted from one continuous block of signal."""

    values: np.ndarray                  # (n_windows, window_samples, n_channels)
    valid: np.ndarray                   # (n_windows,) bool
    validity_fraction: np.ndarray       # (n_windows,) float in [0, 1]
    t_start: np.ndarray                 # (n_windows,) seconds, same base as input
    t_end: np.ndarray                   # (n_windows,) seconds
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def filter_valid(self) -> "WindowSet":
        """Keep only windows that met the validity threshold."""
        idx = np.flatnonzero(self.valid)
        return WindowSet(
            values=self.values[idx],
            valid=self.valid[idx],
            validity_fraction=self.validity_fraction[idx],
            t_start=self.t_start[idx],
            t_end=self.t_end[idx],
            meta=dict(self.meta),
        )


def make_windows(
    values: np.ndarray,
    valid: np.ndarray,
    t0: float,
    fs: float = 25.0,
    window_samples: int = 64,
    hop_samples: int = 32,
    min_validity: float = 0.90,
    meta: Optional[Dict[str, Any]] = None,
) -> WindowSet:
    """Slice a (n_samples, n_channels) block into overlapping windows.

    A window is marked valid when at least ``min_validity`` of its samples are
    backed by real measurements. Invalid windows are retained in the array (so
    indices stay aligned with timestamps) but flagged; callers either drop them
    or emit an explicit "no reliable data" answer for that span.
    """
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got {values.shape}")
    if values.shape[0] != valid.shape[0]:
        raise ValueError("values and valid must have the same length")
    if hop_samples <= 0 or window_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")

    n_samples = values.shape[0]
    if n_samples < window_samples:
        empty = np.zeros((0, window_samples, values.shape[1]), dtype=values.dtype)
        return WindowSet(empty, np.zeros(0, bool), np.zeros(0), np.zeros(0),
                         np.zeros(0), dict(meta or {}))

    starts = np.arange(0, n_samples - window_samples + 1, hop_samples, dtype=np.int64)
    # Strided view avoids copying the whole recording per window.
    idx = starts[:, None] + np.arange(window_samples)[None, :]
    windows = values[idx]                       # (n_windows, W, C)
    frac = valid[idx].mean(axis=1)              # (n_windows,)

    t_start = t0 + starts / fs
    t_end = t_start + window_samples / fs
    return WindowSet(
        values=np.ascontiguousarray(windows),
        valid=frac >= min_validity,
        validity_fraction=frac.astype(np.float32),
        t_start=t_start,
        t_end=t_end,
        meta=dict(meta or {}),
    )


def concat_windowsets(sets: List[WindowSet]) -> WindowSet:
    """Concatenate window sets (e.g. all 20 s sessions of one user)."""
    sets = [s for s in sets if len(s) > 0]
    if not sets:
        raise ValueError("nothing to concatenate")
    return WindowSet(
        values=np.concatenate([s.values for s in sets], axis=0),
        valid=np.concatenate([s.valid for s in sets], axis=0),
        validity_fraction=np.concatenate([s.validity_fraction for s in sets], axis=0),
        t_start=np.concatenate([s.t_start for s in sets], axis=0),
        t_end=np.concatenate([s.t_end for s in sets], axis=0),
        meta={"n_blocks": len(sets)},
    )

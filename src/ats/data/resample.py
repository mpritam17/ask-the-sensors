"""Resampling irregular sensor streams onto a uniform 25 Hz grid.

Design notes (report: "Preprocessing"):

ExtraSensory records a 20-second session once per minute. Within a session the
phone accelerometer and gyroscope run at a *nominal* 40 Hz, but the actual
inter-sample spacing is irregular (OS scheduling, thermal throttling, sensor
batching) and stretches of samples are sometimes missing entirely.

Two things follow, and both are handled here rather than swept aside:

1. Naive `scipy.signal.resample` / decimation assumes a uniform input grid.
   Applied to jittered timestamps it silently smears real motion. We instead
   interpolate against the recorded timestamps.

2. Linear interpolation across a 3-second dropout will invent a smooth ramp
   that looks like slow, steady motion — exactly the signature the recogniser
   is trained to read. So every output sample carries a validity flag: it is
   valid only if a real sample exists within `max_gap` of it. Downstream,
   windows below a validity threshold are discarded rather than classified.
   This is what lets us honestly say "no data" instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ResampledStream:
    """A uniformly sampled multi-channel stream plus per-sample validity."""

    values: np.ndarray      # (n_samples, n_channels) float32
    valid: np.ndarray       # (n_samples,) bool — backed by real measurements
    t0: float               # grid start, seconds (same base as input t)
    fs: float               # sampling rate, Hz

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.n_samples, dtype=np.float64) / self.fs

    @property
    def coverage(self) -> float:
        """Fraction of the grid backed by real measurements."""
        return float(self.valid.mean()) if self.valid.size else 0.0


def resample_to_grid(
    t: np.ndarray,
    x: np.ndarray,
    fs: float = 25.0,
    duration: Optional[float] = None,
    t0: Optional[float] = None,
    max_gap: float = 0.08,
) -> ResampledStream:
    """Resample an irregularly sampled multi-channel signal onto a uniform grid.

    Parameters
    ----------
    t : (n,) array of sample times in seconds, assumed non-decreasing.
    x : (n, c) array of channel values.
    fs : target sampling rate (Hz).
    duration : grid length in seconds. Defaults to the span of ``t``.
    t0 : grid start. Defaults to ``t[0]``.
    max_gap : an output sample is invalid if the nearest input sample is
        further away than this, in seconds.

    Returns
    -------
    ResampledStream with linearly interpolated values and a validity mask.
    Invalid positions still carry an interpolated value (so the array is
    contiguous and FFT-safe) but must not be trusted — check ``.valid``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if t.shape[0] != x.shape[0]:
        raise ValueError(f"t has {t.shape[0]} samples but x has {x.shape[0]}")

    # Guard against duplicated/out-of-order timestamps, which do occur.
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x = t[keep], x[keep]

    grid_t0 = float(t[0]) if t0 is None else float(t0)
    if duration is None:
        duration = float(t[-1] - grid_t0) if t.size > 1 else 0.0
    n_out = max(int(np.floor(duration * fs)), 0)

    n_channels = x.shape[1]
    if n_out == 0 or t.size == 0:
        return ResampledStream(
            values=np.zeros((0, n_channels), dtype=np.float32),
            valid=np.zeros((0,), dtype=bool),
            t0=grid_t0,
            fs=fs,
        )

    grid = grid_t0 + np.arange(n_out, dtype=np.float64) / fs

    if t.size == 1:
        # Degenerate stream: hold the single value, mark validity by proximity.
        values = np.repeat(x, n_out, axis=0)
        valid = _validity_mask(grid, t, max_gap)
        return ResampledStream(values.astype(np.float32), valid, grid_t0, fs)

    values = np.empty((n_out, n_channels), dtype=np.float64)
    for c in range(n_channels):
        # left/right hold the edge value; those positions are marked invalid
        # below anyway if they sit beyond max_gap from any real sample.
        values[:, c] = np.interp(grid, t, x[:, c])

    valid = _validity_mask(grid, t, max_gap)
    return ResampledStream(values.astype(np.float32), valid, grid_t0, fs)


def _validity_mask(grid: np.ndarray, t: np.ndarray, max_gap: float) -> np.ndarray:
    """True where a real sample lies within ``max_gap`` of the grid point."""
    idx = np.searchsorted(t, grid)
    idx_left = np.clip(idx - 1, 0, t.size - 1)
    idx_right = np.clip(idx, 0, t.size - 1)
    dist = np.minimum(np.abs(grid - t[idx_left]), np.abs(grid - t[idx_right]))
    # Epsilon: grid positions are accumulated k/fs, so a distance that is
    # mathematically exactly max_gap can land a few ULPs above it and flip a
    # legitimately covered sample to invalid.
    return dist <= max_gap + 1e-9


def align_streams(
    acc: ResampledStream, gyro: ResampledStream
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Stack an accelerometer and a gyroscope stream into one 6-channel block.

    Both must already sit on the same grid (same ``fs``); the shorter one
    truncates the pair. Channel order is fixed everywhere in the codebase:

        0,1,2 = Acc X, Acc Y, Acc Z
        3,4,5 = Gyro X, Gyro Y, Gyro Z

    Returns (values (n, 6), valid (n,), t0). A sample is valid only if BOTH
    modalities are valid there — an answer citing "both" modalities must be
    backed by both.
    """
    if abs(acc.fs - gyro.fs) > 1e-9:
        raise ValueError(f"sampling rates differ: {acc.fs} vs {gyro.fs}")
    if abs(acc.t0 - gyro.t0) > (0.5 / acc.fs):
        raise ValueError(f"grids are not aligned: t0 {acc.t0} vs {gyro.t0}")

    n = min(acc.n_samples, gyro.n_samples)
    values = np.concatenate([acc.values[:n], gyro.values[:n]], axis=1)
    valid = acc.valid[:n] & gyro.valid[:n]
    return values.astype(np.float32), valid, acc.t0

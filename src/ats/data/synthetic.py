"""Synthetic recordings shaped exactly like the ExtraSensory raw release.

Purpose, and its limits:

  * The real acc + gyro archives are ~15 GB. Nobody should need them to check
    that the pipeline runs, and the teaching team must be able to reproduce the
    end-to-end system from a fresh clone without a multi-hour download. This
    module writes a small dataset with the same directory layout, file format,
    minute-spaced 20 s session structure, sampling jitter and dropouts.

  * These signals are PHYSICALLY MOTIVATED PLACEHOLDERS, not real data. Any
    number computed from them is a smoke test, never a result. Every figure and
    table generated from synthetic input is stamped SYNTHETIC by the evaluation
    harness so it cannot be mistaken for a measurement (see constraints in the
    challenge brief: no invented evaluation numbers).

Units: accelerometer in g (gravity magnitude ~1.0), gyroscope in rad/s, which
matches the ExtraSensory phone-sensor convention. The dataset builder verifies
the real files against this and rescales if a user's device reported m/s^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CLASSES = [
    "lying_down", "sitting", "standing_in_place",
    "standing_and_moving", "walking", "running", "bicycling",
]

# Per-class signal recipe. Chosen so the classes are separable by the same
# features a real recogniser would use (magnitude variance, dominant frequency,
# gyroscope energy), and confusable in the same places real data is confusable
# (sitting vs lying differ only in gravity direction; walking vs running differ
# mainly in cadence and amplitude).
@dataclass(frozen=True)
class ActivityProfile:
    gravity_axis: int          # which accelerometer axis carries gravity
    gravity_sign: float
    cadence_hz: float          # dominant motion frequency (0 = none)
    acc_amplitude: float       # g, peak of the periodic component
    acc_noise: float           # g, broadband
    gyro_amplitude: float      # rad/s, periodic
    gyro_noise: float          # rad/s
    impact_sharpness: float    # 0 = smooth sinusoid, 1 = spiky heel strikes


PROFILES: Dict[str, ActivityProfile] = {
    # Lying: gravity along z, almost nothing else moving.
    "lying_down":         ActivityProfile(2, -1.0, 0.00, 0.000, 0.006, 0.000, 0.004, 0.0),
    # Sitting: phone upright-ish in a pocket, gravity along y, small fidget.
    "sitting":            ActivityProfile(1, -1.0, 0.00, 0.000, 0.012, 0.000, 0.010, 0.0),
    # Standing still: upright, postural sway around 0.3 Hz.
    "standing_in_place":  ActivityProfile(1, -1.0, 0.30, 0.020, 0.018, 0.010, 0.016, 0.0),
    # Standing and moving: upright, irregular low-frequency torso movement.
    "standing_and_moving":ActivityProfile(1, -1.0, 0.80, 0.090, 0.055, 0.090, 0.050, 0.2),
    # Walking: ~1.9 Hz cadence with heel-strike spikes.
    "walking":            ActivityProfile(1, -1.0, 1.90, 0.300, 0.060, 0.350, 0.060, 0.7),
    # Running: faster cadence, much larger amplitude, sharper impacts.
    "running":            ActivityProfile(1, -1.0, 2.80, 0.950, 0.150, 0.800, 0.120, 0.9),
    # Bicycling: smooth pedalling, no heel strikes, sustained gyro oscillation.
    "bicycling":          ActivityProfile(1, -1.0, 1.15, 0.140, 0.040, 0.450, 0.045, 0.0),
}


def _periodic(t: np.ndarray, freq: float, sharpness: float, rng: np.random.Generator
              ) -> np.ndarray:
    """A cadence waveform: sinusoid, optionally sharpened toward impact spikes."""
    if freq <= 0:
        return np.zeros_like(t)
    phase = 2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi)
    base = np.sin(phase)
    if sharpness > 0:
        # Add harmonics to steepen the peak, as a foot strike does.
        base = ((1 - sharpness) * base
                + sharpness * (np.sign(base) * np.abs(base) ** 0.4))
        base += sharpness * 0.35 * np.sin(2 * phase)
    return base


def synthesize_session(
    label: str,
    duration: float = 20.0,
    fs: float = 40.0,
    jitter: float = 0.004,
    dropout_prob: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 20-second session. Returns (t, acc (n,3), gyro (n,3))."""
    rng = rng or np.random.default_rng()
    profile = PROFILES[label]

    n = int(duration * fs)
    t = np.arange(n) / fs + rng.normal(0.0, jitter, n)
    t = np.sort(np.clip(t, 0.0, duration))

    wave = _periodic(t, profile.cadence_hz, profile.impact_sharpness, rng)

    acc = rng.normal(0.0, profile.acc_noise, (n, 3))
    acc[:, profile.gravity_axis] += profile.gravity_sign * 1.0
    # Vertical component carries most of the cadence; lateral gets a third.
    acc[:, profile.gravity_axis] += profile.acc_amplitude * wave
    acc[:, (profile.gravity_axis + 1) % 3] += 0.33 * profile.acc_amplitude * wave

    gyro = rng.normal(0.0, profile.gyro_noise, (n, 3))
    gyro[:, 0] += profile.gyro_amplitude * wave
    gyro[:, 2] += 0.5 * profile.gyro_amplitude * np.roll(wave, n // 8)

    # Slow drift, as a real device shows.
    acc += rng.normal(0.0, 0.002, (1, 3))

    # Dropout: excise a contiguous stretch, as happens when the OS deschedules
    # the sampling callback. This is what the validity mask exists to catch.
    if rng.random() < dropout_prob and n > 40:
        start = rng.integers(0, n - 30)
        length = rng.integers(10, min(120, n - start))
        keep = np.ones(n, dtype=bool)
        keep[start:start + length] = False
        t, acc, gyro = t[keep], acc[keep], gyro[keep]

    return t, acc, gyro


def write_synthetic_user(
    root: Path,
    uuid: str,
    schedule: Sequence[Tuple[str, int]],
    start_timestamp: int = 1_444_000_000,
    seed: int = 0,
    missing_example_prob: float = 0.03,
) -> pd.DataFrame:
    """Write one synthetic user in ExtraSensory raw layout.

    ``schedule`` is a list of (class, n_minutes) pairs describing the day, e.g.
    [("sitting", 30), ("walking", 12), ("sitting", 20)]. One example (a 20 s
    session) is written per minute, matching the real collection protocol.

    Returns the ground-truth per-example table (timestamp, label).
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    acc_dir = root / "raw_acc" / uuid
    gyro_dir = root / "proc_gyro" / uuid
    acc_dir.mkdir(parents=True, exist_ok=True)
    gyro_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    ts = int(start_timestamp)
    for label, minutes in schedule:
        if label not in PROFILES:
            raise ValueError(f"unknown class {label!r}")
        for _ in range(int(minutes)):
            rows.append({"timestamp": ts, "label": label})
            if rng.random() >= missing_example_prob:   # sensor sometimes absent
                t, acc, gyro = synthesize_session(label, rng=rng)
                np.savetxt(acc_dir / f"{ts}.m_raw_acc.dat",
                           np.column_stack([t, acc]), fmt="%.6f")
                np.savetxt(gyro_dir / f"{ts}.m_proc_gyro.dat",
                           np.column_stack([t, gyro]), fmt="%.6f")
            ts += 60
    truth = pd.DataFrame(rows)
    _write_label_files(root, uuid, truth)
    return truth


def _write_label_files(root: Path, uuid: str, truth: pd.DataFrame) -> None:
    """Write original-label and cleaned-label CSVs matching the real schema."""
    original_dir = root / "original_labels"
    cleaned_dir = root / "features_labels"
    original_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    original_cols = {
        "lying_down": "label:LYING_DOWN",
        "sitting": "label:SITTING",
        "standing_in_place": "label:STANDING_IN_PLACE",
        "standing_and_moving": "label:STANDING_AND_MOVING",
        "walking": "label:WALKING",
        "running": "label:RUNNING",
        "bicycling": "label:BICYCLING",
    }
    original = pd.DataFrame({"timestamp": truth["timestamp"]})
    for cls, col in original_cols.items():
        original[col] = (truth["label"] == cls).astype(int).to_numpy()
    original.to_csv(original_dir / f"{uuid}.original_labels.csv", index=False)

    cleaned_map = {
        "lying_down": "label:LYING_DOWN",
        "sitting": "label:SITTING",
        "standing_in_place": "label:OR_standing",
        "standing_and_moving": "label:OR_standing",
        "walking": "label:FIX_walking",
        "running": "label:FIX_running",
        "bicycling": "label:BICYCLING",
    }
    cleaned = pd.DataFrame({"timestamp": truth["timestamp"]})
    for col in sorted(set(cleaned_map.values())):
        active = truth["label"].map(lambda l: cleaned_map.get(l) == col)
        # Real cleaned labels carry NaN for "unknown"; reproduce that texture.
        values = np.where(active, 1.0, 0.0)
        unknown = np.random.default_rng(7).random(len(values)) < 0.15
        values[unknown & ~active.to_numpy()] = np.nan
        cleaned[col] = values
    cleaned.to_csv(cleaned_dir / f"{uuid}.features_labels.csv", index=False)


DEFAULT_SCHEDULE: List[Tuple[str, int]] = [
    ("lying_down", 25),         # early morning rest
    ("sitting", 18),            # breakfast, seated
    ("standing_and_moving", 6), # moving about the kitchen
    ("walking", 14),            # the "usual morning walk"
    ("standing_in_place", 3),   # waiting to cross
    ("walking", 9),
    ("sitting", 30),
    ("bicycling", 11),
    ("sitting", 12),
    ("running", 4),             # rare class, deliberately short
    ("walking", 7),
    ("lying_down", 40),         # prolonged afternoon rest
    ("sitting", 21),
]

"""Reading ExtraSensory raw measurement files.

Source: ExtraSensory raw measurements release,
http://extrasensory.ucsd.edu/data/raw_measurements/  (Vaizman et al., 2017).
Cited at point of use per the course academic-integrity policy.

The raw archives unzip to one small text file per (user, example-timestamp,
sensor). Each file holds the ~20-second recording session for one sensor.

IMPORTANT — verify before trusting at scale
-------------------------------------------
The exact on-disk layout (directory nesting, filename suffix, whether the first
column is a timestamp) is confirmed by running:

    python scripts/inspect_raw_layout.py --root data/raw/extrasensory

once after the first download. The reader below auto-detects the two layouts we
expect (3 columns = x,y,z with timestamps implied by the nominal rate;
4 columns = t,x,y,z) rather than hard-coding an assumption we could not check
in advance. `inspect_raw_layout.py` prints which branch fired so the choice is
recorded in the report rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Sensor -> filename fragment used by the ExtraSensory raw release.
SENSOR_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "acc": ("raw_acc", "m_raw_acc"),
    "gyro": ("proc_gyro", "m_proc_gyro", "raw_gyro"),
}

NOMINAL_HZ: Dict[str, float] = {"acc": 40.0, "gyro": 40.0}

_TIMESTAMP_RE = re.compile(r"(\d{9,13})")


@dataclass
class RawExample:
    """One sensor's 20-second session for one example timestamp."""

    uuid: str
    timestamp: int          # Unix seconds — the example's identifier
    sensor: str             # "acc" | "gyro"
    t: np.ndarray           # (n,) seconds, relative to session start
    x: np.ndarray           # (n, 3) channel values

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def parse_measurement_file(
    path: Path, sensor: str, nominal_hz: Optional[float] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse one .dat file into (t_seconds_relative, values (n, 3)).

    Returns None for dummy/empty/all-NaN files, which the dataset explicitly
    warns are present when a sensor was unavailable for an example.
    """
    try:
        arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (ValueError, OSError):
        # Some dummy files contain the literal 'nan' or are truncated.
        try:
            arr = np.genfromtxt(path, dtype=np.float64, ndmin=2)
        except Exception:
            return None

    if arr.size == 0 or arr.ndim != 2 or arr.shape[0] < 2:
        return None

    n_cols = arr.shape[1]
    if n_cols >= 4:
        # Layout A: t, x, y, z (extra columns, if any, are ignored).
        t = arr[:, 0]
        values = arr[:, 1:4]
        # Timestamps may be absolute Unix seconds or already relative.
        t = t - t[0]
        # Guard against millisecond units (a 20 s session cannot span 20000).
        if t[-1] > 200.0:
            t = t / 1000.0
    elif n_cols == 3:
        # Layout B: x, y, z only -> synthesise a uniform nominal-rate axis.
        hz = nominal_hz or NOMINAL_HZ.get(sensor, 40.0)
        values = arr[:, 0:3]
        t = np.arange(arr.shape[0], dtype=np.float64) / hz
    else:
        return None

    finite = np.isfinite(values).all(axis=1) & np.isfinite(t)
    if finite.sum() < 2:
        return None
    return t[finite], values[finite]


def _timestamp_from_name(path: Path) -> Optional[int]:
    match = _TIMESTAMP_RE.search(path.stem)
    if not match:
        return None
    value = int(match.group(1))
    # 13-digit values are milliseconds.
    return value // 1000 if value > 10_000_000_000 else value


def discover_sensor_files(root: Path, sensor: str) -> Dict[str, Dict[int, Path]]:
    """Index the unzipped raw tree as {uuid: {timestamp: path}}.

    Layout-agnostic: it globs for files whose *path* contains a known sensor
    fragment and whose name contains a Unix timestamp, then reads the UUID from
    whichever path component looks like a UUID. This survives the archive being
    nested as <sensor>/<uuid>/<ts>.dat or <uuid>/<ts>.<sensor>.dat.
    """
    root = Path(root)
    patterns = SENSOR_PATTERNS[sensor]
    index: Dict[str, Dict[int, Path]] = {}
    uuid_re = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
                         r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        joined = str(path).lower()
        if not any(p in joined for p in patterns):
            continue
        timestamp = _timestamp_from_name(path)
        if timestamp is None:
            continue
        uuid = next((part for part in path.parts if uuid_re.match(part)), None)
        if uuid is None:
            continue
        index.setdefault(uuid, {})[timestamp] = path
    return index


def iter_examples(
    root: Path, uuid: str, timestamps: Optional[List[int]] = None
) -> Iterator[Tuple[int, Optional[RawExample], Optional[RawExample]]]:
    """Yield (timestamp, acc_example, gyro_example) for one user.

    Either element may be None when that sensor was unavailable — the caller
    decides whether to drop the example or fall back to a single modality.
    """
    acc_index = discover_sensor_files(root, "acc").get(uuid, {})
    gyro_index = discover_sensor_files(root, "gyro").get(uuid, {})
    keys = sorted(set(acc_index) | set(gyro_index))
    if timestamps is not None:
        wanted = set(timestamps)
        keys = [k for k in keys if k in wanted]

    for ts in keys:
        out: List[Optional[RawExample]] = []
        for sensor, index in (("acc", acc_index), ("gyro", gyro_index)):
            path = index.get(ts)
            parsed = parse_measurement_file(path, sensor) if path else None
            if parsed is None:
                out.append(None)
            else:
                t, values = parsed
                out.append(RawExample(uuid, ts, sensor, t, values))
        yield ts, out[0], out[1]

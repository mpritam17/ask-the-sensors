#!/usr/bin/env python3
"""Report the on-disk layout of the unzipped ExtraSensory raw measurements.

Run this ONCE after the first download and paste the output into the report.
It confirms, rather than assumes, the facts the reader in src/ats/data/raw_io.py
depends on: directory nesting, filename pattern, column count, whether column 0
is a timestamp, actual sampling rate, and unit convention (g vs m/s^2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ats.data import raw_io  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/extrasensory")
    parser.add_argument("--n", type=int, default=3, help="files to inspect per sensor")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"{root} does not exist — run scripts/fetch_extrasensory.py first")
        return 1

    for sensor in ("acc", "gyro"):
        index = raw_io.discover_sensor_files(root, sensor)
        n_users = len(index)
        n_files = sum(len(v) for v in index.values())
        print(f"\n=== {sensor} ===")
        print(f"users: {n_users}   files: {n_files}")
        if not index:
            print("  no files matched — check SENSOR_PATTERNS in src/ats/data/raw_io.py")
            continue

        uuid = sorted(index)[0]
        for ts in sorted(index[uuid])[: args.n]:
            path = index[uuid][ts]
            print(f"\n  {path.relative_to(root)}")
            with open(path, "r", errors="replace") as handle:
                first = handle.readline().rstrip()
            print(f"    first line : {first[:100]}")
            parsed = raw_io.parse_measurement_file(path, sensor)
            if parsed is None:
                print("    -> parsed as dummy/empty (sensor unavailable)")
                continue
            t, x = parsed
            span = float(t[-1] - t[0]) if t.size > 1 else 0.0
            rate = (t.size - 1) / span if span > 0 else float("nan")
            mag = float(np.median(np.linalg.norm(x, axis=1)))
            print(f"    samples={t.size}  span={span:.2f}s  rate={rate:.1f}Hz"
                  f"  median|x|={mag:.3f}")
            if sensor == "acc":
                unit = "g" if mag < 5 else "m/s^2 (will be rescaled to g)"
                print(f"    accelerometer units look like: {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

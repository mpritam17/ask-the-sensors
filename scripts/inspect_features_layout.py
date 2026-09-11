#!/usr/bin/env python3
"""Inspect one official ExtraSensory features/labels file without training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.data.precomputed import sensor_feature_columns  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    frame = pd.read_csv(path, nrows=5)
    acc = sensor_feature_columns(frame.columns, "acc")
    gyro = sensor_feature_columns(frame.columns, "gyro")
    labels = [column for column in frame.columns if "label:" in column.lower()]
    print(f"path: {path}")
    print(f"columns: {len(frame.columns)}")
    print(f"raw_acc feature columns: {len(acc)}")
    print(f"proc/raw gyro feature columns: {len(gyro)}")
    print(f"label columns: {len(labels)}")
    print("first accelerometer features:", acc[:8])
    print("first gyroscope features:", gyro[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

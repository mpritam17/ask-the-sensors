#!/usr/bin/env python3
"""Generate the small, explicitly synthetic CLI demonstration recording."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.data.synthetic import synthesize_session  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    demo = root / "demo"
    demo.mkdir(exist_ok=True)
    rng = np.random.default_rng(31)
    schedule = [("sitting", 7.0), ("walking", 9.0), ("running", 7.0), ("lying_down", 12.0)]
    blocks, intervals = [], []
    offset = 0.0
    for activity, duration in schedule:
        t, acc, gyro = synthesize_session(
            activity, duration=duration, dropout_prob=0.0, rng=rng
        )
        blocks.append(np.column_stack([t + offset, acc, gyro]))
        intervals.append({"activity": activity, "start": offset, "end": offset + duration})
        offset += duration
    frame = pd.DataFrame(np.vstack(blocks), columns=[
        "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"
    ])
    frame.to_csv(demo / "recording.csv", index=False, float_format="%.6f")
    (demo / "expected_intervals.json").write_text(
        json.dumps({"dataset_kind": "synthetic", "intervals": intervals}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(demo / "recording.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

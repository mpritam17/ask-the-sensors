#!/usr/bin/env python3
"""Inspect and validate one official ExtraSensory original-label file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.data.labels import build_label_table, label_summary, read_user_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)
    raw = read_user_csv(path)
    table = build_label_table(path)
    print(f"path: {path}")
    print(f"rows: {len(raw)}")
    print(f"columns: {len(raw.columns)}")
    print(f"timestamp range: {int(raw['timestamp'].min())} .. {int(raw['timestamp'].max())}")
    print("seven-class parse summary:")
    print(label_summary(table).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

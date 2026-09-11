#!/usr/bin/env python3
"""Generate all five required figures from a saved evaluation JSON file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.eval.figures import generate_from_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()
    for path in generate_from_json(Path(args.results), Path(args.out)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

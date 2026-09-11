#!/usr/bin/env python3
"""Generate a small synthetic dataset in the ExtraSensory raw layout.

Lets anyone (including the teaching team) run the full pipeline end to end
without the ~15 GB download. Results computed from this data are smoke tests,
never reported numbers — see the warning in src/ats/data/synthetic.py.

    python scripts/make_synthetic_data.py --users 4 --out data/raw/synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ats.data.synthetic import DEFAULT_SCHEDULE, write_synthetic_user  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--out", default="data/raw/synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every schedule block length")
    args = parser.parse_args()

    root = Path(args.out)
    for i in range(args.users):
        uuid = f"{i:08X}-0000-4000-8000-{i:012X}".upper()
        schedule = [(label, max(1, int(round(n * args.scale))))
                    for label, n in DEFAULT_SCHEDULE]
        truth = write_synthetic_user(root, uuid, schedule,
                                     start_timestamp=1_444_000_000 + i * 86_400,
                                     seed=args.seed + i)
        minutes = len(truth)
        print(f"user {uuid}: {minutes} examples ({minutes/60:.1f} h simulated)")
    print(f"\nWritten to {root}")
    print("Next: python scripts/build_dataset.py --raw-root", root, "--synthetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

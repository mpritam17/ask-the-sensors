#!/usr/bin/env python3
"""Create the reproducible user-disjoint split manifest from processed NPZs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.splits import make_user_splits, write_split_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=None, help="directory containing per-user NPZs")
    parser.add_argument("--out", default="artifacts/split_manifest.json")
    args = parser.parse_args()

    cfg = load_config("data")
    processed = Path(args.processed) if args.processed else resolve(cfg["output"]["processed_dir"])
    users = sorted(p.stem for p in processed.glob("*.npz"))
    if len(users) < 4:
        parser.error(f"need at least four user NPZs in {processed}; found {len(users)}")

    split_cfg = cfg["split"]
    splits = make_user_splits(
        users,
        seed=int(split_cfg["seed"]),
        n_qa_users=int(split_cfg["n_qa_users"]),
    )
    path = write_split_manifest(
        splits, resolve(args.out), seed=int(split_cfg["seed"]), source="processed_users"
    )
    for name, values in splits.items():
        print(f"{name:18s}: {len(values):3d}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

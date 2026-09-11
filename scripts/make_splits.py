#!/usr/bin/env python3
"""Create the reproducible user-disjoint split manifest from processed NPZs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.splits import load_official_fold_map, make_user_splits, write_split_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=None, help="directory containing per-user NPZs")
    parser.add_argument("--cv-folds", default=None, help="directory containing official cv_5_folds")
    parser.add_argument("--out", default="artifacts/split_manifest.json")
    args = parser.parse_args()

    cfg = load_config("data")
    processed = Path(args.processed) if args.processed else resolve(cfg["output"]["processed_dir"])
    users = sorted(p.stem for p in processed.glob("*.npz"))
    if len(users) < 4:
        parser.error(f"need at least four user NPZs in {processed}; found {len(users)}")

    split_cfg = cfg["split"]
    fold_root = (
        Path(args.cv_folds)
        if args.cv_folds
        else resolve(cfg["dataset"]["root"]) / "cv_folds"
    )
    official_folds = load_official_fold_map(fold_root) if fold_root.exists() else None
    splits = make_user_splits(
        users,
        seed=int(split_cfg["seed"]),
        n_qa_users=int(split_cfg["n_qa_users"]),
        official_folds=official_folds,
        test_fold=int(split_cfg["fold"]),
    )
    source = f"official_cv5_fold_{split_cfg['fold']}" if official_folds else "seeded_processed_users"
    path = write_split_manifest(
        splits, resolve(args.out), seed=int(split_cfg["seed"]), source=source
    )
    for name, values in splits.items():
        print(f"{name:18s}: {len(values):3d}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

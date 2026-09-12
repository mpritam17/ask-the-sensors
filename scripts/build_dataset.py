#!/usr/bin/env python3
"""Build windowed, labelled arrays from raw ExtraSensory (or synthetic) data.

    # real data, after scripts/fetch_extrasensory.py
    python scripts/build_dataset.py

    # synthetic smoke test, no download needed
    python scripts/make_synthetic_data.py --users 3
    python scripts/build_dataset.py --raw-root data/raw/synthetic --synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from ats.config import load_config, resolve  # noqa: E402
from ats.data.build_dataset import (  # noqa: E402
    build_user, manifest_summary, save_user, write_manifest,
)
from ats.data import raw_io  # noqa: E402


def _label_paths(root: Path, uuid: str, synthetic: bool):
    """Locate this user's original-label and cleaned-label files."""
    original_dir = root / "original_labels"
    cleaned_dir = root / "features_labels"
    original = next(original_dir.glob(f"{uuid}*"), None) if original_dir.exists() else None
    cleaned = next(cleaned_dir.glob(f"{uuid}*"), None) if cleaned_dir.exists() else None
    return original, cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--users", nargs="*", default=None, help="UUID subset")
    parser.add_argument("--limit", type=int, default=None, help="max users to build")
    parser.add_argument("--synthetic", action="store_true",
                        help="tag outputs as synthetic so no figure is mistaken for a result")
    parser.add_argument(
        "--modalities",
        choices=("acc", "both"),
        default="both",
        help="build both sensors, or an explicit accelerometer-only fallback",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    label_cfg = load_config("labels")
    raw_root = Path(args.raw_root) if args.raw_root else resolve(data_cfg["dataset"]["root"])
    out_dir = Path(args.out) if args.out else resolve(data_cfg["output"]["processed_dir"])
    if args.synthetic:
        out_dir = out_dir / "synthetic"

    if not raw_root.exists():
        print(f"raw root not found: {raw_root}")
        return 1

    print("Indexing sensor files once ...")
    acc_by_user = raw_io.discover_sensor_files(raw_root, "acc")
    gyro_by_user = (
        raw_io.discover_sensor_files(raw_root, "gyro")
        if args.modalities == "both" else {}
    )
    uuids = args.users or sorted(acc_by_user)
    if args.limit:
        uuids = uuids[: args.limit]
    if not uuids:
        print("no users found — check the raw layout with scripts/inspect_raw_layout.py")
        return 1

    print(
        f"Building {len(uuids)} user(s) from {raw_root} -> {out_dir} "
        f"(modalities={args.modalities})"
    )
    reports = []
    for uuid in uuids:
        original, cleaned = _label_paths(raw_root, uuid, args.synthetic)
        if original is None:
            print(f"  {uuid}: no original-label file, skipped")
            continue
        arrays, report = build_user(raw_root, uuid, original, cleaned,
                                    data_cfg=data_cfg, label_cfg=label_cfg,
                                    acc_index=acc_by_user.get(uuid, {}),
                                    gyro_index=gyro_by_user.get(uuid, {}),
                                    modalities=args.modalities)
        reports.append(report)
        if arrays is None:
            print(f"  {uuid}: produced no usable windows "
                  f"(labelled={report.n_examples_labelled}, "
                  f"no_sensor={report.dropped_no_sensor})")
            continue
        path = save_user(arrays, out_dir, uuid)
        print(f"  {uuid}: {report.n_windows_valid:6d} valid windows "
              f"from {report.n_examples_used} examples -> {path.name}")

    if reports:
        write_manifest(reports, out_dir)
        summary = manifest_summary(reports)
        pd.set_option("display.width", 160)
        print("\nPer-user class counts (valid windows):")
        print(summary.to_string(index=False))
        totals = summary.drop(columns=["uuid"]).sum()
        print("\nTotals:")
        print(totals.to_string())
        if args.synthetic:
            print("\n*** SYNTHETIC DATA — for pipeline smoke testing only. ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

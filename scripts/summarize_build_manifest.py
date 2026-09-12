#!/usr/bin/env python3
"""Create a compact, auditable summary of a raw dataset build manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    reports = json.loads(raw)
    if not isinstance(reports, list) or not reports:
        raise ValueError("build manifest must contain a non-empty list")
    additive = (
        "n_examples_seen",
        "n_examples_labelled",
        "n_examples_used",
        "n_windows",
        "n_windows_valid",
        "dropped_no_sensor",
        "dropped_low_coverage",
    )
    totals = {
        key: sum(int(report.get(key, 0)) for report in reports) for key in additive
    }
    labelled_without_indexed_example = sum(
        max(
            0,
            int(report.get("n_examples_labelled", 0))
            - int(report.get("n_examples_seen", 0)),
        )
        for report in reports
    )
    class_counts: Counter[str] = Counter()
    for report in reports:
        class_counts.update(
            {str(key): int(value) for key, value in report.get("class_counts", {}).items()}
        )
    return {
        "dataset_kind": "real",
        "source": "official ExtraSensory raw-measurement build manifest",
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "users": len(reports),
        "modalities": sorted({str(report.get("modalities", "both")) for report in reports}),
        "users_with_accelerometer_unit_rescaling": sum(
            bool(report.get("unit_rescaled")) for report in reports
        ),
        "n_labelled_without_indexed_example": labelled_without_indexed_example,
        **totals,
        "class_counts_valid_windows": dict(class_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = summarize(Path(args.manifest))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

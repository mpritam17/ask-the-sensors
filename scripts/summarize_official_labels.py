#!/usr/bin/env python3
"""Save aggregate seven-class and filtering counts from official metadata."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.labels import build_label_table  # noqa: E402


def uuid_from_path(path: Path) -> str:
    return path.name.split(".", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="artifacts/official_label_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    originals = {
        uuid_from_path(path): path
        for path in (root / "original_labels").glob("*.csv*")
    }
    cleaned = {
        uuid_from_path(path): path
        for path in (root / "features_labels").glob("*.csv*")
    }
    if not originals:
        parser.error(f"no original-label CSV files found under {root}")

    classes = list(load_config("labels")["classes"])
    class_counts = Counter({name: 0 for name in classes})
    drop_counts: Counter[str] = Counter()
    total_rows = 0
    retained_rows = 0
    filtered_users = 0
    for index, (uuid, original_path) in enumerate(sorted(originals.items()), start=1):
        cleaned_path = cleaned.get(uuid)
        filtered_users += int(cleaned_path is not None)
        table = build_label_table(original_path, cleaned_path)
        total_rows += len(table)
        kept = table.loc[table["kept"]]
        retained_rows += len(kept)
        class_counts.update(kept["label"].value_counts().to_dict())
        drop_counts.update(table.loc[~table["kept"], "drop_reason"].value_counts().to_dict())
        if index % 10 == 0 or index == len(originals):
            print(f"summarized {index}/{len(originals)} users", flush=True)

    payload = {
        "dataset_kind": "real",
        "source": "official ExtraSensory per-user metadata",
        "users": len(originals),
        "users_with_cleaned_consistency_filter": filtered_users,
        "total_examples": total_rows,
        "retained_examples": retained_rows,
        "retained_fraction": retained_rows / total_rows,
        "class_counts": dict(class_counts),
        "drop_reasons": dict(drop_counts),
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

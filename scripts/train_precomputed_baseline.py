#!/usr/bin/env python3
"""Train an explicitly separate real-data baseline from official sensor features."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.precomputed import feature_provenance_from_columns, load_precomputed_user  # noqa: E402
from ats.data.splits import make_user_splits, write_split_manifest  # noqa: E402
from ats.recognition import classification_metrics, predict_probabilities, save_bundle, train_and_select  # noqa: E402


def uuid_from_path(path: Path) -> str:
    return path.name.split(".", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="download root containing original_labels/features_labels")
    parser.add_argument("--modalities", choices=("acc", "gyro", "both"), default="both")
    parser.add_argument("--max-per-class-user", type=int, default=3000)
    parser.add_argument("--model", default="artifacts/precomputed_real_recognizer.joblib")
    parser.add_argument("--splits", default="artifacts/precomputed_real_split_manifest.json")
    parser.add_argument("--results", default="artifacts/precomputed_real_results.json")
    args = parser.parse_args()

    root = Path(args.root)
    feature_paths = {uuid_from_path(path): path for path in (root / "features_labels").glob("*.csv*")}
    original_paths = {uuid_from_path(path): path for path in (root / "original_labels").glob("*.csv*")}
    users = sorted(set(feature_paths) & set(original_paths))
    if len(users) < 4:
        parser.error(f"need at least four matched official users; found {len(users)}")

    data_cfg = load_config("data")
    seed = int(data_cfg["split"]["seed"])
    splits = make_user_splits(users, seed=seed, n_qa_users=int(data_cfg["split"]["n_qa_users"]))
    split_path = resolve(args.splits)
    write_split_manifest(splits, split_path, seed=seed, source="official_precomputed_features")

    cached = {}
    schema = None
    rng = np.random.default_rng(seed)
    for uuid in users:
        x, y, columns = load_precomputed_user(
            feature_paths[uuid], original_paths[uuid],
            feature_columns=schema, modalities=args.modalities,
        )
        schema = columns
        chosen = []
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            if len(indices) > args.max_per_class_user:
                indices = rng.choice(indices, args.max_per_class_user, replace=False)
            chosen.extend(indices.tolist())
        chosen = np.asarray(sorted(chosen), dtype=int)
        cached[uuid] = (x[chosen], y[chosen])

    def partition(name):
        return (
            np.concatenate([cached[uuid][0] for uuid in splits[name]]),
            np.concatenate([cached[uuid][1] for uuid in splits[name]]),
        )

    x_train, y_train = partition("train")
    x_validation, y_validation = partition("validation")
    classes = list(load_config("labels")["classes"])
    model_cfg = load_config("model")["recognition"]
    bundle, candidates = train_and_select(
        x_train, y_train, x_validation, y_validation,
        classes=classes,
        config=model_cfg,
        dataset_kind="real",
        schema=schema,
        provenance=feature_provenance_from_columns(schema),
        feature_source=f"official_precomputed_{args.modalities}",
    )
    save_bundle(bundle, resolve(args.model))
    x_test, y_test = partition("recognition_test")
    prediction = np.argmax(predict_probabilities(bundle, x_test), axis=1)
    payload = {
        "dataset_kind": "real",
        "scope": "official precomputed sensor-feature baseline; not raw-window end-to-end QA",
        "modalities": args.modalities,
        "feature_count": len(schema),
        "selected_model": bundle["selected_model"],
        "candidate_validation": candidates,
        "recognition_test": classification_metrics(y_test, prediction, classes),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        "split_manifest": str(split_path),
    }
    results = resolve(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(results)
    print(json.dumps(payload["recognition_test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

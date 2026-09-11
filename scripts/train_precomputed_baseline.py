#!/usr/bin/env python3
"""Train an explicitly separate real-data baseline from official sensor features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.precomputed import feature_provenance_from_columns, load_precomputed_user  # noqa: E402
from ats.data.splits import load_official_fold_map, make_user_splits, write_split_manifest  # noqa: E402
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
    fold_root = root / "cv_folds"
    official_folds = load_official_fold_map(fold_root)
    test_fold = int(data_cfg["split"]["fold"])
    splits = make_user_splits(
        users,
        seed=seed,
        n_qa_users=int(data_cfg["split"]["n_qa_users"]),
        official_folds=official_folds,
        test_fold=test_fold,
    )
    split_path = resolve(args.splits)
    write_split_manifest(
        splits,
        split_path,
        seed=seed,
        source=f"official_precomputed_features_cv5_fold_{test_fold}",
    )

    cached = {}
    schema = None
    rng = np.random.default_rng(seed)
    for user_index, uuid in enumerate(users, start=1):
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
        if user_index % 10 == 0 or user_index == len(users):
            print(
                f"loaded {user_index}/{len(users)} users; "
                f"latest retained rows={len(chosen)}",
                flush=True,
            )

    def partition(name):
        x = np.concatenate([cached[uuid][0] for uuid in splits[name]])
        y = np.concatenate([cached[uuid][1] for uuid in splits[name]])
        if not len(y):
            raise ValueError(f"{name} partition contains no retained examples")
        return x, y

    def class_counts(values):
        counts = np.bincount(values, minlength=len(classes))
        return {name: int(counts[index]) for index, name in enumerate(classes)}

    classes = list(load_config("labels")["classes"])
    x_train, y_train = partition("train")
    x_validation, y_validation = partition("validation")
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
    model_path = save_bundle(bundle, resolve(args.model))
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    x_test, y_test = partition("recognition_test")
    prediction = np.argmax(predict_probabilities(bundle, x_test), axis=1)
    recognition_metrics = classification_metrics(y_test, prediction, classes)
    payload = {
        "dataset_kind": "real",
        "scope": "official precomputed sensor-feature baseline; not raw-window end-to-end QA",
        "modalities": args.modalities,
        "feature_count": len(schema),
        "selected_model": bundle["selected_model"],
        "model_disk_bytes": model_path.stat().st_size,
        "model_sha256": model_sha256,
        "model_config_sha256": bundle["config_sha256"],
        "candidate_validation": candidates,
        "recognition_test": recognition_metrics,
        "confusion_matrix": {
            "classes": classes,
            "matrix": confusion_matrix(
                y_test, prediction, labels=np.arange(len(classes))
            ).tolist(),
        },
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        "users": {name: len(split_users) for name, split_users in splits.items()},
        "class_counts": {
            "train": class_counts(y_train),
            "validation": class_counts(y_validation),
            "recognition_test": class_counts(y_test),
        },
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

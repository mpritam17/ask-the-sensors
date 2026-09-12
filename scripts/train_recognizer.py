#!/usr/bin/env python3
"""Train and select the seven-class recognizer on user-disjoint data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.splits import make_user_splits, write_split_manifest  # noqa: E402
from ats.features import (  # noqa: E402
    extract_features,
    feature_indices_for_modalities,
    feature_names,
    feature_provenance,
)
from ats.recognition import classification_metrics, predict_probabilities, save_bundle, train_and_select  # noqa: E402


def load_partition(processed: Path, users, cap: int, seed: int, feature_indices=None):
    x_blocks, y_blocks = [], []
    rng = np.random.default_rng(seed)
    classes = None
    for uuid in users:
        with np.load(processed / f"{uuid}.npz", allow_pickle=True) as data:
            valid = np.asarray(data["valid"], dtype=bool)
            windows = np.asarray(data["windows"])[valid]
            labels = np.asarray(data["y"], dtype=int)[valid]
            classes = [str(x) for x in data["classes"]]
        selected = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if len(indices) > cap:
                indices = rng.choice(indices, cap, replace=False)
            selected.extend(indices.tolist())
        selected = np.asarray(sorted(selected), dtype=int)
        features = extract_features(windows[selected])
        if feature_indices is not None:
            features = features[:, feature_indices]
        x_blocks.append(features)
        y_blocks.append(labels[selected])
    if not x_blocks:
        raise ValueError("partition contains no usable windows")
    return np.concatenate(x_blocks), np.concatenate(y_blocks), classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--splits", default="artifacts/split_manifest.json")
    parser.add_argument("--model", default="artifacts/recognizer.joblib")
    parser.add_argument("--results", default="artifacts/recognition_results.json")
    parser.add_argument("--synthetic", action="store_true", help="stamp outputs as synthetic smoke-test data")
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")["recognition"]
    processed = Path(args.processed) if args.processed else resolve(data_cfg["output"]["processed_dir"])
    users = sorted(p.stem for p in processed.glob("*.npz"))
    if len(users) < 4:
        parser.error(f"need at least four processed users in {processed}; found {len(users)}")

    with np.load(processed / f"{users[0]}.npz", allow_pickle=True) as first:
        available_modalities = (
            [str(item) for item in first["modalities"]]
            if "modalities" in first.files
            else ["accelerometer", "gyroscope"]
        )
    for uuid in users[1:]:
        with np.load(processed / f"{uuid}.npz", allow_pickle=True) as data:
            current = (
                [str(item) for item in data["modalities"]]
                if "modalities" in data.files
                else ["accelerometer", "gyroscope"]
            )
        if current != available_modalities:
            parser.error("processed users mix incompatible sensor modality sets")
    feature_indices = feature_indices_for_modalities(available_modalities)
    schema = [feature_names()[index] for index in feature_indices]
    provenance = feature_provenance(schema)

    split_path = resolve(args.splits)
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    else:
        splits = make_user_splits(
            users,
            seed=int(data_cfg["split"]["seed"]),
            n_qa_users=int(data_cfg["split"]["n_qa_users"]),
        )
        write_split_manifest(
            splits,
            split_path,
            seed=int(data_cfg["split"]["seed"]),
            source="synthetic_processed_users" if args.synthetic else "real_processed_users",
        )

    cap = int(load_config("labels")["imbalance"]["max_windows_per_class_per_user"])
    seed = int(model_cfg["random_seed"])
    x_train, y_train, classes = load_partition(
        processed, splits["train"], cap, seed, feature_indices
    )
    x_val, y_val, _ = load_partition(
        processed, splits["validation"], cap, seed + 1, feature_indices
    )
    bundle, candidates = train_and_select(
        x_train,
        y_train,
        x_val,
        y_val,
        classes=classes,
        config=model_cfg,
        dataset_kind="synthetic" if args.synthetic else "real",
        schema=schema,
        provenance=provenance,
        feature_source=(
            "engineered_windows" if len(available_modalities) == 2
            else "engineered_windows_accelerometer"
        ),
        available_modalities=available_modalities,
    )
    model_path = save_bundle(bundle, resolve(args.model))

    x_test, y_test, _ = load_partition(
        processed, splits["recognition_test"], cap, seed + 2, feature_indices
    )
    test_prediction = np.argmax(predict_probabilities(bundle, x_test), axis=1)
    payload = {
        "dataset_kind": bundle["dataset_kind"],
        "selected_model": bundle["selected_model"],
        "config_sha256": bundle["config_sha256"],
        "candidate_validation": candidates,
        "recognition_test": classification_metrics(y_test, test_prediction, classes),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "available_modalities": available_modalities,
    }
    result_path = resolve(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"selected={bundle['selected_model']} model={model_path}")
    print(json.dumps(payload["recognition_test"], indent=2))
    if args.synthetic:
        print("*** SYNTHETIC SMOKE-TEST RESULT - NOT A REAL-DATA METRIC ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

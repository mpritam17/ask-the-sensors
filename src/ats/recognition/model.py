"""Class-balanced activity recognizers and validated model bundles."""
from __future__ import annotations

import hashlib
import json
import pickle
from time import perf_counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ats.features import feature_names, feature_provenance

BUNDLE_VERSION = 1


def make_candidates(config: Mapping[str, object], seed: int) -> Dict[str, object]:
    candidates = config["candidates"]
    return {
        "logistic": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=int(candidates["logistic"]["max_iter"]),
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "compact_rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=int(candidates["compact_rf"]["n_estimators"]),
                max_depth=int(candidates["compact_rf"]["max_depth"]),
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "full_rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=int(candidates["full_rf"]["n_estimators"]),
                max_depth=int(candidates["full_rf"]["max_depth"]),
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_select(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    classes: List[str],
    config: Mapping[str, object],
    dataset_kind: str,
    schema: List[str] | None = None,
    provenance: List[Mapping[str, object]] | None = None,
    feature_source: str = "engineered_windows",
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    seed = int(config["random_seed"])
    results: Dict[str, Dict[str, object]] = {}
    fitted: Dict[str, object] = {}
    for name, estimator in make_candidates(config, seed).items():
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_validation)
        metrics = classification_metrics(y_validation, prediction, classes)
        metrics["serialized_bytes"] = len(pickle.dumps(estimator, protocol=5))
        for _ in range(3):
            estimator.predict(x_validation)
        timings = []
        for _ in range(30):
            started = perf_counter()
            estimator.predict(x_validation)
            timings.append((perf_counter() - started) * 1000.0 / max(len(x_validation), 1))
        metrics["median_latency_ms_per_window"] = float(np.median(timings))
        metrics["p95_latency_ms_per_window"] = float(np.percentile(timings, 95))
        results[name] = metrics
        fitted[name] = estimator

    best_score = max(float(v["macro_f1"]) for v in results.values())
    tolerance = float(config["selection_tolerance_macro_f1"])
    eligible = [
        name for name, values in results.items()
        if float(values["macro_f1"]) >= best_score - tolerance
    ]
    selected = min(eligible, key=lambda name: int(results[name]["serialized_bytes"]))
    schema = list(schema or feature_names())
    provenance = list(provenance or feature_provenance(schema))
    config_json = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "estimator": fitted[selected],
        "selected_model": selected,
        "classes": list(classes),
        "feature_names": schema,
        "feature_provenance": provenance,
        "feature_source": feature_source,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "dataset_kind": dataset_kind,
        "validation_metrics": results[selected],
    }
    return bundle, results


def save_bundle(bundle: Mapping[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), path, compress=3)
    return path


def load_bundle(path: Path) -> Dict[str, object]:
    bundle = joblib.load(path)
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"unsupported recognizer bundle version: {bundle.get('bundle_version')}")
    if bundle.get("feature_source", "engineered_windows") == "engineered_windows" and list(bundle.get("feature_names", [])) != feature_names():
        raise ValueError("recognizer feature schema does not match this source version")
    return bundle


def predict_probabilities(bundle: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    estimator = bundle["estimator"]
    probabilities = estimator.predict_proba(features)
    estimator_classes = np.asarray(estimator.classes_, dtype=int)
    full = np.zeros((len(features), len(bundle["classes"])), dtype=np.float64)
    full[:, estimator_classes] = probabilities
    return full

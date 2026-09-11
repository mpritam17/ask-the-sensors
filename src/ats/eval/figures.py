"""Generate the five required figures strictly from a saved results payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _stamp(ax, dataset_kind: str) -> None:
    if dataset_kind != "real":
        ax.text(
            0.5, 0.5, "SYNTHETIC\nNOT A REAL-DATA RESULT", transform=ax.transAxes,
            ha="center", va="center", rotation=22, fontsize=11, color="#B00020",
            alpha=0.28, weight="bold",
        )


def _finish(fig, ax, path: Path, dataset_kind: str) -> None:
    _stamp(ax, dataset_kind)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_required_figures(payload: Mapping[str, object], out_dir: Path) -> list[Path]:
    required = {"dataset_kind", "accuracy_by_question_type", "confusion_matrix", "strictness", "overhead", "robustness"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"results payload is missing: {sorted(missing)}")
    dataset_kind = str(payload["dataset_kind"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    values = payload["accuracy_by_question_type"]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    names, scores = list(values), [float(values[name]) for name in values]
    ax.bar(names, scores, color="#4472C4")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by question type")
    ax.tick_params(axis="x", rotation=25)
    path = out_dir / "figure_accuracy_by_question_type.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    confusion = payload["confusion_matrix"]
    matrix = np.asarray(confusion["matrix"], dtype=float)
    labels = list(confusion["classes"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Seven-class activity confusion matrix")
    path = out_dir / "figure_confusion_matrix.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    strictness = payload["strictness"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(strictness["threshold"], strictness["accuracy"], marker="o", color="#70AD47")
    ax.set_xlabel("Temporal IoU threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy versus strictness")
    path = out_dir / "figure_accuracy_vs_strictness.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    points = payload["overhead"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for point in points:
        x, y = float(point["latency_ms"]), float(point["accuracy"])
        ax.scatter(x, y, s=55)
        ax.annotate(
            str(point["model"]), (x, y), xytext=(5, -14),
            textcoords="offset points", va="top",
        )
    ax.set_xlabel("Median latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy versus overhead")
    path = out_dir / "figure_accuracy_vs_overhead.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    robustness = payload["robustness"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(robustness["dropped_percent"], robustness["accuracy"], marker="o", color="#ED7D31")
    ax.set_xlabel("Dropped input samples (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Robustness curve")
    path = out_dir / "figure_robustness.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)
    return paths


def generate_from_json(results_path: Path, out_dir: Path) -> list[Path]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return generate_required_figures(payload, out_dir)

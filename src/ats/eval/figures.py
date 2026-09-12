"""Generate the five required figures strictly from a saved results payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np


def _stamp(ax, dataset_kind: str) -> None:
    if dataset_kind != "real":
        ax.text(
            0.5, 0.5, "SYNTHETIC\nNOT A REAL-DATA RESULT", transform=ax.transAxes,
            ha="center", va="center", rotation=22, fontsize=11, color="#B00020",
            alpha=0.28, weight="bold",
        )


def _finish(fig, ax, path: Path, dataset_kind: str) -> None:
    for axis in np.atleast_1d(ax).flat:
        _stamp(axis, dataset_kind)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scope_label(payload: Mapping[str, object]) -> str:
    if str(payload["dataset_kind"]) != "real":
        return "Synthetic smoke test"
    modalities = set(map(str, payload.get("available_modalities", [])))
    if modalities == {"accelerometer"}:
        return "Real raw-window accelerometer baseline"
    return "Real raw-window dual-sensor system"


def generate_required_figures(payload: Mapping[str, object], out_dir: Path) -> list[Path]:
    required = {"dataset_kind", "accuracy_by_question_type", "confusion_matrix", "strictness", "overhead", "robustness"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"results payload is missing: {sorted(missing)}")
    dataset_kind = str(payload["dataset_kind"])
    scope_label = _scope_label(payload)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    values = payload["accuracy_by_question_type"]
    names = [str(name).replace("_", " ") for name in values] + ["overall"]
    scores = [float(score) for score in values.values()]
    scores.append(float(payload.get("overall_accuracy_macro", np.mean(scores))))
    grounded_values = payload.get("grounded_accuracy_by_question_type")
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    x = np.arange(len(names))
    if grounded_values:
        grounded = [float(grounded_values[name]) for name in values]
        grounded.append(float(payload.get("overall_grounded_accuracy_macro", np.mean(grounded))))
        width = 0.39
        ax.bar(x - width / 2, scores, width, color="#4472C4", label="Answer correct")
        ax.bar(x + width / 2, grounded, width, color="#70AD47", label="Answer + evidence correct")
        ax.legend(loc="upper right", fontsize=7)
        for index, score in enumerate(grounded):
            ax.text(index + width / 2, score + 0.022, f"{score:.2f}", ha="center", fontsize=6)
    else:
        ax.bar(x, scores, color="#4472C4")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{scope_label}: accuracy by question type")
    ax.set_xticks(x, names, rotation=28, ha="right")
    for index, score in enumerate(scores):
        position = index - width / 2 if grounded_values else index
        ax.text(position, score + 0.022, f"{score:.2f}", ha="center", fontsize=6)
    path = out_dir / "figure_accuracy_by_question_type.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    confusion = payload["confusion_matrix"]
    counts = np.asarray(confusion["matrix"], dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, rows, out=np.zeros_like(counts), where=rows != 0)
    labels = [str(label).replace("_", " ") for label in confusion["classes"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, label="Fraction of true class")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{scope_label}: row-normalized confusion")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            ax.text(
                column, row, f"{100 * value:.0f}%", ha="center", va="center",
                fontsize=7, color="white" if value > 0.52 else "black",
            )
    path = out_dir / "figure_confusion_matrix.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    strictness = payload["strictness"]
    if "numeric_tolerance_seconds" in strictness:
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
        axes[0].plot(strictness["threshold"], strictness["accuracy"], marker="o", color="#70AD47")
        axes[0].set_xlabel("Evidence IoU threshold")
        axes[0].set_ylabel("Fraction accepted")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Cited intervals")
        axes[1].plot(
            strictness["numeric_tolerance_seconds"], strictness["duration_accuracy"],
            marker="o", label="Duration", color="#4472C4",
        )
        axes[1].plot(
            strictness["numeric_tolerance_seconds"], strictness["onset_accuracy"],
            marker="s", label="Onset", color="#ED7D31",
        )
        axes[1].set_xlabel("Absolute error tolerance (s)")
        axes[1].set_ylabel("Fraction accepted")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Numeric answers")
        axes[1].legend(fontsize=7)
        fig.suptitle(f"{scope_label}: accuracy versus strictness", fontsize=11)
        ax = axes
    else:
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        ax.plot(strictness["threshold"], strictness["accuracy"], marker="o", color="#70AD47")
        ax.set_xlabel("Temporal IoU threshold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{scope_label}: accuracy versus strictness")
    path = out_dir / "figure_accuracy_vs_strictness.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    points = payload["overhead"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    max_latency = max(float(point["latency_ms"]) for point in points)
    min_latency = min(float(point["latency_ms"]) for point in points)
    for point in points:
        x, y = float(point["latency_ms"]), float(point["accuracy"])
        ax.scatter(x, y, s=55)
        size = point.get("serialized_bytes")
        annotation = str(point["model"])
        if size is not None:
            size_kib = float(size) / 1024
            annotation += (
                f"\n{size_kib:.0f} KiB" if size_kib < 1024
                else (
                    f"\n{size_kib / (1024 ** 2):.2f} GiB"
                    if size_kib >= 1024 ** 2
                    else f"\n{size_kib / 1024:.1f} MiB"
                )
            )
        rightmost = abs(x - max_latency) <= 1e-12
        y_offset = -30 if not rightmost and x > min_latency * 5 else -14
        ax.annotate(
            annotation,
            (x, y), xytext=(-5 if rightmost else 5, y_offset),
            textcoords="offset points", va="top",
            ha="right" if rightmost else "left", fontsize=8,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    metric = str(payload.get("overhead_metric", "recognition_validation_accuracy"))
    if metric == "overall_qa_macro_accuracy":
        ax.set_xlabel("Median QA-stage latency per query (ms, log scale)")
        ax.set_ylabel("Overall QA macro accuracy")
    else:
        ax.set_xlabel("Median validation latency per window (ms, log scale)")
        ax.set_ylabel("Recognition accuracy")
    ax.set_ylim(0, 1.05)
    ax.margins(x=0.12)
    ax.set_title(f"{scope_label}: accuracy versus overhead")
    ax.grid(alpha=0.2)
    ordered = sorted(
        ((float(point["latency_ms"]), float(point["accuracy"])) for point in points),
        key=lambda pair: pair[0],
    )
    frontier = []
    best_accuracy = -np.inf
    for candidate in ordered:
        if candidate[1] > best_accuracy + 1e-12:
            frontier.append(candidate)
            best_accuracy = candidate[1]
    if len(frontier) >= 2:
        ax.plot(
            [point[0] for point in frontier], [point[1] for point in frontier],
            linestyle="--", color="#404040", linewidth=1.1, label="Pareto frontier",
        )
        ax.legend(fontsize=7)
    elif frontier:
        ax.scatter(
            [frontier[0][0]], [frontier[0][1]], marker="*", s=150,
            facecolors="none", edgecolors="#404040", linewidths=1.0,
            label="Only non-dominated point",
        )
        ax.legend(fontsize=7)
    path = out_dir / "figure_accuracy_vs_overhead.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)

    robustness = payload["robustness"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(robustness["dropped_percent"], robustness["accuracy"], marker="o", color="#ED7D31")
    ax.set_xlabel("Dropped input samples (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{scope_label}: robustness curve")
    path = out_dir / "figure_robustness.png"
    _finish(fig, ax, path, dataset_kind); paths.append(path)
    return paths


def generate_from_json(results_path: Path, out_dir: Path) -> list[Path]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return generate_required_figures(payload, out_dir)

#!/usr/bin/env python3
"""Plot honest real-data figures for the scoped precomputed-feature baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="report/figures/real_precomputed")
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != "real" or "precomputed" not in payload.get("scope", ""):
        parser.error("input must be a real precomputed-feature baseline result")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    confusion = payload["confusion_matrix"]
    counts = np.asarray(confusion["matrix"], dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, rows, out=np.zeros_like(counts), where=rows != 0)
    labels = [str(name).replace("_", " ") for name in confusion["classes"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, label="Fraction of true class")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title("Real precomputed-feature baseline: row-normalized confusion")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.52 else "black",
            )
    confusion_path = out_dir / "figure_real_precomputed_confusion.png"
    finish(fig, confusion_path)

    candidates = payload["candidate_validation"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    max_latency = max(
        float(values["median_latency_ms_per_window"])
        for values in candidates.values()
    )
    for name, values in candidates.items():
        latency = float(values["median_latency_ms_per_window"])
        accuracy = float(values["accuracy"])
        size_kib = float(values["serialized_bytes"]) / 1024
        rightmost = latency == max_latency
        ax.scatter(latency, accuracy, s=65)
        ax.annotate(
            f"{name}\n{size_kib:.0f} KiB pickle",
            (latency, accuracy),
            xytext=(-6 if rightmost else 6, 5),
            textcoords="offset points",
            fontsize=8,
            ha="right" if rightmost else "left",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Median validation latency per example (ms, log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Real precomputed-feature baseline: accuracy versus overhead")
    ax.grid(alpha=0.2)
    overhead_path = out_dir / "figure_real_precomputed_overhead.png"
    finish(fig, overhead_path)

    print(confusion_path)
    print(overhead_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

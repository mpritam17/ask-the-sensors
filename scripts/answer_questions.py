#!/usr/bin/env python3
"""Answer sensor questions using a trained recognizer or transparent fallback."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ats.config import load_config, resolve  # noqa: E402
from ats.data.resample import resample_to_grid  # noqa: E402
from ats.data.windowing import make_windows  # noqa: E402
from ats.features import extract_features  # noqa: E402
from ats.qa import answer_question, route_question  # noqa: E402
from ats.recognition import load_bundle, predict_probabilities  # noqa: E402
from ats.timeline import build_timeline  # noqa: E402

INPUT_COLUMNS = ("timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")


def heuristic_probabilities(windows: np.ndarray, classes) -> np.ndarray:
    """Conservative no-model fallback for demos; never used as a reported model."""
    probabilities = np.full((len(windows), len(classes)), 0.02, dtype=np.float64)
    for i, window in enumerate(windows):
        acc = window[:, :3]
        gyro = window[:, 3:]
        acc_std = float(np.linalg.norm(acc.std(axis=0)))
        gyro_std = float(np.linalg.norm(gyro.std(axis=0)))
        mean = np.abs(acc.mean(axis=0))
        if acc_std > 0.55:
            label = "running"
        elif gyro_std > 0.30 and acc_std < 0.28:
            label = "bicycling"
        elif acc_std > 0.16:
            label = "walking"
        elif acc_std > 0.055 or gyro_std > 0.055:
            label = "standing_and_moving"
        elif acc_std > 0.022:
            label = "standing_in_place"
        elif mean[2] > mean[1]:
            label = "lying_down"
        else:
            label = "sitting"
        probabilities[i, classes.index(label)] = 0.88
        probabilities[i] /= probabilities[i].sum()
    return probabilities


def read_recording(path: Path):
    frame = pd.read_csv(path)
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"recording is missing columns: {', '.join(missing)}")
    numeric = frame.loc[:, INPUT_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 2:
        raise ValueError("recording must contain at least two finite samples")
    timestamps = numeric["timestamp"].to_numpy(dtype=float)
    timestamps -= timestamps[0]
    values = numeric.loc[:, INPUT_COLUMNS[1:]].to_numpy(dtype=float)
    return timestamps, values


def window_summaries(windows: np.ndarray):
    summaries = []
    for window in windows:
        summaries.append({
            "acc_rms_g": float(np.sqrt(np.mean(window[:, :3] ** 2))),
            "gyro_rms_rad_s": float(np.sqrt(np.mean(window[:, 3:] ** 2))),
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="artifacts/recognizer.joblib")
    parser.add_argument("--slm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--no-slm", action="store_true", help="use grounded deterministic Task 4 fallbacks only")
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    fs = float(data_cfg["sampling"]["target_hz"])
    timestamps, values = read_recording(Path(args.recording))
    duration = float(timestamps[-1] + 1.0 / fs)
    stream = resample_to_grid(
        timestamps,
        values,
        fs=fs,
        duration=duration,
        t0=0.0,
        max_gap=float(data_cfg["sampling"]["max_interp_gap_seconds"]),
    )
    win_cfg = data_cfg["windowing"]
    windows = make_windows(
        stream.values,
        stream.valid,
        t0=0.0,
        fs=fs,
        window_samples=int(win_cfg["window_samples"]),
        hop_samples=int(win_cfg["hop_samples"]),
        min_validity=float(win_cfg["min_window_validity"]),
    )
    if len(windows) == 0:
        raise ValueError("recording is shorter than one analysis window")

    classes = list(load_config("labels")["classes"])
    model_path = resolve(args.model)
    if model_path.exists():
        bundle = load_bundle(model_path)
        classes = list(bundle["classes"])
        probabilities = predict_probabilities(bundle, extract_features(windows.values, fs=fs))
    else:
        print(
            f"WARNING: {model_path} not found; using untrained demo heuristic. "
            "Do not report its output as model accuracy.",
            file=sys.stderr,
        )
        probabilities = heuristic_probabilities(windows.values, classes)

    timeline_cfg = model_cfg["timeline"]
    timeline = build_timeline(
        probabilities,
        windows.t_start,
        windows.t_end,
        classes,
        valid=windows.valid,
        feature_summaries=window_summaries(windows.values),
        confidence_threshold=float(model_cfg["recognition"]["confidence_threshold"]),
        smoothing_windows=int(timeline_cfg["smoothing_windows"]),
        max_gap_seconds=float(timeline_cfg["max_gap_seconds"]),
        minimum_interval_seconds=float(timeline_cfg["minimum_interval_seconds"]),
    )

    questions = [line.strip() for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    slm = None
    answers = []
    for question in questions:
        intent, _ = route_question(question)
        if intent == "open_world" and not args.no_slm:
            try:
                if slm is None:
                    from ats.slm import GroundedQwen

                    slm = GroundedQwen(args.slm_model)
                answers.append(slm.answer(question, timeline))
                continue
            except Exception as exc:  # validated deterministic fallback is required
                print(f"WARNING: Task 4 SLM unavailable or rejected ({exc}); using grounded fallback.", file=sys.stderr)
        answers.append(answer_question(question, timeline))
    output = "\n\n".join(answer.format() for answer in answers)
    Path(args.out).write_text(output + ("\n" if output else ""), encoding="utf-8")
    print(f"wrote {len(questions)} answer(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

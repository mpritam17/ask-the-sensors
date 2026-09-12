"""Engineered time/frequency features with explicit sensor provenance."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

CHANNELS: Tuple[str, ...] = (
    "Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"
)
PER_CHANNEL_FEATURES: Tuple[str, ...] = (
    "mean", "std", "median", "iqr", "min", "max", "rms", "energy",
    "zero_crossing_rate", "dominant_frequency", "spectral_entropy",
    "band_power_0p3_3", "band_power_3_8", "band_power_8_12",
)
CORRELATION_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)
)


def feature_names(channels: Sequence[str] = CHANNELS) -> List[str]:
    names = [f"{channel}|{feature}" for channel in channels for feature in PER_CHANNEL_FEATURES]
    names.extend(f"corr|{channels[a]}|{channels[b]}" for a, b in CORRELATION_PAIRS)
    for modality in ("accelerometer", "gyroscope"):
        names.extend(
            f"{modality}_magnitude|{name}"
            for name in ("mean", "std", "rms", "max")
        )
    return names


def feature_provenance(names: Sequence[str] | None = None) -> List[Dict[str, object]]:
    """Map every feature to the modality/channels it was computed from."""
    result: List[Dict[str, object]] = []
    for name in names or feature_names():
        parts = name.split("|")
        if parts[0].startswith("Acc"):
            result.append({"feature": name, "modality": "accelerometer", "channels": [parts[0]]})
        elif parts[0].startswith("Gyro"):
            result.append({"feature": name, "modality": "gyroscope", "channels": [parts[0]]})
        elif parts[0] == "corr":
            modality = "accelerometer" if parts[1].startswith("Acc") else "gyroscope"
            result.append({"feature": name, "modality": modality, "channels": parts[1:3]})
        else:
            modality = parts[0].removesuffix("_magnitude")
            prefix = "Acc" if modality == "accelerometer" else "Gyro"
            result.append({"feature": name, "modality": modality, "channels": [f"{prefix} X/Y/Z"]})
    return result


def feature_indices_for_modalities(
    modalities: Sequence[str], names: Sequence[str] | None = None
) -> np.ndarray:
    """Indices of engineered features supported by the available sensors."""
    allowed = set(modalities)
    unknown = allowed - {"accelerometer", "gyroscope"}
    if unknown:
        raise ValueError(f"unsupported modalities: {sorted(unknown)}")
    schema = list(names or feature_names())
    provenance = feature_provenance(schema)
    return np.asarray(
        [index for index, item in enumerate(provenance) if item["modality"] in allowed],
        dtype=int,
    )


def select_engineered_features(
    features: np.ndarray, schema: Sequence[str]
) -> np.ndarray:
    """Select a bundle's engineered schema from the canonical 98 features."""
    canonical = feature_names()
    index = {name: position for position, name in enumerate(canonical)}
    missing = [name for name in schema if name not in index]
    if missing:
        raise ValueError(f"non-engineered feature(s) in bundle schema: {missing[:3]}")
    return np.asarray(features)[:, [index[name] for name in schema]]


def _spectral_features(values: np.ndarray, fs: float) -> np.ndarray:
    """Return dominant frequency, entropy, and three band powers per window."""
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / fs)
    power[:, 0] = 0.0
    total = power.sum(axis=1, keepdims=True)
    normalized = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    entropy = -np.sum(normalized * np.log2(normalized + 1e-12), axis=1)
    entropy /= np.log2(max(power.shape[1] - 1, 2))
    dominant = frequencies[np.argmax(power, axis=1)]

    bands = []
    for low, high in ((0.3, 3.0), (3.0, 8.0), (8.0, 12.0)):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(power[:, mask].sum(axis=1) / max(values.shape[1], 1))
    return np.column_stack([dominant, entropy, *bands])


def extract_features(windows: np.ndarray, fs: float = 25.0) -> np.ndarray:
    """Extract a stable ``(n_windows, 98)`` float32 feature matrix."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3 or windows.shape[2] != len(CHANNELS):
        raise ValueError(f"expected (n, samples, 6) windows, got {windows.shape}")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if windows.shape[0] == 0:
        return np.empty((0, len(feature_names())), dtype=np.float32)

    blocks = []
    for channel in range(windows.shape[2]):
        values = windows[:, :, channel]
        centered = values - values.mean(axis=1, keepdims=True)
        q75, q25 = np.percentile(values, [75, 25], axis=1)
        zero_crossings = np.mean(centered[:, 1:] * centered[:, :-1] < 0, axis=1)
        time_features = np.column_stack([
            values.mean(axis=1),
            values.std(axis=1),
            np.median(values, axis=1),
            q75 - q25,
            values.min(axis=1),
            values.max(axis=1),
            np.sqrt(np.mean(values ** 2, axis=1)),
            np.mean(values ** 2, axis=1),
            zero_crossings,
        ])
        blocks.append(np.column_stack([time_features, _spectral_features(values, fs)]))

    correlations = []
    for a, b in CORRELATION_PAIRS:
        av = windows[:, :, a] - windows[:, :, a].mean(axis=1, keepdims=True)
        bv = windows[:, :, b] - windows[:, :, b].mean(axis=1, keepdims=True)
        numerator = np.sum(av * bv, axis=1)
        denominator = np.sqrt(np.sum(av * av, axis=1) * np.sum(bv * bv, axis=1))
        correlations.append(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12))

    magnitude_features = []
    for start in (0, 3):
        magnitude = np.linalg.norm(windows[:, :, start:start + 3], axis=2)
        magnitude_features.extend([
            magnitude.mean(axis=1),
            magnitude.std(axis=1),
            np.sqrt(np.mean(magnitude ** 2, axis=1)),
            magnitude.max(axis=1),
        ])

    result = np.column_stack([*blocks, *correlations, *magnitude_features])
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape[1] != len(feature_names()):
        raise AssertionError("feature schema and extractor are out of sync")
    return result.astype(np.float32)

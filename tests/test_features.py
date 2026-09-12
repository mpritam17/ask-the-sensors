import numpy as np

from ats.features import (
    extract_features,
    feature_indices_for_modalities,
    feature_names,
    feature_provenance,
    select_engineered_features,
)


def test_feature_schema_is_finite_and_provenanced():
    rng = np.random.default_rng(2)
    windows = rng.normal(size=(4, 64, 6)).astype(np.float32)
    matrix = extract_features(windows)
    names = feature_names()
    provenance = feature_provenance(names)
    assert matrix.shape == (4, len(names))
    assert len(names) == 98
    assert np.isfinite(matrix).all()
    assert all(item["modality"] in {"accelerometer", "gyroscope"} for item in provenance)


def test_accelerometer_feature_subset_matches_provenance():
    rng = np.random.default_rng(3)
    all_features = extract_features(rng.normal(size=(2, 64, 6)))
    indices = feature_indices_for_modalities(["accelerometer"])
    schema = [feature_names()[index] for index in indices]
    selected = select_engineered_features(all_features, schema)
    assert selected.shape == (2, 49)
    assert np.allclose(selected, all_features[:, indices])
    assert {p["modality"] for p in feature_provenance(schema)} == {"accelerometer"}


def test_feature_extractor_accepts_empty_batch():
    matrix = extract_features(np.empty((0, 64, 6), dtype=np.float32))
    assert matrix.shape == (0, 98)

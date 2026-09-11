import numpy as np

from ats.recognition.model import predict_probabilities, train_and_select


def test_recognizer_bundle_preserves_class_order():
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(-2, 0.2, (20, 6)), rng.normal(2, 0.2, (20, 6))])
    y = np.repeat([0, 1], 20)
    config = {
        "random_seed": 7,
        "selection_tolerance_macro_f1": 0.01,
        "candidates": {
            "logistic": {"max_iter": 100},
            "compact_rf": {"n_estimators": 8, "max_depth": 4},
            "full_rf": {"n_estimators": 12, "max_depth": 5},
        },
    }
    # The bundle schema is fixed at 98 features; pad this compact separability
    # example so the serialization/load contract stays realistic.
    x = np.pad(x, ((0, 0), (0, 92)))
    train_idx = np.r_[0:15, 20:35]
    validation_idx = np.r_[15:20, 35:40]
    bundle, results = train_and_select(
        x[train_idx], y[train_idx], x[validation_idx], y[validation_idx], classes=["a", "b"],
        config=config, dataset_kind="synthetic"
    )
    probabilities = predict_probabilities(bundle, x[validation_idx])
    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(results) == {"logistic", "compact_rf", "full_rf"}

import numpy as np
import pandas as pd

from ats.data.precomputed import load_precomputed_user, sensor_feature_columns


def test_sensor_feature_selection_and_original_label_join(tmp_path):
    features = pd.DataFrame({
        "timestamp": [10, 20],
        "raw_acc:magnitude_stats:mean": [1.0, 2.0],
        "proc_gyro:3d:mean_x": [0.1, np.nan],
        "watch_acc:mean": [9.0, 9.0],
        "label:FIX_walking": [1.0, 0.0],
    })
    original = pd.DataFrame({
        "timestamp": [10, 20],
        "original_label:WALKING": [1, 0],
        "original_label:SITTING": [0, 1],
    })
    feature_path, label_path = tmp_path / "features.csv", tmp_path / "original.csv"
    features.to_csv(feature_path, index=False)
    original.to_csv(label_path, index=False)
    columns = sensor_feature_columns(features.columns)
    assert columns == ["raw_acc:magnitude_stats:mean", "proc_gyro:3d:mean_x"]
    x, y, names = load_precomputed_user(feature_path, label_path)
    # Walking is retained; sitting is not contradicted because its cleaned
    # consistency column is absent rather than treated as zero.
    assert x.shape == (2, 2)
    assert names == columns
    assert y.tolist() == [4, 1]

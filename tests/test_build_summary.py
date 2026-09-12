import json

from scripts.summarize_build_manifest import summarize


def test_build_manifest_summary_is_aggregated_and_hashed(tmp_path):
    path = tmp_path / "build_manifest.json"
    path.write_text(json.dumps([
        {
            "uuid": "a", "modalities": "acc", "unit_rescaled": False,
            "n_examples_labelled": 3, "n_examples_seen": 2,
            "n_examples_used": 2, "n_windows_valid": 10,
            "class_counts": {"walking": 10},
        },
        {
            "uuid": "b", "modalities": "acc", "unit_rescaled": True,
            "n_examples_labelled": 3, "n_examples_seen": 3,
            "n_examples_used": 3, "n_windows_valid": 12,
            "class_counts": {"walking": 4, "running": 8},
        },
    ]), encoding="utf-8")
    payload = summarize(path)
    assert payload["users"] == 2
    assert payload["modalities"] == ["acc"]
    assert payload["users_with_accelerometer_unit_rescaling"] == 1
    assert payload["n_examples_used"] == 5
    assert payload["n_labelled_without_indexed_example"] == 1
    assert payload["n_windows_valid"] == 22
    assert payload["class_counts_valid_windows"] == {"walking": 14, "running": 8}
    assert len(payload["source_manifest_sha256"]) == 64

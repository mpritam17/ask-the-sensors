import json

import pytest

from ats.data.splits import load_official_fold_map, make_user_splits, write_split_manifest


def test_user_splits_are_complete_disjoint_and_repeatable(tmp_path):
    users = [f"user-{i:02d}" for i in range(20)]
    first = make_user_splits(users, seed=11, n_qa_users=5)
    second = make_user_splits(reversed(users), seed=11, n_qa_users=5)
    assert first == second
    assert len(first["qa"]) == 5
    all_members = [u for values in first.values() for u in values]
    assert len(all_members) == len(set(all_members)) == len(users)

    path = write_split_manifest(first, tmp_path / "splits.json", seed=11, source="test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["strategy"] == "by_user"
    assert len(payload["sha256"]) == 64


def test_user_splits_reject_too_few_users():
    with pytest.raises(ValueError):
        make_user_splits(["a", "b", "c"])


def test_official_test_fold_is_preserved_and_qa_is_disjoint(tmp_path):
    fold_dir = tmp_path / "cv_5_folds"
    fold_dir.mkdir()
    (fold_dir / "fold_0_test_android_uuids.txt").write_text("user-00\nuser-01\n")
    (fold_dir / "fold_0_test_iphone_uuids.txt").write_text("user-02\n")
    (fold_dir / "fold_1_test_android_uuids.txt").write_text(
        "\n".join(f"user-{index:02d}" for index in range(3, 20)) + "\n"
    )
    mapping = load_official_fold_map(tmp_path)
    splits = make_user_splits(
        [f"user-{index:02d}" for index in range(20)],
        seed=11,
        n_qa_users=5,
        official_folds=mapping,
        test_fold=0,
    )
    assert splits["recognition_test"] == ["user-00", "user-01", "user-02"]
    assert len(splits["qa"]) == 5
    assert not (set(splits["recognition_test"]) & set(splits["qa"]))

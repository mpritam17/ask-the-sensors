import json

import pytest

from ats.data.splits import make_user_splits, write_split_manifest


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

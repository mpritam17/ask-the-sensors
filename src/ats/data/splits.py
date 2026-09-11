"""Deterministic user-disjoint dataset partitions.

Every window from one user belongs to exactly one of train, validation,
recognition-test, or end-to-end QA.  This prevents adjacent windows and
user-specific sensor placement from leaking between evaluation partitions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

SPLIT_NAMES = ("train", "validation", "recognition_test", "qa")


def make_user_splits(
    uuids: Iterable[str],
    *,
    seed: int = 20255,
    n_qa_users: int = 5,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    official_folds: Optional[Mapping[str, int]] = None,
    test_fold: int = 0,
) -> Dict[str, List[str]]:
    """Return mutually exclusive, deterministic user partitions.

    When ``official_folds`` is supplied, its selected fold seeds the
    recognition-test partition. Remaining allocations are seeded shuffles.
    Small smoke-test datasets degrade gracefully while keeping train nonempty.
    """
    users = sorted({str(u) for u in uuids})
    if len(users) < 4:
        raise ValueError("at least four users are required for four disjoint splits")
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(users)[rng.permutation(len(users))])

    qa_count = min(max(int(n_qa_users), 1), len(users) - 3)
    qa = [str(u) for u in shuffled[:qa_count]]
    remaining = [u for u in shuffled if u not in set(qa)]

    if official_folds:
        recognition_test = sorted(
            u for u in remaining if official_folds.get(u) == int(test_fold)
        )
    else:
        recognition_test = []
    if not recognition_test:
        test_count = min(max(1, round(len(remaining) * test_fraction)), len(remaining) - 2)
        recognition_test = remaining[:test_count]

    remaining = [u for u in remaining if u not in set(recognition_test)]
    val_count = min(max(1, round(len(remaining) * validation_fraction)), len(remaining) - 1)
    validation = remaining[:val_count]
    train = remaining[val_count:]

    result = {
        "train": sorted(train),
        "validation": sorted(validation),
        "recognition_test": sorted(recognition_test),
        "qa": sorted(qa),
    }
    assert set().union(*(set(v) for v in result.values())) == set(users)
    assert sum(len(v) for v in result.values()) == len(users)
    return result


def write_split_manifest(
    splits: Mapping[str, Iterable[str]], out_path: Path, *, seed: int, source: str
) -> Path:
    normalized = {name: sorted(map(str, splits.get(name, []))) for name in SPLIT_NAMES}
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    payload = {
        "strategy": "by_user",
        "source": source,
        "seed": int(seed),
        "splits": normalized,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path

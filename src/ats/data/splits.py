"""Deterministic user-disjoint dataset partitions.

Every window from one user belongs to exactly one of train, validation,
recognition-test, or end-to-end QA.  This prevents adjacent windows and
user-specific sensor placement from leaking between evaluation partitions.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

SPLIT_NAMES = ("train", "validation", "recognition_test", "qa")


def load_official_fold_map(root: Path) -> Dict[str, int]:
    """Read the official fold test lists as ``{uuid: fold_number}``."""
    root = Path(root)
    fold_dir = root / "cv_5_folds" if (root / "cv_5_folds").is_dir() else root
    mapping: Dict[str, int] = {}
    paths = sorted(fold_dir.glob("fold_*_test_*_uuids.txt"))
    if not paths:
        raise ValueError(f"no official test-fold UUID lists found under {root}")
    for path in paths:
        match = re.match(r"fold_(\d+)_test_", path.name)
        if not match:
            continue
        fold = int(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines():
            uuid = line.strip()
            if not uuid:
                continue
            previous = mapping.get(uuid)
            if previous is not None and previous != fold:
                raise ValueError(f"user {uuid} occurs in test folds {previous} and {fold}")
            mapping[uuid] = fold
    if not mapping:
        raise ValueError(f"official fold lists under {root} contain no UUIDs")
    return mapping


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

    if official_folds:
        recognition_test = sorted(
            u for u in users if official_folds.get(u) == int(test_fold)
        )
        if recognition_test:
            remaining = [u for u in shuffled if u not in set(recognition_test)]
            if len(remaining) < 3:
                raise ValueError("official test fold leaves too few users for train/validation/QA")
            qa_count = min(max(int(n_qa_users), 1), len(remaining) - 2)
            qa = [str(u) for u in remaining[:qa_count]]
            remaining = [u for u in remaining if u not in set(qa)]
        else:
            qa = []
            remaining = shuffled
    else:
        qa_count = min(max(int(n_qa_users), 1), len(users) - 3)
        qa = [str(u) for u in shuffled[:qa_count]]
        remaining = [u for u in shuffled if u not in set(qa)]
        recognition_test = []
    if not recognition_test:
        if not qa:
            qa_count = min(max(int(n_qa_users), 1), len(users) - 3)
            qa = [str(u) for u in remaining[:qa_count]]
            remaining = [u for u in remaining if u not in set(qa)]
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

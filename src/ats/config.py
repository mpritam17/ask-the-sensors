"""Configuration loading.

All tunable parameters live in configs/*.yaml so that (a) no magic number is
buried in code, and (b) the report can cite a single source of truth for every
design decision.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root = two levels above this file (src/ats/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class Config(dict):
    """dict with attribute access and dotted lookup, e.g. cfg.get_path('a.b.c')."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node


def load_config(name: str) -> Config:
    """Load configs/<name>.yaml (the .yaml suffix is optional)."""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle) or {})


def load_all() -> Dict[str, Config]:
    """Load every config in configs/, keyed by stem."""
    return {p.stem: load_config(p.name) for p in sorted(CONFIG_DIR.glob("*.y*ml"))}


def resolve(path_like: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(os.path.expanduser(str(path_like)))
    return p if p.is_absolute() else (REPO_ROOT / p)

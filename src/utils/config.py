"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(*paths: str) -> dict:
    cfg: dict[str, Any] = {}
    root = project_root()
    for p in paths:
        path = root / p if not Path(p).is_absolute() else Path(p)
        with open(path, "r", encoding="utf-8") as f:
            merged = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, merged)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

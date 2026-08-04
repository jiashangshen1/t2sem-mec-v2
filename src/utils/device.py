"""Device selection."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_device(requested: str = "auto", *, config_default: str = "cpu") -> str:
    """Pick the torch device string.

    For this project, ``auto`` follows ``config_default`` (typically ``cpu``):
    env stepping dominates runtime and the MLPs are small, so GPU transfer
    overhead often exceeds any compute savings. Pass ``--device cuda`` to override.
    """
    if requested == "auto":
        return config_default
    return requested


def device_from_config(requested: str, cfg: Mapping[str, Any]) -> str:
    """Resolve device using ``drl.device`` from a loaded config when ``auto``."""
    default = str(cfg.get("drl", {}).get("device", "cpu"))
    return resolve_device(requested, config_default=default)

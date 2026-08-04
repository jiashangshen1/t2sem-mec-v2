"""SLA tier definitions and penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SLATier:
    name: str
    t_max: float
    s_min: float
    weight: float


def load_sla_tiers(cfg: dict) -> List[SLATier]:
    tiers = []
    for name, spec in cfg.items():
        tiers.append(
            SLATier(
                name=name,
                t_max=float(spec["t_max"]),
                s_min=float(spec["s_min"]),
                weight=float(spec.get("weight", 1.0)),
            )
        )
    return tiers


def sample_tier(
    tiers: List[SLATier], rng: np.random.Generator, weights: np.ndarray | None = None
) -> SLATier:
    if weights is None:
        w = np.array([t.weight for t in tiers], dtype=np.float64)
    else:
        w = weights.astype(np.float64)
    w = np.clip(w, 1e-6, None)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        w = np.ones(len(tiers), dtype=np.float64) / len(tiers)
    else:
        w = w / total
    idx = int(rng.choice(len(tiers), p=w))
    return tiers[idx]


def sla_satisfied(delay: float, sim: float, tier: SLATier) -> bool:
    return delay <= tier.t_max and sim >= tier.s_min


def penalty(delay_violation: float, sem_violation: float, rho_d: float = 10.0, rho_s: float = 10.0) -> float:
    return rho_d * delay_violation + rho_s * sem_violation

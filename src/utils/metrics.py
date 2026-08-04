"""Episode metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EpisodeMetrics:
    tasks_total: int = 0
    tasks_sla_met: int = 0
    sla_rate: float = 0.0
    avg_delay: float = 0.0
    total_energy: float = 0.0
    avg_reward: float = 0.0
    offload_local: int = 0
    offload_uav: int = 0
    offload_hap: int = 0


def aggregate_metrics(rows: List[EpisodeMetrics]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    return {
        "sla_rate_mean": sum(r.sla_rate for r in rows) / n,
        "sla_rate_std": float(
            (sum((r.sla_rate - sum(x.sla_rate for x in rows) / n) ** 2 for r in rows) / max(n - 1, 1)) ** 0.5
        ),
        "avg_delay_mean": sum(r.avg_delay for r in rows) / n,
        "total_energy_mean": sum(r.total_energy for r in rows) / n,
        "avg_reward_mean": sum(r.avg_reward for r in rows) / n,
    }

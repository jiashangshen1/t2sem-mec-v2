"""Greedy baseline: nearest UAV/HAP with moderate semantic compression."""

from __future__ import annotations

import numpy as np

from ..env.channel import distance_3d
from ..env.entities import TARGET_HAP, TARGET_LOCAL, TARGET_UAV
from ..env.multi_mec_env import MultiMECEnv


class GreedyPolicy:
    def __init__(self, env: MultiMECEnv, mu: float = 0.6):
        self.env = env
        self.mu = mu

    def act(self, obs: np.ndarray) -> np.ndarray:
        env = self.env
        action = np.zeros(env.short_action_dim, dtype=np.float32)
        mu_norm = (self.mu - env.mu_min) / max(env.mu_max - env.mu_min, 1e-6)
        mu_a = mu_norm * 2.0 - 1.0

        for i, term in enumerate(env.terminals):
            base = i * env.action_dim_per_terminal
            if not term.queue:
                action[base : base + 4] = np.array([-1.0, mu_a, 0.0, 0.0], dtype=np.float32)
                continue

            tier = term.queue[0].tier
            tier_obj = env.tier_map[tier]
            dists_uav = [distance_3d(term.position, u.position) for u in env.uavs]
            dists_hap = [distance_3d(term.position, h.position) for h in env.haps]

            if tier_obj.t_max <= 0.15 and env.uavs:
                target_idx = 1 + int(np.argmin(dists_uav))
            elif env.haps and tier_obj.t_max > 0.4:
                target_idx = 1 + env.num_uavs + int(np.argmin(dists_hap))
            elif env.uavs:
                target_idx = 1 + int(np.argmin(dists_uav))
            else:
                target_idx = 0

            n = env.num_offload_targets
            offload_raw = (target_idx + 0.5) / n
            offload_a = offload_raw * 2.0 - 1.0
            action[base : base + 4] = np.array([offload_a, mu_a, 0.5, 0.5], dtype=np.float32)
        return action

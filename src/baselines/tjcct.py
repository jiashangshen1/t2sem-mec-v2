"""Lite-TJCCT: K-means UAV centroid tracking + priority water-filling short scale."""

from __future__ import annotations

import numpy as np

from ..env.channel import distance_3d
from ..env.entities import TARGET_HAP, TARGET_LOCAL, TARGET_UAV
from ..env.multi_mec_env import MultiMECEnv

_TIER_PRIO = {"critical": 3.0, "interactive": 2.0, "best_effort": 1.0}


class TJCCTHeuristicPolicy:
    def __init__(self, env: MultiMECEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._update_uav_positions()
        return self._short_waterfill()

    def _short_waterfill(self) -> np.ndarray:
        env = self.env
        action = np.zeros(env.short_action_dim, dtype=np.float32)
        active: list[tuple[int, float]] = []
        for i, term in enumerate(env.terminals):
            if not term.queue:
                continue
            tier = term.queue[0].tier
            prio = _TIER_PRIO.get(tier, 1.0)
            active.append((i, prio * (1.0 + len(term.queue))))

        if not active:
            return action

        total_w = sum(w for _, w in active)
        shares = {i: w / total_w for i, w in active}

        for i, term in enumerate(env.terminals):
            base = i * env.action_dim_per_terminal
            if i not in shares:
                action[base : base + 4] = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                continue

            tier_obj = env.tier_map[term.queue[0].tier]
            mu = env._min_mu_by_tier.get(tier_obj.name, 0.6)
            mu_norm = (mu - env.mu_min) / max(env.mu_max - env.mu_min, 1e-6)
            mu_a = mu_norm * 2.0 - 1.0
            share = shares[i]
            res_a = float(np.clip(share * 2.0, 0.2, 1.0) * 2.0 - 1.0)

            dists_uav = [distance_3d(term.position, u.position) for u in env.uavs]
            dists_hap = [
                distance_3d(term.position, h.position)
                for k, h in enumerate(env.haps)
                if env._hap_is_available(k)
            ]
            hap_ids = [k for k in range(env.num_haps) if env._hap_is_available(k)]

            if tier_obj.t_max <= 0.15 and env.uavs:
                target_idx = 1 + int(np.argmin(dists_uav))
            elif hap_ids and tier_obj.t_max > 0.35:
                local_hap = int(np.argmin(dists_hap))
                target_idx = 1 + env.num_uavs + hap_ids[local_hap]
            elif env.uavs:
                target_idx = 1 + int(np.argmin(dists_uav))
            else:
                target_idx = 0

            n = env.num_offload_targets
            offload_a = ((target_idx + 0.5) / n) * 2.0 - 1.0
            action[base : base + 4] = np.array([offload_a, mu_a, res_a, res_a], dtype=np.float32)
        return action

    def _update_uav_positions(self) -> None:
        env = self.env
        if not env.uavs or not env.terminals:
            return
        active = [t for t in env.terminals if t.queue]
        if not active:
            return
        positions = np.array([t.position[:2] for t in active])
        if len(env.uavs) == 1:
            centroids = [positions.mean(axis=0)]
        else:
            centroids = self._kmeans(positions, len(env.uavs))

        speed_scale = 0.65
        for uav, centroid in zip(env.uavs, centroids):
            direction = np.zeros(3)
            direction[:2] = centroid - uav.position[:2]
            norm = np.linalg.norm(direction[:2])
            if norm > 1e-3:
                step = min(norm, env.uav_max_speed * env.slot_duration * speed_scale)
                direction[:2] = direction[:2] / norm * step
                uav.position = uav.position + direction
                uav.position[:2] = np.clip(uav.position[:2], 0, env.area_size)
                uav.velocity = direction / max(env.slot_duration, 1e-6)

    @staticmethod
    def _kmeans(points: np.ndarray, k: int, iters: int = 8) -> list:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=min(k, len(points)), replace=False)
        centers = points[idx].astype(np.float64)
        for _ in range(iters):
            dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            for j in range(len(centers)):
                mask = labels == j
                if mask.any():
                    centers[j] = points[mask].mean(axis=0)
        return [c for c in centers]

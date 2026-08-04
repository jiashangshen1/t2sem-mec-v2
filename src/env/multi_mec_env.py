"""Gymnasium environment: multi-UAV, multi-user, multi-terminal semantic MEC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .channel import (
    channel_gain,
    computation_delay,
    distance_3d,
    los_probability,
    path_loss_db,
    shannon_rate,
    transmission_delay,
)
from .entities import (
    HAPState,
    StepResult,
    TARGET_HAP,
    TARGET_LOCAL,
    TARGET_UAV,
    Task,
    TerminalState,
    UAVState,
    UserState,
)
from .semantic import (
    clip_mu,
    remote_cpu_cycles,
    semantic_extraction_delay,
    similarity,
    transmission_bits,
)
from .sla import SLATier, load_sla_tiers, penalty, sample_tier, sla_satisfied


def load_yaml_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MultiMECEnv(gym.Env):
    """Two-timescale hierarchical MEC with M UAVs, K HAPs, N users, T terminals."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        *,
        disable_hap: bool = False,
        fixed_mu: Optional[float] = None,
        sla_penalty_enabled: bool = True,
        two_timescale: bool = True,
        resource_contention: Optional[bool] = None,
    ):
        super().__init__()
        self.cfg = config
        self.disable_hap = disable_hap
        self.fixed_mu = fixed_mu
        self.sla_penalty_enabled = sla_penalty_enabled
        self.two_timescale = two_timescale
        self.resource_contention = (
            bool(config.get("resource_contention", True))
            if resource_contention is None
            else resource_contention
        )

        self.num_users = int(config["num_users"])
        self.terminals_per_user = int(config["terminals_per_user"])
        self.num_terminals = self.num_users * self.terminals_per_user
        self.num_uavs = int(config["num_uavs"])
        self.num_haps = 0 if disable_hap else int(config["num_haps"])
        self.slots_per_frame = int(config["slots_per_frame"])
        self.slot_duration = float(config.get("slot_duration", 1.0))
        self.area_size = float(config["area_size"])
        self.seed_val = int(config.get("seed", 42))

        self.sla_tiers = load_sla_tiers(config["sla_tiers"])
        self.tier_map: Dict[str, SLATier] = {t.name: t for t in self.sla_tiers}
        self.num_tiers = len(self.sla_tiers)

        self.terminal_type_cfg = config["terminal_types"]
        self.terminal_assignment = list(config["terminal_assignment"])
        self.device_type_names = list(self.terminal_type_cfg.keys())
        self.num_device_types = len(self.device_type_names)

        sem = config["semantic"]
        self.mu_min = float(sem["mu_min"])
        self.mu_max = float(sem["mu_max"])
        self.c_sem = float(sem["c_sem"])
        self.s_max = float(sem["s_max"])
        self.lambda_acc = float(sem["lambda_acc"])

        ch = config["channel"]
        self.bandwidth = float(ch["bandwidth"])
        self.noise_power = float(ch["noise_power"])
        self.carrier_freq = float(ch["carrier_freq"])
        self.los_a = float(ch["los_prob_a"])
        self.los_b = float(ch["los_prob_b"])

        rw = config["reward"]
        self.w_throughput = float(rw["w_throughput"])
        self.w_energy = float(rw["w_energy"])
        self.w_sla = float(rw["w_sla"]) if sla_penalty_enabled else 0.0
        self.w_queue = float(rw["w_queue"])
        self.w_traj_energy = float(rw["w_traj_energy"])
        self.w_fairness = float(rw.get("w_fairness", 0.0))
        # Configurable SLA penalty scale (was hardcoded rho=10 in sla.penalty).
        self.rho_delay = float(rw.get("rho_delay", 10.0))
        self.rho_sem = float(rw.get("rho_sem", 10.0))
        # Optional bound so large w_sla does not produce an exploding penalty
        # cliff that drowns out the throughput/shaping signal (0 = disabled).
        self.penalty_clip = float(rw.get("penalty_clip", 0.0))
        # Optional bound on the per-task energy reward contribution so a single
        # high-energy offload cannot dominate the gradient (0 = disabled).
        self.energy_reward_clip = float(rw.get("energy_reward_clip", 0.0))
        # Dense potential-based shaping toward SLA feasibility (off by default).
        self.shaping_enabled = bool(rw.get("shaping_enabled", False))
        self.w_shaping = float(rw.get("w_shaping", 0.0))
        # Lagrangian constrained-RL: maximize SLA quality scaled by an adaptive
        # multiplier while always paying the energy cost, so that once the SLA
        # target is met the policy is free to cut energy. lambda is updated
        # externally (dual ascent) via set_sla_lambda(); default behaviour
        # (disabled) is unchanged.
        self.lagrangian_enabled = bool(rw.get("lagrangian_enabled", False))
        self.sla_lambda = float(rw.get("sla_lambda_init", 1.0))
        # Per-tier Lagrange multipliers (multi-constraint CMDP): one dual
        # variable per SLA tier so each QoS class is guaranteed independently.
        self.sla_lambda_by_tier: Dict[str, float] = {
            t.name: float(rw.get("sla_lambda_init", 1.0)) for t in self.sla_tiers
        }
        # Model-informed feasibility region: clamp mu up to meet similarity and
        # restrict the offload target to those that can meet the deadline.
        self.feasibility_mask_enabled = bool(rw.get("feasibility_mask_enabled", False))
        # Safety margins make the mask robust to task-size variance (+-20%) and
        # stochastic NLoS, pushing the *realized* SLA toward the feasible ceiling.
        self.mask_delay_margin = float(rw.get("mask_delay_margin", 1.0))  # alpha: est<=alpha*t_max
        by_tier = rw.get("mask_delay_margin_by_tier") or {}
        self.mask_delay_margin_by_tier: Dict[str, float] = {
            t.name: float(by_tier.get(t.name, self.mask_delay_margin)) for t in self.sla_tiers
        }
        self.mask_sim_margin = float(rw.get("mask_sim_margin", 0.0))      # extra similarity headroom
        self.mask_channel = str(rw.get("mask_channel", "los"))            # los|expected|nlos
        self._min_mu_by_tier: Dict[str, float] = {
            t.name: self._min_mu_for_sim(t.s_min) for t in self.sla_tiers
        }

        user_cfg = config["user"]
        self.user_cpu = float(user_cfg["max_cpu"])
        self.user_power = float(user_cfg["max_power"])
        self.user_mobility = float(user_cfg["mobility_speed"])
        self.hotspot_drift = bool(user_cfg.get("hotspot_drift", False))
        self.hotspot_speed = float(user_cfg.get("hotspot_speed", self.user_mobility))

        hap_cfg = config.get("hap", {})
        self.hap_availability = float(hap_cfg.get("availability", 1.0))
        self._hap_available: List[bool] = []

        task_cfg = config["task"]
        self.task_data = float(task_cfg["data_bits"])
        self.task_cycles = float(task_cfg["cpu_cycles"])
        self.arrival_rate = float(task_cfg["arrival_rate"])

        uav_cfg = config["uav"]
        self.uav_inits = [np.array(p, dtype=np.float64) for p in uav_cfg["init_positions"][: self.num_uavs]]
        while len(self.uav_inits) < self.num_uavs:
            self.uav_inits.append(np.array([250.0, 250.0, 100.0]))
        self.uav_max_speed = float(uav_cfg["max_speed"])
        self.uav_cpu = float(uav_cfg["max_cpu"])
        self.uav_bw = float(uav_cfg["max_bandwidth"])
        self.uav_prop_coeff = float(uav_cfg["energy_prop_coeff"])
        self.uav_alt_min = float(uav_cfg.get("altitude_min", 50.0))
        self.uav_alt_max = float(uav_cfg.get("altitude_max", 300.0))

        self.hap_positions = [np.array(p, dtype=np.float64) for p in hap_cfg["positions"][: self.num_haps]]
        self.hap_cpu = float(hap_cfg["max_cpu"])
        self.hap_bw = float(hap_cfg["max_bandwidth"])

        self.num_offload_targets = 1 + self.num_uavs + self.num_haps
        self.action_dim_per_terminal = 4
        self.short_action_dim = self.num_terminals * self.action_dim_per_terminal
        self.long_action_dim = 3 * self.num_uavs + self.num_tiers

        self.local_obs_dim = self._local_obs_dim()
        self.global_obs_dim = self._global_obs_dim()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.global_obs_dim,), dtype=np.float32
        )
        self.short_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.short_action_dim,), dtype=np.float32
        )
        self.long_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.long_action_dim,), dtype=np.float32
        )
        self.action_space = self.short_action_space

        self.rng = np.random.default_rng(self.seed_val)
        self.users: List[UserState] = []
        self.terminals: List[TerminalState] = []
        self.uavs: List[UAVState] = []
        self.haps: List[HAPState] = []
        self.slot_idx = 0
        self.frame_idx = 0
        self.tier_weights = np.ones(self.num_tiers) / self.num_tiers
        self.last_step_results: List[StepResult] = []
        self.episode_metrics: Dict[str, float] = {}
        self._tier_total: Dict[str, int] = {}
        self._tier_met: Dict[str, int] = {}

    def _local_obs_dim(self) -> int:
        # queue, energy, device_oh, tier_oh, user_xy(2), dist_uav(M), dist_hap(K), uav_load(M), hap_load(K)
        return (
            2
            + 1
            + 1
            + self.num_device_types
            + self.num_tiers
            + self.num_uavs
            + self.num_haps
            + self.num_uavs
            + self.num_haps
        )

    def _global_obs_dim(self) -> int:
        uav_block = self.num_uavs * 8
        hap_block = self.num_haps * 4
        return uav_block + hap_block + self.num_terminals * self.local_obs_dim

    @property
    def frame_state_dim(self) -> int:
        return self.global_obs_dim + 3 + self.num_tiers

    def _device_onehot(self, device_type: str) -> np.ndarray:
        oh = np.zeros(self.num_device_types, dtype=np.float32)
        if device_type in self.device_type_names:
            oh[self.device_type_names.index(device_type)] = 1.0
        return oh

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.slot_idx = 0
        self.frame_idx = 0
        self.tier_weights = np.ones(self.num_tiers) / self.num_tiers
        self.last_step_results = []
        self.episode_metrics = {
            "tasks_total": 0,
            "tasks_sla_met": 0,
            "total_energy": 0.0,
            "total_delay": 0.0,
        }
        self._tier_total: Dict[str, int] = {}
        self._tier_met: Dict[str, int] = {}
        self._hotspot = self.rng.uniform(0.2, 0.8, size=2) * self.area_size
        self._refresh_hap_availability()

        self.users = []
        for i in range(self.num_users):
            pos = self.rng.uniform(0, self.area_size, size=3)
            pos[2] = 0.0
            self.users.append(
                UserState(user_id=i, position=pos, mobility_speed=self.user_mobility)
            )

        self.terminals = []
        tid = 0
        for u in self.users:
            for j in range(self.terminals_per_user):
                dtype = self.terminal_assignment[j % len(self.terminal_assignment)]
                tcfg = self.terminal_type_cfg[dtype]
                cpu = self.user_cpu * float(tcfg["cpu_scale"])
                power = self.user_power * float(tcfg["power_scale"])
                self.terminals.append(
                    TerminalState(
                        terminal_id=tid,
                        user_id=u.user_id,
                        device_type=dtype,
                        position=u.position.copy(),
                        cpu_freq=cpu,
                        tx_power=power,
                    )
                )
                tid += 1

        self.uavs = []
        for m in range(self.num_uavs):
            self.uavs.append(
                UAVState(
                    uav_id=m,
                    position=self.uav_inits[m].copy(),
                    velocity=np.zeros(3),
                    cpu_freq=self.uav_cpu,
                    bandwidth=self.uav_bw,
                )
            )

        self.haps = []
        for k in range(self.num_haps):
            pos = self.hap_positions[k] if k < len(self.hap_positions) else self.hap_positions[-1]
            self.haps.append(
                HAPState(
                    hap_id=k,
                    position=pos.copy(),
                    cpu_freq=self.hap_cpu,
                    bandwidth=self.hap_bw,
                )
            )

        return self._get_obs(), {}

    def _terminal_local_obs(self, term: TerminalState) -> np.ndarray:
        user = self.users[term.user_id]
        tier_oh = np.zeros(self.num_tiers, dtype=np.float32)
        if term.queue:
            tname = term.queue[0].tier
            names = [t.name for t in self.sla_tiers]
            if tname in names:
                tier_oh[names.index(tname)] = 1.0

        dist_uav = np.array(
            [distance_3d(term.position, u.position) / self.area_size for u in self.uavs],
            dtype=np.float32,
        )
        dist_hap = np.array(
            [distance_3d(term.position, h.position) / self.area_size for h in self.haps],
            dtype=np.float32,
        )
        if self.num_haps == 0:
            dist_hap = np.zeros(0, dtype=np.float32)

        uav_load = np.array([len(u.queue) / 10.0 for u in self.uavs], dtype=np.float32)
        hap_load = np.array([len(h.queue) / 10.0 for h in self.haps], dtype=np.float32)
        if self.num_haps == 0:
            hap_load = np.zeros(0, dtype=np.float32)

        return np.concatenate(
            [
                user.position[:2] / self.area_size,
                [len(term.queue) / 10.0],
                [term.energy / 100.0],
                self._device_onehot(term.device_type),
                tier_oh,
                dist_uav,
                dist_hap,
                uav_load,
                hap_load,
            ]
        ).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        uav_parts = []
        for u in self.uavs:
            uav_parts.extend(
                [
                    u.position / self.area_size,
                    u.velocity / max(self.uav_max_speed, 1e-6),
                    [len(u.queue) / 10.0],
                    [u.energy / 5000.0],
                ]
            )
        hap_parts = []
        for h in self.haps:
            hap_parts.extend(
                [
                    h.position / max(self.area_size, h.position[2]),
                    [len(h.queue) / 10.0],
                ]
            )
        term_parts = [self._terminal_local_obs(t) for t in self.terminals]
        blocks = []
        if uav_parts:
            blocks.append(np.concatenate([np.asarray(p, dtype=np.float32).flatten() for p in uav_parts]))
        if hap_parts:
            blocks.append(np.concatenate([np.asarray(p, dtype=np.float32).flatten() for p in hap_parts]))
        blocks.append(np.concatenate(term_parts))
        return np.concatenate(blocks).astype(np.float32)

    def get_local_obs_batch(self) -> np.ndarray:
        return np.stack([self._terminal_local_obs(t) for t in self.terminals], axis=0)

    def set_sla_lambda(self, lam) -> None:
        """Update the SLA-violation Lagrange multiplier(s).

        Accepts a scalar (applied to all tiers) or a per-tier dict.
        """
        if isinstance(lam, dict):
            for name, val in lam.items():
                if name in self.sla_lambda_by_tier:
                    self.sla_lambda_by_tier[name] = float(val)
            self.sla_lambda = float(np.mean(list(self.sla_lambda_by_tier.values())))
        else:
            self.sla_lambda = float(lam)
            for name in self.sla_lambda_by_tier:
                self.sla_lambda_by_tier[name] = float(lam)

    def _mask_delay_budget(self, tier: SLATier) -> float:
        margin = self.mask_delay_margin_by_tier.get(tier.name, self.mask_delay_margin)
        return margin * tier.t_max

    def _hap_is_available(self, hap_id: int) -> bool:
        if self.disable_hap or hap_id >= len(self._hap_available):
            return False
        return bool(self._hap_available[hap_id])

    def _refresh_hap_availability(self) -> None:
        self._hap_available = [
            self.rng.random() < self.hap_availability for _ in range(self.num_haps)
        ]

    def feasible_target_count(self) -> int:
        """Average feasible offload options per queued terminal (mobility diagnostic)."""
        total = 0
        queued = 0
        for term in self.terminals:
            if not term.queue:
                continue
            tier = self.tier_map[term.queue[0].tier]
            mu = self._min_mu_by_tier[tier.name]
            feasible = self._feasible_targets(term, tier, mu, 0.5, 0.5)
            total += len(feasible)
            queued += 1
        return total if queued == 0 else total

    def tier_sla_rates(self) -> Dict[str, float]:
        """Per-tier SLA satisfaction rate for the most recent episode."""
        return {
            t.name: (self._tier_met.get(t.name, 0) / self._tier_total[t.name])
            if self._tier_total.get(t.name, 0) > 0
            else 1.0
            for t in self.sla_tiers
        }

    def _min_mu_for_sim(self, s_min: float) -> float:
        """Minimum semantic ratio mu so that similarity(mu) >= s_min (+margin)."""
        target = min(max(s_min + self.mask_sim_margin, 0.0), 1.0 - 1e-6)
        ratio = min(max(target / max(self.s_max, 1e-6), 0.0), 1.0 - 1e-6)
        mu = -np.log(1.0 - ratio) / max(self.lambda_acc, 1e-6)
        return float(min(max(mu, self.mu_min), self.mu_max))

    def _det_gain(self, ue_pos: np.ndarray, aerial_pos: np.ndarray) -> float:
        """Deterministic channel gain for feasibility estimation.

        mask_channel selects optimism: 'los' (best case), 'nlos' (worst case),
        or 'expected' (LoS-probability weighted) for a realistic estimate.
        """
        dist = distance_3d(ue_pos, aerial_pos)
        g_los = 10 ** (-path_loss_db(dist, self.carrier_freq, los=True) / 10.0)
        if self.mask_channel == "los":
            return g_los
        g_nlos = 10 ** (-path_loss_db(dist, self.carrier_freq, los=False) / 10.0)
        if self.mask_channel == "nlos":
            return g_nlos
        horiz = max(np.linalg.norm(ue_pos[:2] - aerial_pos[:2]), 1.0)
        elev = np.degrees(np.arctan2(aerial_pos[2] - ue_pos[2], horiz))
        p_los = los_probability(elev, self.los_a, self.los_b)
        return p_los * g_los + (1.0 - p_los) * g_nlos

    def _estimate_delay(
        self, term, kind: int, tid: int, mu: float, cpu_frac: float, bw_frac: float
    ) -> float:
        """Deterministic delay estimate for the feasibility mask (nominal task)."""
        t_sem = semantic_extraction_delay(self.task_data, mu, term.cpu_freq, self.c_sem)
        if kind == TARGET_LOCAL:
            t_comp = computation_delay(self.task_cycles, term.cpu_freq * cpu_frac)
            return t_sem + t_comp
        data_tx = transmission_bits(self.task_data, mu)
        cycles_remote = remote_cpu_cycles(self.task_cycles, mu)
        if kind == TARGET_UAV:
            node = self.uavs[tid]
        else:
            node = self.haps[tid]
        bw = node.bandwidth * bw_frac + 1e5
        rate = shannon_rate(bw, term.tx_power, self._det_gain(term.position, node.position), self.noise_power)
        t_tx = transmission_delay(data_tx, rate)
        t_comp = computation_delay(cycles_remote, node.cpu_freq * cpu_frac)
        return t_sem + t_tx + t_comp

    def _feasible_targets(self, term, tier: SLATier, mu: float, cpu_est: float, bw_est: float):
        """Candidate offload targets whose estimated delay meets the tier deadline."""
        candidates = [(TARGET_LOCAL, 0)]
        candidates += [(TARGET_UAV, m) for m in range(self.num_uavs)]
        if not self.disable_hap:
            candidates += [
                (TARGET_HAP, k) for k in range(self.num_haps) if self._hap_is_available(k)
            ]
        budget = self._mask_delay_budget(tier)
        feasible = [
            (k, t)
            for (k, t) in candidates
            if self._estimate_delay(term, k, t, mu, cpu_est, bw_est) <= budget
        ]
        if not feasible:
            # No target can meet the deadline -> pick the lowest-delay one.
            feasible = [
                min(
                    candidates,
                    key=lambda kt: self._estimate_delay(term, kt[0], kt[1], mu, cpu_est, bw_est),
                )
            ]
        return feasible

    def apply_long_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        for m, uav in enumerate(self.uavs):
            delta = action[m * 3 : m * 3 + 3] * self.uav_max_speed * self.slot_duration
            uav.position = uav.position + delta
            uav.position[:2] = np.clip(uav.position[:2], 0, self.area_size)
            uav.position[2] = np.clip(uav.position[2], self.uav_alt_min, self.uav_alt_max)
            uav.velocity = delta / max(self.slot_duration, 1e-6)

        tw = action[3 * self.num_uavs : 3 * self.num_uavs + self.num_tiers] + 1.0
        if tw.size == self.num_tiers:
            tw = np.clip(tw, 1e-6, None)
            self.tier_weights = tw / tw.sum()

    def _decode_offload(self, raw: float) -> Tuple[str, int]:
        raw = float(np.clip(raw, 0.0, 1.0))
        n = self.num_offload_targets
        idx = min(int(raw * n), n - 1)
        if idx == 0:
            return TARGET_LOCAL, 0
        if idx <= self.num_uavs:
            return TARGET_UAV, idx - 1
        return TARGET_HAP, idx - 1 - self.num_uavs

    def _parse_short_action(self, action: np.ndarray) -> List[dict]:
        action = np.clip(action, -1.0, 1.0).reshape(
            self.num_terminals, self.action_dim_per_terminal
        )
        parsed = []
        for i in range(self.num_terminals):
            a = action[i]
            offload_raw = (a[0] + 1.0) / 2.0
            target_kind, target_id = self._decode_offload(offload_raw)
            if self.disable_hap and target_kind == TARGET_HAP:
                target_kind, target_id = TARGET_UAV, 0
            mu = self.fixed_mu if self.fixed_mu is not None else clip_mu(
                (a[1] + 1.0) / 2.0, self.mu_min, self.mu_max
            )

            # Model-informed feasibility region: guarantee similarity and route
            # the task only to deadline-feasible targets (fixes critical->HAP and
            # the always-failing local-compute trap), all without extra energy.
            term_i = self.terminals[i]
            if (
                self.feasibility_mask_enabled
                and self.fixed_mu is None
                and term_i.queue
            ):
                tier = self.tier_map[term_i.queue[0].tier]
                mu = float(min(max(mu, self._min_mu_by_tier[tier.name]), self.mu_max))
                cpu_est = max((a[3] + 1.0) / 2.0, 0.3)
                bw_est = max((a[2] + 1.0) / 2.0, 0.3)
                feasible = self._feasible_targets(term_i, tier, mu, cpu_est, bw_est)
                idx = min(int(offload_raw * len(feasible)), len(feasible) - 1)
                target_kind, target_id = feasible[idx]

            parsed.append(
                {
                    "target_kind": target_kind,
                    "target_id": target_id,
                    "mu": mu,
                    "bw_req": (a[2] + 1.0) / 2.0,
                    "cpu_req": (a[3] + 1.0) / 2.0,
                }
            )
        return self._apply_resource_contention(parsed)

    def _apply_resource_contention(self, decisions: List[dict]) -> List[dict]:
        if not self.resource_contention:
            for d in decisions:
                d["bw_frac"] = d["bw_req"]
                d["cpu_frac"] = d["cpu_req"] + 0.1
            return decisions

        uav_bw: Dict[int, List[int]] = {m: [] for m in range(self.num_uavs)}
        uav_cpu: Dict[int, List[int]] = {m: [] for m in range(self.num_uavs)}
        hap_bw: Dict[int, List[int]] = {k: [] for k in range(self.num_haps)}
        hap_cpu: Dict[int, List[int]] = {k: [] for k in range(self.num_haps)}

        for i, d in enumerate(decisions):
            if d["target_kind"] == TARGET_UAV:
                m = d["target_id"]
                uav_bw[m].append(i)
                uav_cpu[m].append(i)
            elif d["target_kind"] == TARGET_HAP:
                k = d["target_id"]
                hap_bw[k].append(i)
                hap_cpu[k].append(i)

        def _normalize(indices: List[int], key: str, out_key: str) -> None:
            if not indices:
                return
            reqs = np.array([decisions[i][key] + 1e-3 for i in indices], dtype=np.float64)
            fracs = reqs / reqs.sum()
            # Floor prevents over-throttling when many terminals share one edge node
            min_share = 1.0 / max(len(indices) * 2, 1)
            fracs = np.maximum(fracs, min_share)
            fracs = fracs / fracs.sum()
            for j, idx in enumerate(indices):
                decisions[idx][out_key] = float(fracs[j])

        for m in range(self.num_uavs):
            _normalize(uav_bw[m], "bw_req", "bw_frac")
            _normalize(uav_cpu[m], "cpu_req", "cpu_frac")
        for k in range(self.num_haps):
            _normalize(hap_bw[k], "bw_req", "bw_frac")
            _normalize(hap_cpu[k], "cpu_req", "cpu_frac")

        for i, d in enumerate(decisions):
            if d["target_kind"] == TARGET_LOCAL:
                d["bw_frac"] = 1.0
                d["cpu_frac"] = d["cpu_req"] + 0.1
            elif "bw_frac" not in d:
                d["bw_frac"] = d["bw_req"]
                d["cpu_frac"] = d["cpu_req"] + 0.1
        return decisions

    def _maybe_arrive_tasks(self) -> None:
        for term in self.terminals:
            tcfg = self.terminal_type_cfg[term.device_type]
            rate = self.arrival_rate * float(tcfg["arrival_scale"])
            if self.rng.random() < rate:
                tier = sample_tier(self.sla_tiers, self.rng, self.tier_weights)
                task = Task(
                    terminal_id=term.terminal_id,
                    user_id=term.user_id,
                    device_type=term.device_type,
                    data_bits=self.task_data * self.rng.uniform(0.8, 1.2),
                    cpu_cycles=self.task_cycles * self.rng.uniform(0.8, 1.2),
                    tier=tier.name,
                    arrival_slot=self.slot_idx,
                    deadline=tier.t_max,
                    s_min=tier.s_min,
                )
                term.queue.append(task)

    def _move_users_and_terminals(self) -> None:
        if self.hotspot_drift:
            drift = self.rng.uniform(-1, 1, size=2) * self.hotspot_speed * self.slot_duration
            self._hotspot = np.clip(self._hotspot + drift, 0.1 * self.area_size, 0.9 * self.area_size)
        for user in self.users:
            if self.hotspot_drift:
                direction = self._hotspot - user.position[:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-3:
                    direction = direction / norm
                jitter = self.rng.uniform(-0.35, 0.35, size=2)
                delta = (0.7 * direction + jitter) * user.mobility_speed * self.slot_duration
            else:
                delta = self.rng.uniform(-1, 1, size=2) * user.mobility_speed * self.slot_duration
            user.position[:2] = np.clip(user.position[:2] + delta, 0, self.area_size)
        for term in self.terminals:
            user = self.users[term.user_id]
            tcfg = self.terminal_type_cfg[term.device_type]
            offset = self.rng.uniform(-5, 5, size=2) * float(tcfg.get("mobility_scale", 1.0))
            term.position = user.position.copy()
            term.position[:2] = np.clip(term.position[:2] + offset, 0, self.area_size)

    def _execute_task(self, term: TerminalState, decision: dict) -> Optional[StepResult]:
        if not term.queue:
            return None
        task = term.queue.pop(0)
        tier = self.tier_map[task.tier]
        mu = decision["mu"]
        kind = decision["target_kind"]
        tid = decision["target_id"]
        sim = similarity(mu, self.s_max, self.lambda_acc)
        t_sem = semantic_extraction_delay(task.data_bits, mu, term.cpu_freq, self.c_sem)
        data_tx = transmission_bits(task.data_bits, mu)
        cycles_remote = remote_cpu_cycles(task.cpu_cycles, mu)
        energy = 0.0

        bw_scale = decision.get("bw_frac", 0.5)
        cpu_scale = decision.get("cpu_frac", 0.5) + 0.1

        if kind == TARGET_LOCAL:
            t_comp = computation_delay(task.cpu_cycles, term.cpu_freq * cpu_scale)
            delay = t_sem + t_comp
            energy = 1e-9 * task.cpu_cycles * cpu_scale
        elif kind == TARGET_UAV:
            uav = self.uavs[tid]
            gain = channel_gain(
                term.position, uav.position, self.carrier_freq, self.los_a, self.los_b, self.rng
            )
            bw = uav.bandwidth * bw_scale + 1e5
            rate = shannon_rate(bw, term.tx_power, gain, self.noise_power)
            t_tx = transmission_delay(data_tx, rate)
            t_comp = computation_delay(cycles_remote, uav.cpu_freq * cpu_scale)
            delay = t_sem + t_tx + t_comp
            energy = term.tx_power * t_tx + 1e-9 * cycles_remote
            uav.queue.append(task)
        else:
            hap = self.haps[tid]
            gain = channel_gain(
                term.position, hap.position, self.carrier_freq, self.los_a, self.los_b, self.rng
            )
            bw = hap.bandwidth * bw_scale + 1e5
            rate = shannon_rate(bw, term.tx_power, gain, self.noise_power)
            t_tx = transmission_delay(data_tx, rate)
            t_comp = computation_delay(cycles_remote, hap.cpu_freq * cpu_scale)
            delay = t_sem + t_tx + t_comp
            energy = term.tx_power * t_tx + 1e-9 * cycles_remote * 0.5
            hap.queue.append(task)

        met = sla_satisfied(delay, sim, tier)
        return StepResult(
            terminal_id=term.terminal_id,
            user_id=term.user_id,
            device_type=term.device_type,
            tier=task.tier,
            offload_target=kind,
            target_id=tid,
            mu=mu,
            delay=delay,
            similarity=sim,
            energy=energy,
            sla_met=met,
            t_max=tier.t_max,
            s_min=tier.s_min,
        )

    def _fairness_penalty(self) -> float:
        loads = [len(u.queue) for u in self.uavs]
        if not loads or sum(loads) == 0:
            return 0.0
        arr = np.array(loads, dtype=np.float64)
        jain = (arr.sum() ** 2) / (len(arr) * (arr ** 2).sum() + 1e-6)
        return 1.0 - jain

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._move_users_and_terminals()
        self._maybe_arrive_tasks()
        decisions = self._parse_short_action(action)

        step_results: List[StepResult] = []
        total_reward = 0.0
        tasks_done = 0
        sla_met_count = 0
        total_energy = 0.0
        total_delay = 0.0
        queue_penalty = sum(len(t.queue) for t in self.terminals)

        for term, dec in zip(self.terminals, decisions):
            if term.queue:
                result = self._execute_task(term, dec)
                if result:
                    step_results.append(result)
                    tasks_done += 1
                    total_energy += result.energy
                    total_delay += result.delay
                    if result.sla_met:
                        sla_met_count += 1
                    dv = max(0.0, result.delay - result.t_max)
                    sv = max(0.0, result.s_min - result.similarity)
                    if self.sla_penalty_enabled:
                        sla_p = penalty(dv, sv, self.rho_delay, self.rho_sem)
                        if self.penalty_clip > 0.0:
                            sla_p = min(sla_p, self.penalty_clip)
                    else:
                        sla_p = 0.0

                    energy_term = result.energy
                    if self.energy_reward_clip > 0.0:
                        energy_term = min(energy_term, self.energy_reward_clip)

                    hit = 1.0 if result.sla_met else 0.0
                    # SLA-quality terms (throughput + feasibility shaping - penalty).
                    sla_quality = self.w_throughput * hit - self.w_sla * sla_p
                    if self.shaping_enabled and self.w_shaping > 0.0:
                        # Bounded progress signals in [0, 1]: 1.0 once the
                        # delay/similarity constraint is satisfied, otherwise a
                        # smooth gradient toward feasibility.
                        delay_prog = min(result.t_max / (result.delay + 1e-6), 1.0)
                        sim_prog = min(result.similarity / max(result.s_min, 1e-6), 1.0)
                        sla_quality += self.w_shaping * (delay_prog + sim_prog)

                    if self.lagrangian_enabled:
                        # Multi-constraint CMDP / PPO-Lagrangian: the (bounded)
                        # positive reward is NOT scaled by lambda; the per-tier
                        # dual variable only weights a bounded SLA-violation cost
                        # (1 - hit) in [0, 1]. Once a tier meets its target, its
                        # lambda decays and the energy cost dominates.
                        lam = self.sla_lambda_by_tier.get(result.tier, self.sla_lambda)
                        task_reward = (
                            sla_quality
                            - self.w_energy * energy_term
                            - lam * (1.0 - hit)
                        )
                    else:
                        task_reward = sla_quality - self.w_energy * energy_term

                    total_reward += task_reward
                    self._tier_total[result.tier] = self._tier_total.get(result.tier, 0) + 1
                    if result.sla_met:
                        self._tier_met[result.tier] = self._tier_met.get(result.tier, 0) + 1

        traj_energy = 0.0
        for uav in self.uavs:
            speed = float(np.linalg.norm(uav.velocity))
            e = self.uav_prop_coeff * speed**2 * self.slot_duration
            uav.energy -= e
            traj_energy += e
        total_energy += traj_energy

        total_reward -= self.w_queue * queue_penalty / max(self.num_terminals, 1)
        total_reward -= self.w_fairness * self._fairness_penalty()

        self.last_step_results = step_results
        self.slot_idx += 1
        if self.slot_idx % self.slots_per_frame == 0:
            self.frame_idx += 1
            if self.hap_availability < 1.0:
                self._refresh_hap_availability()

        self.episode_metrics["tasks_total"] += tasks_done
        self.episode_metrics["tasks_sla_met"] += sla_met_count
        self.episode_metrics["total_energy"] += total_energy
        self.episode_metrics["total_delay"] += total_delay

        terminated = self.frame_idx >= int(self.cfg.get("num_long_frames", 200))
        info = {
            "step_results": step_results,
            "tasks_done": tasks_done,
            "sla_met": sla_met_count,
            "energy": total_energy,
            "traj_energy": traj_energy,
            "frame_idx": self.frame_idx,
            "slot_idx": self.slot_idx,
        }
        return self._get_obs(), float(total_reward), terminated, False, info

    def get_frame_reward(self, frame_rewards: List[float], traj_energy: float) -> float:
        if not frame_rewards:
            return 0.0
        return float(np.mean(frame_rewards)) - self.w_traj_energy * traj_energy

    def frame_state(self) -> np.ndarray:
        obs = self._get_obs()
        extra = np.array(
            [
                self.frame_idx / max(int(self.cfg.get("num_long_frames", 200)), 1),
                len(self.last_step_results) / max(self.num_terminals, 1),
                self.episode_metrics.get("tasks_sla_met", 0)
                / max(self.episode_metrics.get("tasks_total", 1), 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs, extra, self.tier_weights.astype(np.float32)])

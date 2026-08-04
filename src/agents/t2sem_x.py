"""Ideal-track prototype: T2Sem-X.

T2Sem-X keeps the same environment API as T2SemDRL but adds:
1) structured short-action post-processing,
2) uncertainty-aware per-tier dual updates,
3) conflict-aware credit shaping across SLA tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .t2sem_drl import T2SemDRL


@dataclass
class DualStats:
    mean_deficit: float = 0.0
    var_deficit: float = 0.0


@dataclass
class UncertaintyAwareDual:
    lr: float = 0.08
    ema: float = 0.2
    var_ema: float = 0.2
    risk_scale: float = 0.5
    lam_min: float = 0.0
    lam_max: float = 10.0
    values: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, DualStats] = field(default_factory=dict)

    def ensure(self, tier_names: list[str], init_val: float) -> None:
        for name in tier_names:
            self.values.setdefault(name, float(init_val))
            self.stats.setdefault(name, DualStats())

    def update(self, deficits: Dict[str, float]) -> Dict[str, float]:
        for name, d_raw in deficits.items():
            d = float(d_raw)
            st = self.stats[name]
            st.mean_deficit = (1 - self.ema) * st.mean_deficit + self.ema * d
            centered = d - st.mean_deficit
            st.var_deficit = (1 - self.var_ema) * st.var_deficit + self.var_ema * (centered * centered)
            risk_bonus = self.risk_scale * float(np.sqrt(max(st.var_deficit, 0.0)))
            eff_deficit = st.mean_deficit + risk_bonus
            lam_new = self.values[name] + self.lr * eff_deficit
            self.values[name] = float(np.clip(lam_new, self.lam_min, self.lam_max))
        return dict(self.values)


class T2SemXDRL(T2SemDRL):
    """T2Sem-X prototype.

    Notes:
    - Uses the same policy networks/checkpoint format as T2SemDRL.
    - Adds lightweight action structuring and reward reweighting at runtime.
    """

    def __init__(self, env, device: str = "cpu", use_long_agent: bool = True):
        super().__init__(env=env, device=device, use_long_agent=use_long_agent)
        reward_cfg = env.cfg.get("reward", {})
        tier_names = [t.name for t in env.sla_tiers]
        self.credit_temp = float(reward_cfg.get("ideal_credit_temp", 4.0))
        self.credit_gain = float(reward_cfg.get("ideal_credit_gain", 0.15))
        self.dual = UncertaintyAwareDual(
            lr=float(reward_cfg.get("ideal_dual_lr", 0.08)),
            ema=float(reward_cfg.get("ideal_dual_ema", 0.2)),
            var_ema=float(reward_cfg.get("ideal_dual_var_ema", 0.2)),
            risk_scale=float(reward_cfg.get("ideal_dual_risk_scale", 0.5)),
            lam_min=float(reward_cfg.get("lambda_min", 0.0)),
            lam_max=float(reward_cfg.get("lambda_max", 8.0)),
        )
        self.dual.ensure(tier_names, float(reward_cfg.get("sla_lambda_init", 1.0)))

    def _select_short(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raw = super()._select_short(obs, deterministic=deterministic)
        return self._structured_postprocess(raw)

    def _structured_postprocess(self, action: np.ndarray) -> np.ndarray:
        """Structured action transform preserving env-compatible shape.

        Action format per terminal remains [target, mu, bw, cpu] in [-1,1].
        We enforce smooth bounded transforms to reduce high-variance extremes.
        """
        a = np.asarray(action, dtype=np.float32).copy()
        if a.size != self.env.short_action_dim:
            return np.clip(a, -1.0, 1.0)
        block = a.reshape(self.env.num_terminals, self.env.action_dim_per_terminal)
        # Target score: saturate less aggressively than raw tanh output.
        block[:, 0] = np.tanh(0.75 * block[:, 0])
        # Semantic ratio control: keep central mass and avoid boundary spikes.
        block[:, 1] = np.tanh(0.6 * block[:, 1])
        # Resource controls: mild shrinkage to limit contention shocks.
        block[:, 2:] = np.tanh(0.8 * block[:, 2:])
        return np.clip(block.reshape(-1), -1.0, 1.0)

    def _conflict_aware_reward(self, reward: float) -> float:
        rates = self.env.tier_sla_rates()
        if not rates:
            return reward
        vals = np.array(list(rates.values()), dtype=np.float64)
        # Higher spread means stronger tier conflict; downweight reward slightly
        # to reduce overfitting to easy tiers.
        spread = float(vals.max() - vals.min())
        conflict = np.tanh(self.credit_temp * spread)
        return float(reward - self.credit_gain * conflict)

    def train_episode(self):
        obs, _ = self.env.reset()
        slot_rewards = []
        traj_energy_acc = 0.0
        done = False
        metrics = type("M", (), {})()
        from ..utils.metrics import EpisodeMetrics

        metrics = EpisodeMetrics()

        if self.use_long_agent and self.long is not None:
            frame_state = self.env.frame_state()
            long_action = self.long.select_action(frame_state, explore=True)
            self.env.apply_long_action(long_action)
            prev_frame_state = frame_state

        while not done:
            action = self._select_short(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            adj_reward = self._conflict_aware_reward(reward)
            self.short.store_reward(adj_reward, terminated or truncated)
            slot_rewards.append(adj_reward)
            traj_energy_acc += info.get("traj_energy", 0.0)
            self._accumulate_metrics(metrics, info)
            obs = next_obs
            done = terminated or truncated

            if (
                self.use_long_agent
                and self.long is not None
                and self.env.slot_idx % self.env.slots_per_frame == 0
                and not done
            ):
                frame_reward = self.env.get_frame_reward(slot_rewards, traj_energy_acc)
                next_frame_state = self.env.frame_state()
                next_long_action = self.long.select_action(next_frame_state, explore=True)
                if self.long_smooth_coef > 0.0:
                    frame_reward -= self.long_smooth_coef * float(np.sum((next_long_action - long_action) ** 2))
                self.long.push(prev_frame_state, long_action, frame_reward, next_frame_state, False)
                self.long.update(batch_size=min(64, max(len(self.long.buffer), 1)))
                long_action = next_long_action
                self.env.apply_long_action(long_action)
                prev_frame_state = next_frame_state
                slot_rewards = []
                traj_energy_acc = 0.0

        self.short.update()
        self._finalize_metrics(metrics, slot_rewards)

        # Update per-tier uncertainty-aware dual variables.
        rates = self.env.tier_sla_rates()
        targets = dict(self.env.cfg.get("reward", {}).get("sla_target_by_tier", {}) or {})
        default_tgt = float(self.env.cfg.get("reward", {}).get("sla_target", 0.8))
        deficits = {k: float(targets.get(k, default_tgt) - rates.get(k, 1.0)) for k in self.dual.values}
        self.dual.update(deficits)
        self.env.set_sla_lambda(self.dual.values)
        return metrics

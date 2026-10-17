"""Two-timescale T2Sem-DRL orchestrator for multi-UAV multi-terminal MEC."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..env.multi_mec_env import MultiMECEnv
from ..utils.metrics import EpisodeMetrics
from .ddpg import DDPGLongAgent
from .ppo import PPOShortAgent
from .td3 import TD3LongAgent


class T2SemDRL:
    def __init__(self, env: MultiMECEnv, device: str = "cpu", use_long_agent: bool = True):
        self.env = env
        self.use_long_agent = use_long_agent
        drl = env.cfg["drl"]
        algo = str(drl["short"].get("algo", "shared_ppo"))
        use_shared = algo == "shared_ppo"
        use_independent = algo == "independent_ppo"

        short_cfg = drl["short"]
        self.short = PPOShortAgent(
            obs_dim=env.global_obs_dim,
            act_dim=env.short_action_dim,
            lr=float(short_cfg["lr"]),
            gamma=float(short_cfg["gamma"]),
            gae_lambda=float(short_cfg["gae_lambda"]),
            clip_ratio=float(short_cfg["clip"]),
            update_epochs=int(short_cfg["update_epochs"]),
            hidden_dim=int(short_cfg["hidden_dim"]),
            device=device,
            shared_terminal=use_shared,
            independent_terminal=use_independent,
            local_obs_dim=env.local_obs_dim,
            act_dim_per_terminal=env.action_dim_per_terminal,
            num_terminals=env.num_terminals,
            entropy_coef=float(short_cfg.get("entropy_coef", 0.0)),
            init_log_std=float(short_cfg.get("init_log_std", 0.0)),
            log_std_min=float(short_cfg.get("log_std_min", -5.0)),
            log_std_max=float(short_cfg.get("log_std_max", 2.0)),
            max_grad_norm=float(short_cfg.get("max_grad_norm", 0.5)),
        )
        long_cfg = drl["long"]
        self.long_smooth_coef = float(long_cfg.get("smooth_coef", 0.0))
        self.long: Optional[DDPGLongAgent] = None
        if use_long_agent:
            long_algo = str(long_cfg.get("algo", "DDPG")).lower()
            common = dict(
                obs_dim=env.frame_state_dim,
                act_dim=env.long_action_dim,
                lr_actor=float(long_cfg["lr_actor"]),
                lr_critic=float(long_cfg["lr_critic"]),
                gamma=float(long_cfg["gamma"]),
                tau=float(long_cfg["tau"]),
                hidden_dim=int(long_cfg["hidden_dim"]),
                noise_std=float(long_cfg["noise_std"]),
                device=device,
                action_l2_coef=float(long_cfg.get("action_l2_coef", 0.0)),
            )
            if long_algo == "td3":
                self.long = TD3LongAgent(
                    **common,
                    policy_delay=int(long_cfg.get("policy_delay", 2)),
                    target_noise=float(long_cfg.get("target_noise", 0.2)),
                    noise_clip=float(long_cfg.get("noise_clip", 0.5)),
                )
            else:
                self.long = DDPGLongAgent(**common)

    def _select_short(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        local = self.env.get_local_obs_batch()
        return self.short.select_action(obs, local, deterministic=deterministic)

    def train_episode(self) -> EpisodeMetrics:
        obs, _ = self.env.reset()
        slot_rewards: list = []
        traj_energy_acc = 0.0
        done = False
        metrics = EpisodeMetrics()

        if self.use_long_agent and self.long is not None:
            frame_state = self.env.frame_state()
            long_action = self.long.select_action(frame_state, explore=True)
            self.env.apply_long_action(long_action)
            prev_frame_state = frame_state

        while not done:
            action = self._select_short(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.short.store_reward(reward, terminated or truncated)
            slot_rewards.append(reward)
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
                    # Discourage erratic high-amplitude trajectory changes, which
                    # waste propulsion energy (energy ~ speed^2).
                    frame_reward -= self.long_smooth_coef * float(
                        np.sum((next_long_action - long_action) ** 2)
                    )
                self.long.push(prev_frame_state, long_action, frame_reward, next_frame_state, False)
                self.long.update(batch_size=min(64, max(len(self.long.buffer), 1)))
                long_action = next_long_action
                self.env.apply_long_action(long_action)
                prev_frame_state = next_frame_state
                slot_rewards = []
                traj_energy_acc = 0.0

        self.short.update()
        self._finalize_metrics(metrics, slot_rewards)
        return metrics

    def evaluate_episode(self, seed: Optional[int] = None) -> EpisodeMetrics:
        obs, _ = self.env.reset(seed=seed)
        done = False
        rewards = []
        metrics = EpisodeMetrics()

        if self.use_long_agent and self.long is not None:
            frame_state = self.env.frame_state()
            long_action = self.long.select_action(frame_state, explore=False)
            self.env.apply_long_action(long_action)

        while not done:
            action = self._select_short(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            self._accumulate_metrics(metrics, info)
            done = terminated or truncated

            if (
                self.use_long_agent
                and self.long is not None
                and self.env.slot_idx % self.env.slots_per_frame == 0
                and not done
            ):
                frame_state = self.env.frame_state()
                long_action = self.long.select_action(frame_state, explore=False)
                self.env.apply_long_action(long_action)

        self._finalize_metrics(metrics, rewards)
        return metrics

    @staticmethod
    def _accumulate_metrics(metrics: EpisodeMetrics, info: dict) -> None:
        for r in info.get("step_results", []):
            metrics.tasks_total += 1
            if r.sla_met:
                metrics.tasks_sla_met += 1
            metrics.total_energy += r.energy
            if r.offload_target == "local":
                metrics.offload_local += 1
            elif r.offload_target == "uav":
                metrics.offload_uav += 1
            else:
                metrics.offload_hap += 1

    def _finalize_metrics(self, metrics: EpisodeMetrics, rewards: list) -> None:
        metrics.avg_reward = float(np.mean(rewards)) if rewards else 0.0
        metrics.sla_rate = metrics.tasks_sla_met / metrics.tasks_total if metrics.tasks_total else 0.0
        em = self.env.episode_metrics
        if em.get("tasks_total", 0) > 0:
            metrics.avg_delay = em["total_delay"] / em["tasks_total"]
            metrics.total_energy = em["total_energy"]

    def save(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.short.save(os.path.join(checkpoint_dir, "short_ppo.pt"))
        if self.long is not None:
            self.long.save(os.path.join(checkpoint_dir, "long_ddpg.pt"))

    def load(self, checkpoint_dir: str) -> None:
        self.short.load(os.path.join(checkpoint_dir, "short_ppo.pt"))
        if self.long is not None:
            self.long.load(os.path.join(checkpoint_dir, "long_ddpg.pt"))

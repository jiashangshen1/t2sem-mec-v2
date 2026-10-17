"""Long-timescale TD3 (Twin Delayed DDPG) for multi-UAV trajectory control.

Drop-in alternative to :class:`DDPGLongAgent` (same public interface), adding
the three TD3 fixes for DDPG's overestimation / instability:

1. Twin critics  -- the target uses min(Q1, Q2) to suppress value overestimation.
2. Delayed policy updates -- the actor (and all target networks) refresh only
   every ``policy_delay`` critic updates, decoupling the actor-critic chase.
3. Target policy smoothing -- clipped noise on target actions so the critic
   cannot sharpen on narrow action spikes.

Config: set ``drl.long.algo: td3`` (overrides/improved_* yaml). Extra knobs
``policy_delay`` (default 2), ``target_noise`` (0.2), ``noise_clip`` (0.5).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.replay_buffer import ReplayBuffer
from .networks import DDPGActor, DDPGCritic


class TD3LongAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.005,
        hidden_dim: int = 256,
        noise_std: float = 0.08,
        device: str = "cpu",
        action_l2_coef: float = 0.0,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.action_l2_coef = float(action_l2_coef)
        self.policy_delay = max(1, int(policy_delay))
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
        self._updates = 0

        self.actor = DDPGActor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic1 = DDPGCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic2 = DDPGCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target = DDPGActor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic1_target = DDPGCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic2_target = DDPGCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr_critic
        )
        self.buffer = ReplayBuffer()

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if explore:
            action = action + np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch_size: int = 64) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # (1)+(3) clipped-noise target actions through twin critics, take min.
        with torch.no_grad():
            noise = (
                torch.randn_like(actions_t) * self.target_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states_t) + noise).clamp(-1.0, 1.0)
            q1 = self.critic1_target(next_states_t, next_actions).squeeze(-1)
            q2 = self.critic2_target(next_states_t, next_actions).squeeze(-1)
            y = rewards_t + self.gamma * (1 - dones_t) * torch.min(q1, q2)

        q1_cur = self.critic1(states_t, actions_t).squeeze(-1)
        q2_cur = self.critic2(states_t, actions_t).squeeze(-1)
        critic_loss = nn.functional.mse_loss(q1_cur, y) + nn.functional.mse_loss(q2_cur, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # (2) delayed actor + target updates: refresh only every d-th update.
        self._updates += 1
        if self._updates % self.policy_delay == 0:
            actor_pred = self.actor(states_t)
            actor_loss = -self.critic1(states_t, actor_pred).squeeze(-1).mean()
            if self.action_l2_coef > 0.0:
                actor_loss = actor_loss + self.action_l2_coef * (actor_pred ** 2).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
        return float(critic_loss.item()) / 2.0

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target.load_state_dict(ckpt["actor"])
        self.critic1_target.load_state_dict(ckpt["critic1"])
        self.critic2_target.load_state_dict(ckpt["critic2"])

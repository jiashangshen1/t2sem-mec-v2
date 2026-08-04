"""Short-timescale PPO with optional parameter-shared terminal actor."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorGaussian, Critic, IndependentTerminalActorBank, SharedTerminalActor


class PPOShortAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        update_epochs: int = 6,
        hidden_dim: int = 256,
        device: str = "cpu",
        *,
        shared_terminal: bool = False,
        independent_terminal: bool = False,
        local_obs_dim: int = 0,
        act_dim_per_terminal: int = 4,
        num_terminals: int = 0,
        entropy_coef: float = 0.0,
        init_log_std: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        max_grad_norm: float = 0.5,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.shared_terminal = shared_terminal
        self.independent_terminal = independent_terminal
        self.local_obs_dim = local_obs_dim
        self.act_dim_per_terminal = act_dim_per_terminal
        self.num_terminals = num_terminals
        self.act_dim = act_dim
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)

        std_kwargs = dict(
            init_log_std=init_log_std,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        if independent_terminal:
            self.actor = IndependentTerminalActorBank(
                num_terminals, local_obs_dim, act_dim_per_terminal, hidden_dim, **std_kwargs
            ).to(self.device)
        elif shared_terminal:
            self.actor = SharedTerminalActor(
                local_obs_dim, act_dim_per_terminal, hidden_dim, **std_kwargs
            ).to(self.device)
        else:
            self.actor = ActorGaussian(obs_dim, act_dim, hidden_dim, **std_kwargs).to(self.device)
        self.critic = Critic(obs_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.states: List[np.ndarray] = []
        self.local_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def _actor_forward(self, global_obs: np.ndarray, local_batch: np.ndarray | None):
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if (self.shared_terminal or self.independent_terminal) and local_batch is not None:
            local_t = torch.as_tensor(local_batch, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.actor(local_t)
            value = self.critic(global_t)
            return dist, value, global_t, local_t
        obs_t = global_t
        dist = self.actor(obs_t)
        value = self.critic(obs_t)
        return dist, value, obs_t, None

    def select_action(
        self,
        obs: np.ndarray,
        local_obs: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        dist, value, global_t, local_t = self._actor_forward(obs, local_obs)
        with torch.no_grad():
            if deterministic:
                if self.shared_terminal or self.independent_terminal:
                    action = dist.mean.squeeze(0).reshape(-1)
                else:
                    action = dist.mean.squeeze(0)
            else:
                if self.shared_terminal or self.independent_terminal:
                    action = dist.sample().squeeze(0).reshape(-1)
                else:
                    action = dist.sample().squeeze(0)
            log_prob = dist.log_prob(
                action.reshape(1, self.num_terminals, self.act_dim_per_terminal)
                if (self.shared_terminal or self.independent_terminal)
                else action.unsqueeze(0)
            ).sum(-1)
            if self.shared_terminal or self.independent_terminal:
                log_prob = log_prob.sum(-1)

        act_np = action.detach().cpu().numpy().astype(np.float32)
        if not deterministic:
            self.states.append(obs)
            if local_obs is not None:
                self.local_states.append(local_obs)
            self.actions.append(act_np)
            self.log_probs.append(float(log_prob.item()))
            self.values.append(float(value.item()))
        return act_np

    def pretrain_behavioral(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        local_observations: np.ndarray | None = None,
        terminal_indices: np.ndarray | None = None,
        epochs: int = 80,
        batch_size: int = 64,
    ) -> float:
        last_loss = 0.0
        if self.independent_terminal and local_observations is not None:
            loc_arr = np.asarray(local_observations, dtype=np.float32)
            act_arr = np.asarray(actions, dtype=np.float32)
            if loc_arr.ndim == 3:
                n_steps, n_term, _ = loc_arr.shape
                loc_flat = loc_arr.reshape(-1, self.local_obs_dim)
                act_flat = act_arr.reshape(-1, self.act_dim_per_terminal)
                term_idx = np.tile(np.arange(n_term, dtype=np.int64), n_steps)
            else:
                loc_flat = loc_arr.reshape(-1, self.local_obs_dim)
                act_flat = act_arr.reshape(-1, self.act_dim_per_terminal)
                if terminal_indices is None:
                    return 0.0
                term_idx = np.asarray(terminal_indices, dtype=np.int64).reshape(-1)
            m = min(loc_flat.shape[0], act_flat.shape[0], term_idx.shape[0])
            loc_flat, act_flat, term_idx = loc_flat[:m], act_flat[:m], term_idx[:m]
            n = loc_flat.shape[0]
            if n == 0:
                return 0.0
            for _ in range(epochs):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                self.optimizer.zero_grad()
                batch_loss = torch.zeros((), device=self.device)
                for i in idx:
                    t_i = int(term_idx[i])
                    loc = torch.as_tensor(loc_flat[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_t = torch.as_tensor(act_flat[i], dtype=torch.float32, device=self.device)
                    pred = self.actor.actors[t_i](loc).mean.squeeze(0)
                    batch_loss = batch_loss + nn.functional.mse_loss(pred, act_t)
                loss = batch_loss / len(idx)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())
            return last_loss

        if self.shared_terminal and local_observations is not None:
            # Flatten everything to per-terminal rows so the parameter-shared
            # actor is cloned at its natural granularity (one local obs -> one
            # terminal action). Works for both full-action demos of shape
            # (n, T, d)/(n, T*apt) and pre-filtered per-terminal demos of
            # shape (n, d)/(n, apt).
            loc_arr = np.asarray(local_observations, dtype=np.float32).reshape(
                -1, self.local_obs_dim
            )
            act_arr = np.asarray(actions, dtype=np.float32).reshape(
                -1, self.act_dim_per_terminal
            )
            m = min(loc_arr.shape[0], act_arr.shape[0])
            loc_arr, act_arr = loc_arr[:m], act_arr[:m]
            n = loc_arr.shape[0]
            if n == 0:
                return 0.0
            for _ in range(epochs):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                loc = torch.as_tensor(loc_arr[idx], dtype=torch.float32, device=self.device)
                act_t = torch.as_tensor(act_arr[idx], dtype=torch.float32, device=self.device)
                pred = torch.tanh(self.actor.net(loc))
                loss = nn.functional.mse_loss(pred, act_t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())
            return last_loss

        n = observations.shape[0]
        for _ in range(epochs):
            idx = np.random.randint(0, n, size=min(batch_size, n))
            obs_t = torch.as_tensor(observations[idx], dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(actions[idx], dtype=torch.float32, device=self.device)
            pred = torch.tanh(self.actor.net(obs_t))
            loss = nn.functional.mse_loss(pred, act_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        return last_loss

    def store_reward(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [0.0], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            adv[t] = last_gae
        returns = adv + values[:-1]
        return adv, returns

    def update(self) -> Dict[str, float]:
        if len(self.states) < 2:
            self._clear()
            return {"policy_loss": 0.0, "value_loss": 0.0}

        adv, returns = self._compute_gae()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        policy_loss_val = 0.0
        value_loss_val = 0.0
        entropy_val = 0.0
        for _ in range(self.update_epochs):
            if (self.shared_terminal or self.independent_terminal) and self.local_states:
                local = torch.as_tensor(np.array(self.local_states), dtype=torch.float32, device=self.device)
                dist = self.actor(local)
                act_shaped = actions.reshape(-1, self.num_terminals, self.act_dim_per_terminal)
                log_probs = dist.log_prob(act_shaped).sum(-1).sum(-1)
            else:
                dist = self.actor(states)
                log_probs = dist.log_prob(actions).sum(-1)

            entropy = dist.entropy().sum(-1)
            while entropy.dim() > 1:
                entropy = entropy.sum(-1)
            entropy = entropy.mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(self.critic(states).squeeze(-1), returns_t)
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            policy_loss_val = float(policy_loss.item())
            value_loss_val = float(value_loss.item())
            entropy_val = float(entropy.item())

        self._clear()
        return {"policy_loss": policy_loss_val, "value_loss": value_loss_val, "entropy": entropy_val}

    def _clear(self) -> None:
        self.states.clear()
        self.local_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path: str) -> None:
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

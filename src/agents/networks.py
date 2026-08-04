"""Neural network modules for shared PPO and multi-UAV DDPG."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes: list, activation=nn.Tanh, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorGaussian(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        *,
        init_log_std: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: torch.Tensor) -> Normal:
        mu = torch.tanh(self.net(obs))
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)


class IndependentTerminalActorBank(nn.Module):
    """Decentralized actors: one Gaussian policy per terminal (MAPPO-style execution)."""

    def __init__(
        self,
        num_terminals: int,
        local_obs_dim: int,
        act_dim_per_terminal: int,
        hidden: int = 256,
        *,
        init_log_std: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.num_terminals = num_terminals
        self.act_dim = act_dim_per_terminal
        self.actors = nn.ModuleList(
            [
                SharedTerminalActor(
                    local_obs_dim,
                    act_dim_per_terminal,
                    hidden,
                    init_log_std=init_log_std,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                )
                for _ in range(num_terminals)
            ]
        )

    def forward(self, local_obs: torch.Tensor) -> Normal:
        # local_obs: (batch, T, local_dim)
        if local_obs.dim() == 2:
            local_obs = local_obs.unsqueeze(0)
        b, t, d = local_obs.shape
        mus = []
        stds = []
        for j in range(t):
            dist_j = self.actors[j](local_obs[:, j, :])
            mus.append(dist_j.loc)
            stds.append(dist_j.scale)
        mu = torch.stack(mus, dim=1)
        std = torch.stack(stds, dim=1)
        return Normal(mu, std)


class SharedTerminalActor(nn.Module):
    """Parameter-shared actor: same weights for every terminal local observation."""

    def __init__(
        self,
        local_obs_dim: int,
        act_dim_per_terminal: int,
        hidden: int = 256,
        *,
        init_log_std: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.act_dim = act_dim_per_terminal
        self.net = mlp([local_obs_dim, hidden, hidden, act_dim_per_terminal])
        self.log_std = nn.Parameter(torch.full((act_dim_per_terminal,), float(init_log_std)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, local_obs: torch.Tensor) -> Normal:
        # local_obs: (batch, local_dim) or (batch, T, local_dim)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        if local_obs.dim() == 3:
            b, t, d = local_obs.shape
            flat = local_obs.reshape(b * t, d)
            mu = torch.tanh(self.net(flat)).reshape(b, t, self.act_dim)
            std = torch.exp(log_std).expand_as(mu)
            return Normal(mu, std)
        mu = torch.tanh(self.net(local_obs))
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DDPGActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))

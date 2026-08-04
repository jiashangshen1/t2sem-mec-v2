"""Local-only baseline."""

from __future__ import annotations

import numpy as np

from ..env.multi_mec_env import MultiMECEnv


class LocalOnlyPolicy:
    def __init__(self, env: MultiMECEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = np.zeros(self.env.short_action_dim, dtype=np.float32)
        mu_a = 1.0
        for i in range(self.env.num_terminals):
            base = i * self.env.action_dim_per_terminal
            action[base : base + 4] = np.array([-1.0, mu_a, 0.0, 0.0], dtype=np.float32)
        return action

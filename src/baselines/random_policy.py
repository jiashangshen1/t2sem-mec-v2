"""Uniform random baseline."""

from __future__ import annotations

import numpy as np

from ..env.multi_mec_env import MultiMECEnv


class RandomPolicy:
    def __init__(self, env: MultiMECEnv, seed: int = 42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.env.short_action_dim).astype(np.float32)

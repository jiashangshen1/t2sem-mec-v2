"""Sanity checks for multi-UAV multi-terminal environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env.multi_mec_env import MultiMECEnv
from src.utils.config import load_config


def test_reset_step_shapes():
    cfg = load_config("configs/default.yaml", "configs/smoke.yaml")
    env = MultiMECEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.global_obs_dim,)
    action = np.zeros(env.short_action_dim, dtype=np.float32)
    obs2, reward, term, trunc, info = env.step(action)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    print(
        f"OK: users={env.num_users} terminals={env.num_terminals} "
        f"uavs={env.num_uavs} haps={env.num_haps} obs={obs.shape} act={env.short_action_dim}"
    )


def test_resource_contention():
    cfg = load_config("configs/default.yaml", "configs/smoke.yaml")
    env = MultiMECEnv(cfg, resource_contention=True)
    env.reset(seed=1)
    action = np.ones(env.short_action_dim, dtype=np.float32)
    # all terminals request high bw/cpu
    _, _, _, _, info = env.step(action)
    assert "step_results" in info
    print(f"OK: contention step tasks={info.get('tasks_done', 0)}")


def test_long_action_dim():
    cfg = load_config("configs/default.yaml")
    env = MultiMECEnv(cfg)
    assert env.long_action_dim == 3 * env.num_uavs + env.num_tiers
    la = np.zeros(env.long_action_dim, dtype=np.float32)
    env.reset()
    env.apply_long_action(la)
    print(f"OK: long_action_dim={env.long_action_dim}")


if __name__ == "__main__":
    test_reset_step_shapes()
    test_resource_contention()
    test_long_action_dim()
    print("All sanity tests passed.")

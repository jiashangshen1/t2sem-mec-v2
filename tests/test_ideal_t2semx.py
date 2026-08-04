"""Sanity checks for ideal-track T2Sem-X prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env
from src.agents.t2sem_x import T2SemXDRL
from src.utils.config import load_config


def test_t2semx_action_shape():
    cfg = load_config("configs/default.yaml", "configs/ideal_v1_base.yaml", "configs/ideal_v1_t2semx.yaml")
    env = build_env(cfg, "full")
    agent = T2SemXDRL(env, device="cpu", use_long_agent=False)
    obs, _ = env.reset(seed=42)
    action = agent._select_short(obs, deterministic=True)
    assert action.shape == (env.short_action_dim,)
    assert np.all(action <= 1.0 + 1e-6)
    assert np.all(action >= -1.0 - 1e-6)


if __name__ == "__main__":
    test_t2semx_action_shape()
    print("T2Sem-X sanity passed")

"""Behavioral-cloning warm-start demonstration collectors.

Two strategies are available:

* ``collect_demonstrations`` -- legacy greedy-only collector. Greedy attains
  ~0% SLA in the v2 multi-UAV setting, so cloning it pulls the policy into a
  poor basin. Kept for backward compatibility.
* ``collect_filtered_demonstrations`` -- runs a mix of exploratory policies
  (random with rotating seeds + greedy) and keeps only the per-terminal
  (local_obs, action) pairs whose resulting task actually met its SLA. This
  gives the shared terminal actor demonstrations of *successful* behaviour.
"""

from __future__ import annotations

import numpy as np

from ..baselines.greedy import GreedyPolicy
from ..baselines.random_policy import RandomPolicy
from ..env.multi_mec_env import MultiMECEnv


def collect_demonstrations(
    env: MultiMECEnv, num_episodes: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = GreedyPolicy(env)
    global_obs, actions, local_obs = [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            local = env.get_local_obs_batch()
            act = policy.act(obs)
            global_obs.append(obs)
            actions.append(act)
            local_obs.append(local)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
    return (
        np.array(global_obs, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(local_obs, dtype=np.float32),
    )


def collect_filtered_demonstrations(
    env: MultiMECEnv,
    num_episodes: int,
    seed: int = 42,
    min_samples: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Collect SLA-successful per-terminal demonstrations from mixed policies.

    Returns ``(local_obs, per_terminal_actions, terminal_indices)`` where
    ``local_obs`` has shape ``(N, local_obs_dim)``,
    ``per_terminal_actions`` has shape ``(N, action_dim_per_terminal)``, and
    ``terminal_indices`` has shape ``(N,)``. Returns ``None`` if too few
    successful samples were found (caller should fall back to the legacy collector).
    """
    apt = env.action_dim_per_terminal
    id_to_idx = {t.terminal_id: i for i, t in enumerate(env.terminals)}

    local_samples: list[np.ndarray] = []
    act_samples: list[np.ndarray] = []
    term_samples: list[int] = []

    for ep in range(num_episodes):
        # Rotate policies: bias toward exploratory random, sprinkle greedy.
        if ep % 3 == 2:
            policy = GreedyPolicy(env)
        else:
            policy = RandomPolicy(env, seed=seed * 131 + ep)

        obs, _ = env.reset(seed=seed + ep)
        # terminal ordering is stable within an episode; rebuild map per reset.
        id_to_idx = {t.terminal_id: i for i, t in enumerate(env.terminals)}
        done = False
        while not done:
            local = env.get_local_obs_batch()
            act = policy.act(obs)
            obs, _, term, trunc, info = env.step(act)
            for r in info.get("step_results", []):
                if getattr(r, "sla_met", False):
                    idx = id_to_idx.get(r.terminal_id)
                    if idx is None:
                        continue
                    local_samples.append(np.asarray(local[idx], dtype=np.float32))
                    act_samples.append(
                        np.asarray(act[idx * apt : (idx + 1) * apt], dtype=np.float32)
                    )
                    term_samples.append(int(idx))
            done = term or trunc

    if len(local_samples) < max(min_samples, 1):
        return None

    return (
        np.stack(local_samples, axis=0).astype(np.float32),
        np.stack(act_samples, axis=0).astype(np.float32),
        np.asarray(term_samples, dtype=np.int64),
    )

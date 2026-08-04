"""Benchmark per-episode runtime for full-experiment time estimate."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.t2sem_drl import T2SemDRL
from src.agents.warmstart import collect_demonstrations
from src.baselines.greedy import GreedyPolicy
from src.baselines.tjcct import TJCCTHeuristicPolicy
from src.env.multi_mec_env import MultiMECEnv
from src.utils.config import load_config
from src.utils.device import device_from_config


def run_baseline_ep(env: MultiMECEnv, policy, seed: int = 42) -> float:
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(policy.act(obs))
        done = term or trunc
    return time.perf_counter() - t0


def main() -> None:
    cfg = load_config("configs/default.yaml")
    device = device_from_config("auto", cfg)
    train_eps = int(cfg["drl"]["total_long_frames"])
    eval_eps = int(cfg["experiment"]["eval_episodes"])
    n_seeds = len(cfg["experiment"]["eval_seeds"])
    bc_demos = int(cfg["drl"]["bc_demo_episodes"])
    bc_epochs = int(cfg["drl"]["bc_train_epochs"])
    steps = int(cfg["num_long_frames"]) * int(cfg["slots_per_frame"])

    print("=== Scale ===")
    print(
        f"terminals={cfg['num_users']*cfg['terminals_per_user']} "
        f"uavs={cfg['num_uavs']} haps={cfg['num_haps']}"
    )
    print(f"train_eps={train_eps} eval_eps={eval_eps} seeds={n_seeds} steps/ep={steps}")
    print(f"device={device}")

    env = MultiMECEnv(cfg)
    greedy_t = run_baseline_ep(env, GreedyPolicy(env))
    tjcct_t = run_baseline_ep(MultiMECEnv(cfg), TJCCTHeuristicPolicy(MultiMECEnv(cfg)))

    env_bc = MultiMECEnv(cfg)
    t0 = time.perf_counter()
    g, a, l = collect_demonstrations(env_bc, bc_demos, seed=42)
    bc_collect = time.perf_counter() - t0

    agent = T2SemDRL(env_bc, device=device)
    t0 = time.perf_counter()
    agent.short.pretrain_behavioral(g, a, l, epochs=bc_epochs)
    bc_train = time.perf_counter() - t0

    env_drl = MultiMECEnv(cfg)
    agent2 = T2SemDRL(env_drl, device=device)
    t0 = time.perf_counter()
    m = agent2.train_episode()
    drl_ep = time.perf_counter() - t0

    t0 = time.perf_counter()
    agent2.evaluate_episode(seed=42)
    eval_ep = time.perf_counter() - t0

    print("\n=== Measured (seconds) ===")
    print(f"greedy_1ep      {greedy_t:.2f}")
    print(f"tjcct_1ep       {tjcct_t:.2f}")
    print(f"bc_collect      {bc_collect:.2f}")
    print(f"bc_train        {bc_train:.2f}")
    print(f"drl_train_1ep   {drl_ep:.2f}  (SLA={m.sla_rate:.3f})")
    print(f"drl_eval_1ep    {eval_ep:.2f}")

    baselines = ["greedy", "tjcct", "random", "local_only"]
    t_baseline_train = train_eps * greedy_t * len(baselines)  # upper bound if all run train_eps
    t_baseline_eval = eval_eps * n_seeds * greedy_t * len(baselines)

    t_drl = bc_collect + bc_train + train_eps * drl_ep
    t_drl_eval = eval_eps * n_seeds * eval_ep

    print("\n=== Full experiment estimate ===")
    print(f"T2Sem-DRL train (BC+{train_eps} ep): {t_drl/60:.1f} min")
    print(f"T2Sem-DRL eval ({eval_eps} ep x {n_seeds} seeds): {t_drl_eval/60:.1f} min")
    print(f"Baselines eval-only ({eval_eps} ep x {n_seeds} seeds x 4): {t_baseline_eval/60:.1f} min")
    print(f"TOTAL (DRL train+eval + baselines eval): {(t_drl+t_drl_eval+t_baseline_eval)/60:.1f} min")
    print(f"TOTAL hours: {(t_drl+t_drl_eval+t_baseline_eval)/3600:.2f} h")


if __name__ == "__main__":
    main()

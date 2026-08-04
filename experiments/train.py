"""Train T2Sem-DRL or run baselines on multi-UAV multi-terminal MEC."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.t2sem_drl import T2SemDRL
from src.agents.warmstart import (
    collect_demonstrations,
    collect_filtered_demonstrations,
)
from src.baselines.greedy import GreedyPolicy
from src.baselines.local_only import LocalOnlyPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.tjcct import TJCCTHeuristicPolicy
from src.env.multi_mec_env import MultiMECEnv
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import EpisodeMetrics, aggregate_metrics


def build_env(cfg: dict, variant: str = "full") -> MultiMECEnv:
    kwargs = {
        "disable_hap": False,
        "fixed_mu": None,
        "sla_penalty_enabled": True,
        "resource_contention": cfg.get("resource_contention", True),
    }
    if variant == "uav_only":
        kwargs["disable_hap"] = True
    elif variant == "no_semantic":
        kwargs["fixed_mu"] = 1.0
    elif variant == "fixed_mu":
        kwargs["fixed_mu"] = 0.5
    elif variant == "no_sla_penalty":
        kwargs["sla_penalty_enabled"] = False
    elif variant == "no_contention":
        kwargs["resource_contention"] = False
    return MultiMECEnv(config=cfg, **kwargs)


def use_long_agent_for_variant(variant: str) -> bool:
    return variant != "single_timescale"


def _init_train_log(log_path: Path, tier_names: list[str], per_tier: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    cols = [
        "episode",
        "sla_overall",
        "avg_delay",
        "total_energy",
        "feasible_targets",
    ]
    for name in tier_names:
        cols.append(f"sla_{name}")
        cols.append(f"viol_{name}")
    if per_tier:
        for name in tier_names:
            cols.append(f"lambda_{name}")
    else:
        cols.append("lambda_global")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()


def _append_train_log(
    log_path: Path,
    episode: int,
    m,
    env: MultiMECEnv,
    *,
    per_tier: bool,
    lam_by_tier: dict[str, float],
    lam_global: float,
    targets_by_tier: dict,
    sla_target: float,
) -> None:
    rates = env.tier_sla_rates()
    row: dict = {
        "episode": episode,
        "sla_overall": f"{m.sla_rate:.6f}",
        "avg_delay": f"{getattr(m, 'avg_delay', 0.0):.6f}",
        "total_energy": f"{getattr(m, 'total_energy', 0.0):.2f}",
        "feasible_targets": env.feasible_target_count(),
    }
    tier_names = [t.name for t in env.sla_tiers]
    for name in tier_names:
        rate = rates.get(name, 0.0)
        tgt = float(targets_by_tier.get(name, sla_target))
        row[f"sla_{name}"] = f"{rate:.6f}"
        row[f"viol_{name}"] = f"{max(0.0, tgt - rate):.6f}"
    if per_tier:
        for name in tier_names:
            row[f"lambda_{name}"] = f"{lam_by_tier.get(name, 0.0):.6f}"
    else:
        row[f"lambda_global"] = f"{lam_global:.6f}"
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writerow(row)


def train_t2sem(
    cfg: dict,
    variant: str,
    seed: int,
    device: str,
    *,
    skip_bc: bool = False,
) -> str:
    cfg = dict(cfg)
    cfg["seed"] = seed
    drl = cfg["drl"]
    exp = cfg["experiment"]
    episodes = int(drl["total_long_frames"])
    env = build_env(cfg, variant)
    agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant(variant))

    if not skip_bc:
        _warm_start(agent, env, drl, seed)

    ckpt_dir = project_root() / exp["checkpoint_dir"] / variant / f"seed_{seed}"

    # Robust checkpoint selection. "train_last" reproduces the legacy noisy
    # single-episode behaviour; "moving_avg" smooths over a window; "eval"
    # periodically runs deterministic eval episodes (most reliable).
    metric_mode = str(exp.get("checkpoint_metric", "train_last"))
    eval_every = int(exp.get("ckpt_eval_every", 20))
    ckpt_eval_eps = int(exp.get("ckpt_eval_episodes", 3))
    window = int(exp.get("ckpt_ma_window", 10))

    # Lagrangian constrained RL: minimize energy subject to SLA >= target.
    # lambda is raised when SLA is below target and lowered (freeing the policy
    # to cut energy) once it is met. Disabled unless reward.lagrangian_enabled.
    rwd = cfg.get("reward", {})
    lagr_enabled = bool(rwd.get("lagrangian_enabled", False))
    per_tier = bool(rwd.get("per_tier_lambda", False))
    sla_target = float(rwd.get("sla_target", 0.6))
    targets_by_tier = dict(rwd.get("sla_target_by_tier", {}) or {})
    lam = float(rwd.get("sla_lambda_init", 1.0))
    lam_lr = float(rwd.get("lambda_lr", 0.05))
    lam_min = float(rwd.get("lambda_min", 0.1))
    lam_max = float(rwd.get("lambda_max", 10.0))
    lam_ema = float(rwd.get("lambda_ema", 0.3))
    deficit_clip = float(rwd.get("lambda_deficit_clip", 0.2))
    lam_by_tier = {name: lam for name in env.sla_lambda_by_tier}
    tier_names = [t.name for t in env.sla_tiers]
    log_dir = project_root() / exp.get("log_dir", "results/logs")
    log_path = log_dir / variant / f"seed_{seed}" / "train_metrics.csv"
    _init_train_log(log_path, tier_names, per_tier)
    if lagr_enabled:
        env.set_sla_lambda(lam_by_tier if per_tier else lam)

    best_score = -1.0
    saved_any = False
    recent: deque = deque(maxlen=window)

    for ep in tqdm(range(episodes), desc=f"train-{variant}-s{seed}", leave=False):
        m = agent.train_episode()
        recent.append(m.sla_rate)

        if lagr_enabled:
            if per_tier:
                rates = env.tier_sla_rates()
                for name in lam_by_tier:
                    tgt = float(targets_by_tier.get(name, sla_target))
                    deficit = float(np.clip(tgt - rates.get(name, 1.0), -0.05, deficit_clip))
                    proposed = lam_by_tier[name] + lam_lr * deficit
                    lam_by_tier[name] = float(
                        np.clip((1 - lam_ema) * lam_by_tier[name] + lam_ema * proposed, lam_min, lam_max)
                    )
                env.set_sla_lambda(lam_by_tier)
            else:
                deficit = float(np.clip(sla_target - m.sla_rate, -0.05, deficit_clip))
                proposed = lam + lam_lr * deficit
                lam = float(np.clip((1 - lam_ema) * lam + lam_ema * proposed, lam_min, lam_max))
                env.set_sla_lambda(lam)

        _append_train_log(
            log_path,
            ep + 1,
            m,
            env,
            per_tier=per_tier,
            lam_by_tier=lam_by_tier,
            lam_global=lam,
            targets_by_tier=targets_by_tier,
            sla_target=sla_target,
        )

        improved = False
        if metric_mode == "moving_avg":
            score = sum(recent) / len(recent)
            improved = score >= best_score
        elif metric_mode == "eval":
            if (ep + 1) % eval_every == 0 or ep == episodes - 1:
                evals = [
                    agent.evaluate_episode(seed=seed * 7919 + k)
                    for k in range(ckpt_eval_eps)
                ]
                score = float(np.mean([e.sla_rate for e in evals]))
                improved = score > best_score
            else:
                score = best_score
        else:  # train_last (legacy)
            score = m.sla_rate
            improved = score >= best_score

        if improved:
            best_score = score
            agent.save(str(ckpt_dir))
            saved_any = True

    if not saved_any:
        agent.save(str(ckpt_dir))
    (ckpt_dir / ".train_complete").write_text(
        f"episodes={episodes}\nbest_score={best_score}\nmetric={metric_mode}\n",
        encoding="utf-8",
    )
    return str(ckpt_dir)


def _warm_start(agent: T2SemDRL, env: MultiMECEnv, drl: dict, seed: int) -> None:
    """Behavioral-cloning warm-start with optional SLA-success filtering."""
    bc_source = str(drl.get("bc_source", "greedy"))
    demo_eps = int(drl["bc_demo_episodes"])
    epochs = int(drl["bc_train_epochs"])

    if bc_source == "filtered":
        demos = collect_filtered_demonstrations(
            env, demo_eps, seed=seed, min_samples=int(drl.get("bc_min_samples", 128))
        )
        if demos is not None:
            local_obs, per_term_acts, term_idx = demos
            loss = agent.short.pretrain_behavioral(
                None, per_term_acts, local_obs, terminal_indices=term_idx, epochs=epochs
            )
            print(f"  BC(filtered, n={local_obs.shape[0]}) loss={loss:.4f}")
            return
        print("  BC(filtered) found too few SLA-met samples; falling back to greedy")

    g_obs, acts, l_obs = collect_demonstrations(env, demo_eps, seed=seed)
    loss = agent.short.pretrain_behavioral(g_obs, acts, l_obs, epochs=epochs)
    print(f"  BC(greedy) loss={loss:.4f}")


def run_baseline(env: MultiMECEnv, name: str, episodes: int) -> list[EpisodeMetrics]:
    if name == "greedy":
        policy = GreedyPolicy(env)
    elif name == "local_only":
        policy = LocalOnlyPolicy(env)
    elif name == "random":
        policy = RandomPolicy(env, seed=int(env.cfg.get("seed", 42)))
    elif name == "tjcct":
        policy = TJCCTHeuristicPolicy(env)
    else:
        raise ValueError(name)

    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(env.cfg.get("seed", 42)) + ep)
        done = False
        rewards = []
        m = EpisodeMetrics()
        while not done:
            action = policy.act(obs)
            obs, r, term, trunc, info = env.step(action)
            rewards.append(r)
            for sr in info.get("step_results", []):
                m.tasks_total += 1
                if sr.sla_met:
                    m.tasks_sla_met += 1
                m.total_energy += sr.energy
            done = term or trunc
        m.avg_reward = float(np.mean(rewards)) if rewards else 0.0
        m.sla_rate = m.tasks_sla_met / m.tasks_total if m.tasks_total else 0.0
        em = env.episode_metrics
        if em.get("tasks_total", 0) > 0:
            m.avg_delay = em["total_delay"] / em["tasks_total"]
            m.total_energy = em["total_energy"]
        results.append(m)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", default="t2sem_drl", choices=["t2sem_drl", "greedy", "local_only", "random", "tjcct"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-bc", action="store_true")
    args = parser.parse_args()

    paths = ["configs/default.yaml"]
    if args.smoke:
        paths.append("configs/smoke.yaml")
    elif args.config != "configs/default.yaml":
        paths.append(args.config)
    cfg = load_config(*paths)

    device = device_from_config(args.device, cfg)
    train_eps = args.episodes or int(cfg["drl"]["total_long_frames"])

    if args.method != "t2sem_drl":
        env = build_env(cfg)
        results = run_baseline(env, args.method, train_eps)
        agg = aggregate_metrics(results)
        print(f"Baseline {args.method}: SLA={agg['sla_rate_mean']:.3f} delay={agg['avg_delay_mean']:.3f}")
        return

    env = build_env(cfg)
    agent = T2SemDRL(env, device=device, use_long_agent=True)

    if not args.no_bc:
        _warm_start(agent, env, cfg["drl"], int(cfg.get("seed", 42)))

    history = []
    for ep in tqdm(range(train_eps), desc="train"):
        m = agent.train_episode()
        history.append(m.sla_rate)
        if (ep + 1) % max(train_eps // 5, 1) == 0:
            print(f"  ep {ep+1}: SLA={m.sla_rate:.3f} delay={m.avg_delay:.3f}")

    ckpt = project_root() / cfg["experiment"]["checkpoint_dir"]
    agent.save(str(ckpt))
    print(f"Saved checkpoint to {ckpt}")
    print(f"Final SLA rate (last episode): {history[-1]:.3f}")


if __name__ == "__main__":
    main()

"""Full experiment: main comparison + ablation suite + summary CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import (
    build_env,
    run_baseline,
    train_t2sem,
    use_long_agent_for_variant,
)
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

VARIANTS = {
    "A1_full": "full",
    "A2_no_semantic": "no_semantic",
    "A3_single_timescale": "single_timescale",
    "A4_no_sla_penalty": "no_sla_penalty",
    "A5_uav_only": "uav_only",
    "A6_fixed_mu": "fixed_mu",
    "A7_no_contention": "no_contention",
}

BASELINES = ["greedy", "tjcct", "random", "local_only"]


def evaluate_trained(
    cfg: dict,
    variant: str,
    seed: int,
    ckpt_dir: str,
    eval_episodes: int,
    device: str,
) -> list:
    cfg = dict(cfg)
    cfg["seed"] = seed
    env = build_env(cfg, variant)
    agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant(variant))
    short_pt = Path(ckpt_dir) / "short_ppo.pt"
    if short_pt.exists():
        agent.load(ckpt_dir)
    return [agent.evaluate_episode(seed=seed * 1000 + ep) for ep in range(eval_episodes)]


def evaluate_baseline_all_seeds(cfg: dict, name: str, eval_eps: int, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        env = build_env(cfg)
        metrics = []
        for ep in range(eval_eps):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            from src.utils.metrics import EpisodeMetrics

            m = EpisodeMetrics()
            if name == "greedy":
                from src.baselines.greedy import GreedyPolicy

                policy = GreedyPolicy(env)
            elif name == "local_only":
                from src.baselines.local_only import LocalOnlyPolicy

                policy = LocalOnlyPolicy(env)
            elif name == "random":
                from src.baselines.random_policy import RandomPolicy

                policy = RandomPolicy(env, seed=seed)
            elif name == "tjcct":
                from src.baselines.tjcct import TJCCTHeuristicPolicy

                policy = TJCCTHeuristicPolicy(env)
            else:
                raise ValueError(name)
            while not done:
                obs, _, term, trunc, info = env.step(policy.act(obs))
                for sr in info.get("step_results", []):
                    m.tasks_total += 1
                    if sr.sla_met:
                        m.tasks_sla_met += 1
                    m.total_energy += sr.energy
                done = term or trunc
            m.sla_rate = m.tasks_sla_met / m.tasks_total if m.tasks_total else 0.0
            em = env.episode_metrics
            if em.get("tasks_total", 0) > 0:
                m.avg_delay = em["total_delay"] / em["tasks_total"]
            metrics.append(m)
        agg = aggregate_metrics(metrics)
        rows.append(
            {
                "method": name,
                "variant": "baseline",
                "seed": seed,
                **{f"{k}": v for k, v in agg.items()},
            }
        )
        print(f"  baseline {name} seed={seed} SLA={agg['sla_rate_mean']:.3f}")
    return rows


def is_training_complete(cfg: dict, variant: str, seed: int) -> bool:
    marker = (
        project_root()
        / cfg["experiment"]["checkpoint_dir"]
        / variant
        / f"seed_{seed}"
        / ".train_complete"
    )
    return marker.exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip training runs already marked complete")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = device_from_config(args.device, cfg)
    print(f"Device: {device}")

    paths = ["configs/default.yaml"]
    if args.smoke:
        paths.append("configs/smoke.yaml")
    elif args.config != "configs/default.yaml":
        paths.append(args.config)
    cfg = load_config(*paths)

    seeds = list(cfg["experiment"]["eval_seeds"])
    eval_eps = int(cfg["experiment"]["eval_episodes"])
    out_dir = project_root() / cfg["experiment"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    variant_items = list(VARIANTS.items())
    if args.variants:
        variant_items = [(k, v) for k, v in VARIANTS.items() if k in args.variants or v in args.variants]

    # ── Phase 1: train all variants ───────────────────────────────────────────
    if not args.skip_train:
        print("\n=== Phase 1: Training ===")
        for name, variant in variant_items:
            for seed in seeds:
                if args.resume and is_training_complete(cfg, variant, seed):
                    print(f"Skip (done): {name} seed={seed}")
                    continue
                print(f"Training {name} seed={seed}...")
                train_t2sem(cfg, variant, seed, device)

    # ── Phase 2: evaluate trained variants ────────────────────────────────────
    print("\n=== Phase 2: Evaluate T2Sem-DRL variants ===")
    for name, variant in variant_items:
        for seed in seeds:
            ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
            metrics = evaluate_trained(cfg, variant, seed, str(ckpt), eval_eps, device)
            agg = aggregate_metrics(metrics)
            rows.append(
                {
                    "method": "T2Sem-DRL",
                    "variant": name,
                    "seed": seed,
                    **agg,
                }
            )
            print(f"  {name} seed={seed} SLA={agg['sla_rate_mean']:.3f} delay={agg['avg_delay_mean']:.3f}")

    # ── Phase 3: baselines ──────────────────────────────────────────────────────
    if not args.skip_baselines:
        print("\n=== Phase 3: Baselines ===")
        for bl in BASELINES:
            rows.extend(evaluate_baseline_all_seeds(cfg, bl, eval_eps, seeds))

    # ── Save summary ────────────────────────────────────────────────────────────
    summary_path = out_dir / "experiment_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {summary_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print("DONE")


if __name__ == "__main__":
    main()

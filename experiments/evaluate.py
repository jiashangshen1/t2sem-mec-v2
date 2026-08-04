"""Evaluate trained T2Sem-DRL or baselines across seeds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, run_baseline
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", default="t2sem_drl")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    paths = ["configs/default.yaml"]
    if args.smoke:
        paths.append("configs/smoke.yaml")
    cfg = load_config(*paths)
    device = device_from_config("auto", cfg)
    eval_eps = int(cfg["experiment"]["eval_episodes"])
    seeds = list(cfg["experiment"]["eval_seeds"])

    rows = []
    for seed in seeds:
        env = build_env(cfg)
        if args.method == "t2sem_drl":
            agent = T2SemDRL(env, device=device)
            ckpt = args.checkpoint or str(project_root() / cfg["experiment"]["checkpoint_dir"])
            try:
                agent.load(ckpt)
            except FileNotFoundError:
                print(f"No checkpoint at {ckpt}, evaluating untrained agent")
            metrics = [agent.evaluate_episode(seed=seed + ep) for ep in range(eval_eps)]
        else:
            metrics = run_baseline(env, args.method, eval_eps)
        agg = aggregate_metrics(metrics)
        rows.append(
            {
                "method": args.method,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
            }
        )
        print(f"seed={seed} SLA={agg['sla_rate_mean']:.3f} delay={agg['avg_delay_mean']:.3f}")

    out_dir = project_root() / cfg["experiment"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{args.method}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""Validation-set alpha selection for feasibility mask (M7).

Pick best delay margin alpha on validation seeds {42, 123}, report on holdout seed 456.
Uses trained t2sem_full checkpoints; no retraining.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, use_long_agent_for_variant
from scripts.mask_sensitivity import DEFAULT_CHANNEL, DEFAULT_SIM, _eval, _set_mask
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config

ALPHAS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved_v4_pertier.yaml")
    parser.add_argument("--val-seeds", type=int, nargs="*", default=[42, 123])
    parser.add_argument("--test-seed", type=int, default=456)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml", args.config)
    device = device_from_config(args.device, cfg)
    eval_eps = int(args.eval_episodes)
    tier_names = [t.name for t in build_env(cfg).sla_tiers]
    out_dir = project_root() / "results" / "supplement_v43"
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_agents(seeds: list[int]):
        agents = []
        for seed in seeds:
            c = dict(cfg)
            c["seed"] = seed
            env = build_env(c, "full")
            agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
            ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
            agent.load(str(ckpt))
            agents.append((env, agent, seed))
        return agents

    val_agents = load_agents(list(args.val_seeds))
    val_rows = []
    for alpha in ALPHAS:
        r = _eval(val_agents, alpha, DEFAULT_CHANNEL, DEFAULT_SIM, eval_eps, tier_names)
        val_rows.append({"alpha": alpha, "split": "validation", **r})
        print(f"[val] alpha={alpha} SLA={r['overall_sla']:.4f}")

    best = max(val_rows, key=lambda x: x["overall_sla"])
    best_alpha = float(best["alpha"])
    print(f"\nBest validation alpha={best_alpha} SLA={best['overall_sla']:.4f}")

    test_agents = load_agents([args.test_seed])
    default_r = _eval(test_agents, 0.8, DEFAULT_CHANNEL, DEFAULT_SIM, eval_eps, tier_names)
    tuned_r = _eval(test_agents, best_alpha, DEFAULT_CHANNEL, DEFAULT_SIM, eval_eps, tier_names)

    summary = [
        {
            "split": "test",
            "seed": args.test_seed,
            "alpha": 0.8,
            "overall_sla": default_r["overall_sla"],
            "energy": default_r["energy"],
            "label": "default_train_alpha",
        },
        {
            "split": "test",
            "seed": args.test_seed,
            "alpha": best_alpha,
            "overall_sla": tuned_r["overall_sla"],
            "energy": tuned_r["energy"],
            "label": "validation_selected_alpha",
        },
    ]
    for n in tier_names:
        summary[0][n] = default_r[n]
        summary[1][n] = tuned_r[n]

    val_csv = out_dir / "alpha_validation_sweep.csv"
    with val_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        w.writeheader()
        w.writerows(val_rows)

    sum_csv = out_dir / "alpha_validation_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote {val_csv} and {sum_csv}")
    print(
        f"Test seed {args.test_seed}: default alpha=0.8 SLA={default_r['overall_sla']:.4f} "
        f"vs selected alpha={best_alpha} SLA={tuned_r['overall_sla']:.4f}"
    )


if __name__ == "__main__":
    main()

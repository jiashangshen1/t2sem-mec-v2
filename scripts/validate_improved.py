"""Quick A/B validation of the improved T2Sem-DRL architecture/training.

Trains the improved-config agent for a small number of episodes on a couple of
seeds, evaluates it deterministically, and compares against the random and
greedy baselines on the same environment. The goal is a fast sanity check that
the improvements lift SLA toward / past the random baseline before committing
to a long multi-round run.

Usage:
    python scripts/validate_improved.py --episodes 80 --seeds 42 123
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import evaluate_baseline_all_seeds, evaluate_trained
from experiments.train import train_t2sem
from src.utils.config import load_config
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123])
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config", default="configs/improved_v2.yaml", help="improved config layer"
    )
    args = parser.parse_args()

    device = device_from_config(args.device, cfg)
    cfg = load_config("configs/default.yaml", args.config)

    # Shrink for a fast validation run.
    cfg["drl"]["total_long_frames"] = int(args.episodes)
    cfg["experiment"]["eval_seeds"] = list(args.seeds)
    cfg["experiment"]["eval_episodes"] = int(args.eval_episodes)
    cfg["experiment"]["checkpoint_dir"] = "results/checkpoints_validate"
    cfg["experiment"]["results_dir"] = "results/validate"

    print(f"Device: {device}")
    print(
        f"Validation: {args.episodes} train-ep x {len(args.seeds)} seeds "
        f"| reward shaping={cfg['reward'].get('shaping_enabled')} "
        f"| bc={cfg['drl'].get('bc_source')} "
        f"| entropy={cfg['drl']['short'].get('entropy_coef')} "
        f"| ckpt={cfg['experiment'].get('checkpoint_metric')}"
    )

    t0 = time.perf_counter()
    variant = "full"

    drl_rows = []
    for seed in args.seeds:
        print(f"\n--- Training improved T2Sem-DRL seed={seed} ---")
        ckpt = train_t2sem(cfg, variant, seed, device)
        metrics = evaluate_trained(
            cfg, variant, seed, ckpt, int(args.eval_episodes), device
        )
        agg = aggregate_metrics(metrics)
        drl_rows.append((seed, agg))
        print(
            f"  [eval] seed={seed} SLA={agg['sla_rate_mean']:.3f} "
            f"delay={agg['avg_delay_mean']:.3f} "
            f"energy={agg['total_energy_mean']:.0f}"
        )

    print("\n--- Baselines (random, greedy) ---")
    rnd_rows = evaluate_baseline_all_seeds(
        cfg, "random", int(args.eval_episodes), list(args.seeds)
    )
    grd_rows = evaluate_baseline_all_seeds(
        cfg, "greedy", int(args.eval_episodes), list(args.seeds)
    )

    drl_sla = float(np.mean([a["sla_rate_mean"] for _, a in drl_rows]))
    rnd_sla = float(np.mean([r["sla_rate_mean"] for r in rnd_rows]))
    grd_sla = float(np.mean([r["sla_rate_mean"] for r in grd_rows]))
    drl_e = float(np.mean([a["total_energy_mean"] for _, a in drl_rows]))
    rnd_e = float(np.mean([r["total_energy_mean"] for r in rnd_rows]))

    print("\n==================== VALIDATION SUMMARY ====================")
    print(f"{'method':<22}{'SLA(mean)':>12}{'energy':>14}{'per-seed SLA':>22}")
    drl_per = ", ".join(f"{a['sla_rate_mean']:.3f}" for _, a in drl_rows)
    print(f"{'Improved T2Sem-DRL':<22}{drl_sla:>12.3f}{drl_e:>14.0f}   [{drl_per}]")
    print(f"{'random':<22}{rnd_sla:>12.3f}{rnd_e:>14.0f}")
    print(f"{'greedy':<22}{grd_sla:>12.3f}")
    print("-" * 70)
    verdict = "PASS (beats random)" if drl_sla >= rnd_sla else "BELOW random"
    print(f"Verdict: improved DRL {drl_sla:.3f} vs random {rnd_sla:.3f} -> {verdict}")
    print(f"Elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()

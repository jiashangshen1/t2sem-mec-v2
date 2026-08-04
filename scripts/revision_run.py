"""Revision v4.3 experiment suite: baselines + key DRL variants + CSV export."""

from __future__ import annotations

import argparse
import csv
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, run_baseline, train_t2sem, use_long_agent_for_variant
from experiments.run_all import evaluate_trained, evaluate_baseline_all_seeds
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

SEEDS = [42, 123, 456]

DRL_JOBS = {
    "t2sem_full": ("full", "configs/improved_v4_pertier.yaml"),
    "t2sem_single_lambda": ("full", "configs/improved_v4_pertier.yaml", {"reward.per_tier_lambda": False}),
    "t2sem_single_ts": ("single_timescale", "configs/improved_v4_pertier.yaml"),
    "independent_ppo": ("full", "configs/improved_v4_independent_ppo.yaml"),
    "mobility_full": ("full", "configs/mobility_critical.yaml"),
    "mobility_single_ts": ("single_timescale", "configs/mobility_critical.yaml"),
}

BASELINES = ["greedy", "tjcct", "random", "local_only"]


def _apply_overrides(cfg: dict, overrides: dict | None) -> dict:
    if not overrides:
        return cfg
    cfg = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        parts = dotted.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip train if .train_complete exists")
    parser.add_argument("--only", default=None, help="comma-separated job names")
    args = parser.parse_args()

    paths = ["configs/default.yaml"]
    if args.smoke:
        paths.append("configs/smoke.yaml")
    device = args.device
    out_dir = project_root() / "results" / ("revision_v43_smoke" if args.smoke else "revision_v43")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_eps = 5 if args.smoke else 30
    episodes_override = 15 if args.smoke else None

    rows: list[dict] = []
    t0 = time.time()

    # --- baselines on main config ---
    cfg_main = load_config("configs/default.yaml", "configs/improved_v4_pertier.yaml")
    if args.smoke:
        cfg_main = load_config("configs/default.yaml", "configs/smoke.yaml", "configs/improved_v4_pertier.yaml")
    for bl in BASELINES:
        if args.only and bl not in args.only.split(","):
            continue
        for seed in ([42] if args.smoke else SEEDS):
            env = build_env(cfg_main)
            env.cfg["seed"] = seed
            metrics = run_baseline(env, bl, eval_eps)
            agg = aggregate_metrics(metrics)
            rows.append(
                {
                    "job": bl,
                    "variant": "baseline",
                    "seed": seed,
                    "sla_rate_mean": agg["sla_rate_mean"],
                    "sla_rate_std": agg["sla_rate_std"],
                    "avg_delay_mean": agg["avg_delay_mean"],
                    "total_energy_mean": agg["total_energy_mean"],
                }
            )
            print(f"[baseline] {bl} seed={seed} SLA={agg['sla_rate_mean']:.3f}")

    # --- DRL jobs ---
    for job_name, spec in DRL_JOBS.items():
        if args.only and job_name not in args.only.split(","):
            continue
        variant = spec[0]
        cfg_paths = ["configs/default.yaml"]
        if args.smoke:
            cfg_paths.append("configs/smoke.yaml")
        cfg_paths.append(spec[1])
        cfg = load_config(*cfg_paths)
        overrides = spec[2] if len(spec) > 2 else None
        cfg = _apply_overrides(cfg, overrides)
        if episodes_override:
            cfg["drl"]["total_long_frames"] = episodes_override
        device_resolved = device_from_config(device, cfg)
        seeds = [42] if args.smoke else SEEDS
        exp = cfg["experiment"]
        ckpt_root = project_root() / exp["checkpoint_dir"] / variant

        if not args.skip_train:
            for seed in seeds:
                ckpt = ckpt_root / f"seed_{seed}"
                if args.resume and (ckpt / ".train_complete").is_file():
                    print(f"[skip-train] {job_name} seed={seed} (complete)")
                    continue
                print(f"[train] {job_name} seed={seed}")
                train_t2sem(cfg, variant, seed, device_resolved)

        for seed in seeds:
            ckpt = ckpt_root / f"seed_{seed}"
            metrics = evaluate_trained(cfg, variant, seed, str(ckpt), eval_eps, device_resolved)
            agg = aggregate_metrics(metrics)
            rows.append(
                {
                    "job": job_name,
                    "variant": variant,
                    "seed": seed,
                    "sla_rate_mean": agg["sla_rate_mean"],
                    "sla_rate_std": agg["sla_rate_std"],
                    "avg_delay_mean": agg["avg_delay_mean"],
                    "total_energy_mean": agg["total_energy_mean"],
                }
            )
            print(f"[eval] {job_name} seed={seed} SLA={agg['sla_rate_mean']:.3f}")

    csv_path = out_dir / "revision_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

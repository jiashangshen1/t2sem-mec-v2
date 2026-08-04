"""Run ideal-track T2Sem-X prototype training/evaluation.

This script is intentionally isolated from revision/supplement pipelines and
writes only to results/ideal_track/*.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env
from src.agents.t2sem_x import T2SemXDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

VARIANTS = {
    "full": {
        "job": "ideal_t2sem_x",
        "configs": ["configs/ideal_v1_t2semx.yaml"],
    },
    "no_robust": {
        "job": "ideal_t2sem_x_no_robust",
        "configs": ["configs/ideal_v1_t2semx.yaml", "configs/ideal_v1_t2semx_no_robust.yaml"],
    },
    "no_credit": {
        "job": "ideal_t2sem_x_no_credit",
        "configs": ["configs/ideal_v1_t2semx.yaml", "configs/ideal_v1_t2semx_no_credit.yaml"],
    },
}


def load_variant_config(variant: str, smoke: bool) -> dict:
    paths = ["configs/default.yaml", "configs/ideal_v1_base.yaml"]
    paths.extend(VARIANTS[variant]["configs"])
    if smoke:
        paths.append("configs/smoke.yaml")
    return load_config(*paths)


def write_rows(csv_path: Path, rows: list[dict], append: bool) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if append and csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        merged = existing + rows
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(merged)
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def run_variant(
    variant: str,
    *,
    device: str,
    seeds: list[int],
    train_eps: int | None,
    eval_eps: int,
    eval_only: bool,
    smoke: bool,
) -> list[dict]:
    cfg = load_variant_config(variant, smoke=smoke)
    device = device_from_config(device, cfg)
    train_eps = train_eps or int(cfg["drl"]["total_long_frames"])
    job = VARIANTS[variant]["job"]
    rows = []

    for seed in seeds:
        c = dict(cfg)
        c["seed"] = int(seed)
        env = build_env(c, "full")
        agent = T2SemXDRL(env=env, device=device, use_long_agent=True)
        ckpt = project_root() / c["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
        if eval_only and (ckpt / "short_ppo.pt").is_file():
            agent.load(str(ckpt))
        else:
            for _ in range(train_eps):
                agent.train_episode()
            ckpt.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt))
            (ckpt / ".train_complete").write_text(f"{job}\n", encoding="utf-8")

        metrics = [agent.evaluate_episode(seed=seed * 1000 + ep) for ep in range(eval_eps)]
        agg = aggregate_metrics(metrics)
        rows.append(
            {
                "job": job,
                "variant": variant,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
                "note": "method_ablation",
            }
        )
        print(f"[{job}] seed={seed} SLA={agg['sla_rate_mean']:.3f} delay={agg['avg_delay_mean']:.3f}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--variant", default="full", choices=list(VARIANTS.keys()) + ["all"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    seeds = [42] if args.smoke else args.seeds
    eval_eps = 5 if args.smoke else (args.eval_episodes or 30)
    train_eps = 15 if args.smoke else args.train_episodes
    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    out_dir = project_root() / "results" / "ideal_track"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ideal_summary.csv"

    all_rows: list[dict] = []
    for i, variant in enumerate(variants):
        rows = run_variant(
            variant,
            device=args.device,
            seeds=seeds,
            train_eps=train_eps,
            eval_eps=eval_eps,
            eval_only=args.eval_only,
            smoke=args.smoke,
        )
        all_rows.extend(rows)
        write_rows(csv_path, rows, append=args.append or i > 0)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

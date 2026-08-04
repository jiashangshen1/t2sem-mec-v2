"""Stress suite for theory-boundary checks in the ideal track."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env
from src.agents.t2sem_drl import T2SemDRL
from src.agents.t2sem_x import T2SemXDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

SCENARIOS = {
    "nlos_mismatch": ["configs/ideal_stress_nlos.yaml"],
    "tier_conflict": ["configs/ideal_stress_tier_conflict.yaml"],
    "extreme_load": ["configs/ideal_stress_extreme_load.yaml"],
    "mask_collapse": ["configs/ideal_stress_nlos.yaml", "configs/ideal_stress_tier_conflict.yaml"],
}


def make_agent(method: str, env, device: str):
    if method == "ideal_t2sem_x":
        return T2SemXDRL(env, device=device, use_long_agent=True)
    return T2SemDRL(env, device=device, use_long_agent=True)


def method_base_config(method: str) -> list[str]:
    if method == "ideal_t2sem_x":
        return ["configs/ideal_v1_t2semx.yaml"]
    return ["configs/improved_v4_pertier.yaml"]


def default_ckpt(method: str) -> str:
    root = project_root()
    if method == "ideal_t2sem_x":
        return str(root / "results/ideal_track/checkpoints_t2semx/full/seed_42")
    return str(root / "results/checkpoints_improved_pertier/full/seed_42")


def bootstrap_train(agent, train_episodes: int) -> None:
    for _ in range(train_episodes):
        agent.train_episode()


def run_scenario(
    method: str,
    scenario: str,
    seeds: list[int],
    eval_eps: int,
    device: str,
    bootstrap_train_eps: int,
) -> dict:
    cfg_paths = ["configs/default.yaml", "configs/ideal_v1_base.yaml"]
    cfg_paths.extend(method_base_config(method))
    cfg_paths.extend(SCENARIOS[scenario])
    cfg = load_config(*cfg_paths)
    rows = []
    used = 0
    fallback_unloaded = 0
    bootstrapped = 0
    for seed in seeds:
        c = dict(cfg)
        c["seed"] = int(seed)
        env = build_env(c, "full")
        agent = make_agent(method, env, device)
        ckpt = Path(default_ckpt(method)).parent / f"seed_{seed}"
        if (ckpt / "short_ppo.pt").is_file():
            try:
                agent.load(str(ckpt))
            except RuntimeError as e:
                fallback_unloaded += 1
                print(f"[warn] load failed for {method} seed={seed} scenario={scenario}: {e}")
                print(traceback.format_exc(limit=1))
        elif bootstrap_train_eps > 0:
            bootstrap_train(agent, bootstrap_train_eps)
            bootstrapped += 1
        metrics = [agent.evaluate_episode(seed=seed * 1000 + ep) for ep in range(eval_eps)]
        agg = aggregate_metrics(metrics)
        rows.append(agg)
        used += 1
    if used == 0:
        raise RuntimeError(f"no usable seeds for scenario={scenario}")
    mean = {
        "sla_rate_mean": sum(r["sla_rate_mean"] for r in rows) / len(rows),
        "avg_delay_mean": sum(r["avg_delay_mean"] for r in rows) / len(rows),
        "total_energy_mean": sum(r["total_energy_mean"] for r in rows) / len(rows),
    }
    return {
        "job": f"ideal_stress_{scenario}",
        "method": method,
        "scenario": scenario,
        "sla_rate_mean": mean["sla_rate_mean"],
        "avg_delay_mean": mean["avg_delay_mean"],
        "total_energy_mean": mean["total_energy_mean"],
        "note": (
            f"theory_check;used_seeds={used};fallback_unloaded={fallback_unloaded};"
            f"bootstrapped={bootstrapped}"
        ),
    }


def write_rows(out_path: Path, out_rows: list[dict], append: bool) -> None:
    if not out_rows:
        return
    fieldnames = list(out_rows[0].keys())
    if append and out_path.is_file():
        with out_path.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        merged = existing + out_rows
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(merged)
    else:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="ideal_t2sem_x", choices=["ideal_t2sem_x", "t2sem_full"])
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--bootstrap-train",
        action="store_true",
        help="train briefly when checkpoint is missing (for t2sem_full baseline)",
    )
    parser.add_argument("--bootstrap-episodes", type=int, default=50)
    args = parser.parse_args()

    device = device_from_config(args.device, load_config("configs/default.yaml"))
    out_dir = project_root() / "results" / "ideal_track"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stress_summary.csv"
    bootstrap_eps = args.bootstrap_episodes if args.bootstrap_train else 0
    out_rows = []
    for scenario in SCENARIOS:
        row = run_scenario(
            args.method,
            scenario,
            args.seeds,
            args.eval_episodes,
            device,
            bootstrap_eps,
        )
        out_rows.append(row)
        print(
            f"[{args.method}:{scenario}] SLA={row['sla_rate_mean']:.3f} "
            f"delay={row['avg_delay_mean']:.3f}"
        )
    write_rows(out_path, out_rows, append=args.append)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

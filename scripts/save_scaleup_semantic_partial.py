"""Save partial scaleup_semantic eval results (seeds with .train_complete)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import evaluate_trained
from src.utils.config import load_config, project_root
from src.utils.metrics import aggregate_metrics

CONFIG = "configs/improved_v4_scaleup_semantic.yaml"
SEEDS = [42, 123, 456]
OUT = project_root() / "results/supplement_v43/supplement_summary.csv"


def main() -> None:
    cfg = load_config("configs/default.yaml", CONFIG)
    rows = list(csv.DictReader(OUT.open(encoding="utf-8")))
    merged = {(r["job"], int(r["seed"])): r for r in rows}
    saved = 0
    for seed in SEEDS:
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
        if not (ckpt / ".train_complete").is_file():
            print(f"[skip] seed={seed} (incomplete train)", flush=True)
            continue
        metrics = evaluate_trained(cfg, "full", seed, str(ckpt), 30, "cpu")
        agg = aggregate_metrics(metrics)
        row = {
            "job": "scaleup_semantic_mappo",
            "variant": "full",
            "seed": seed,
            "sla_rate_mean": agg["sla_rate_mean"],
            "sla_rate_std": agg["sla_rate_std"],
            "avg_delay_mean": agg["avg_delay_mean"],
            "total_energy_mean": agg["total_energy_mean"],
            "note": "scaleup_semantic_paused_save",
        }
        merged[(row["job"], int(row["seed"]))] = row
        saved += 1
        print(f"[saved] seed={seed} SLA={agg['sla_rate_mean']:.3f}", flush=True)
    fields = list(rows[0].keys()) if rows else list(row.keys())
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(merged.values(), key=lambda r: (r["job"], int(r["seed"]))))
    print(f"Updated {OUT} (+{saved} scaleup_semantic rows)", flush=True)


if __name__ == "__main__":
    main()

"""Close the scale-up deployment triangle: train + eval Semantic MAPPO + F
on the scale-up topology (16 UE / 32 terminals / 5 UAVs), seeds {42,123,456}.

Writes job=scaleup_semantic_mappo into supplement_summary.csv, then refreshes
bootstrap stats and syncs results to the paper data dir.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import train_t2sem
from experiments.run_all import evaluate_trained
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

CONFIG = "configs/improved_v4_scaleup_semantic.yaml"
SEEDS = [42, 123, 456]
OUT_DIR = project_root() / "results" / "supplement_v43"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _row_key(row: dict) -> tuple[str, int]:
    return (str(row["job"]), int(row["seed"]))


def _load_csv(path: Path) -> dict[tuple[str, int], dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {_row_key(r): r for r in csv.DictReader(f)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _merge_rows(existing: dict[tuple[str, int], dict], new_rows: list[dict]) -> list[dict]:
    merged = dict(existing)
    for row in new_rows:
        merged[_row_key(row)] = row
    return sorted(merged.values(), key=lambda r: (r["job"], int(r["seed"])))


def _ensure_fresh_train(ckpt: Path) -> None:
    if (ckpt / ".train_complete").is_file():
        return
    if ckpt.exists():
        _log(f"[retrain] removing incomplete checkpoint: {ckpt}")
        shutil.rmtree(ckpt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-post", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/default.yaml", CONFIG)
    device = device_from_config(args.device, cfg)
    variant = "full"
    t0 = time.time()
    new_rows: list[dict] = []

    for seed in SEEDS:
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
        if not args.skip_train:
            _ensure_fresh_train(ckpt)
            if not (ckpt / ".train_complete").is_file():
                _log(f"[train] scaleup_semantic seed={seed}")
                train_t2sem(cfg, variant, seed, device)
        metrics = evaluate_trained(cfg, variant, seed, str(ckpt), args.eval_episodes, device)
        agg = aggregate_metrics(metrics)
        new_rows.append(
            {
                "job": "scaleup_semantic_mappo",
                "variant": variant,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
                "note": "scaleup_semantic_matched_training",
            }
        )
        _log(f"[eval] scaleup_semantic seed={seed} SLA={agg['sla_rate_mean']:.3f}")

    csv_path = OUT_DIR / "supplement_summary.csv"
    merged = _merge_rows(_load_csv(csv_path), new_rows)
    _write_csv(csv_path, merged)
    _log(f"Updated {csv_path} (+{len(new_rows)} rows, total {len(merged)})")

    if not args.skip_post:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap_stats.py")], check=False)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_supplement_to_paper.py")], check=False)

    _log(f"SCALEUP_SEMANTIC_DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

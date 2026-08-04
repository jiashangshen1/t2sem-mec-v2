"""Estimate statistical power / CI width for ideal-track comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci_width(values: list[float], n_boot: int = 5000) -> float:
    if len(values) < 2:
        return float("inf")
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(arr[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "results/ideal_track/ideal_summary.csv"))
    parser.add_argument("--metric", default="sla_rate_mean")
    parser.add_argument("--target-width-pp", type=float, default=3.0)
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_job: dict[str, list[float]] = {}
    for r in rows:
        by_job.setdefault(r["job"], []).append(float(r[args.metric]) * 100.0)

    print("# Ideal-track power snapshot")
    for job, vals in sorted(by_job.items()):
        vals = list(vals)
        width = bootstrap_ci_width(vals)
        print(f"{job}: n={len(vals)} CI_width={width:.2f} pp")
        if len(vals) >= 2:
            std = float(np.std(vals, ddof=1))
            n_req = int(np.ceil((1.96 * std / max(args.target_width_pp / 2.0, 1e-6)) ** 2))
            print(f"  estimated seeds for <= {args.target_width_pp:.2f} pp width: {max(n_req, 2)}")


if __name__ == "__main__":
    main()

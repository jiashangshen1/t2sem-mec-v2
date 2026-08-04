"""Build a lightweight artifact list for ideal-track submission package."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ideal_track" / "artifact_manifest.csv"

FILES = [
    "ideal_track/BASELINE_FREEZE_V43.md",
    "ideal_track/THEORY_NOTE_V1.md",
    "ideal_track/THEOREM_EXPERIMENT_MAP.md",
    "ideal_track/POWER_AND_BASELINE_PLAN.md",
    "configs/ideal_v1_base.yaml",
    "configs/ideal_v1_t2semx.yaml",
    "configs/ideal_power_eval.yaml",
    "configs/ideal_stress_nlos.yaml",
    "configs/ideal_stress_tier_conflict.yaml",
    "configs/ideal_stress_extreme_load.yaml",
    "scripts/ideal_t2semx_run.py",
    "scripts/ideal_stress_suite.py",
    "scripts/ideal_seed_power.py",
    "scripts/ideal_baseline_repro.py",
    "tests/test_ideal_t2semx.py",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in FILES:
        p = ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": str(p.exists()).lower(),
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
        )
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

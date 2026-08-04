"""Copy supplement experiment CSVs to the paper package."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(r"C:\Users\12293\Desktop\T2Sem-DRL-v4-paper\data")
SRC = ROOT / "results" / "supplement_v43"

FILES = [
    "supplement_summary.csv",
    "alpha_validation_sweep.csv",
    "alpha_validation_summary.csv",
    "stats_summary.txt",
    "per_tier_critical_priority.csv",
    "load_sweep.csv",
    "load_sweep_summary.csv",
    "mask_sensitivity_critical_priority.csv",
    "mask_sensitivity_scaleup.csv",
]


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, PAPER / name)
            print(f"Copied {name}")
        else:
            print(f"Skip (missing): {name}")


if __name__ == "__main__":
    main()

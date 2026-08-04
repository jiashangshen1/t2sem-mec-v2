"""Round-2: train A1 with optimized config, then evaluate vs baselines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    py = sys.executable
    cfg = "configs/optimized_round2.yaml"
    run([py, "experiments/run_all.py", "--config", cfg, "--variants", "A1_full", "--device", "auto"])
    run([py, "scripts/analyze_and_optimize.py", "--summary", "results/round2/experiment_summary.csv"])
    print("\nRound-2 complete. See results/round2/experiment_summary.csv")


if __name__ == "__main__":
    main()

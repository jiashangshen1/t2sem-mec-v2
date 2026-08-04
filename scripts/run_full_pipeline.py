"""End-to-end: experiment -> ablation eval -> analysis -> round-2 config."""

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
    resume = "--resume" in sys.argv
    args = [py, "experiments/run_all.py", "--device", "auto"]
    if resume:
        args.append("--resume")
    run(args)
    run([py, "scripts/analyze_and_optimize.py"])
    run([py, "scripts/apply_round2_config.py"])
    print("\n=== Pipeline complete ===")
    print("Round-2 config: configs/optimized_round2.yaml")
    print("Re-train with: python experiments/run_all.py --config configs/optimized_round2.yaml --variants A1_full")


if __name__ == "__main__":
    main()

"""One-command smoke test: baselines + short T2Sem-DRL training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    py = sys.executable
    for method in ["greedy", "tjcct", "random"]:
        run([py, "experiments/train.py", "--smoke", "--method", method, "--episodes", "3"])
    run([py, "experiments/train.py", "--smoke", "--method", "t2sem_drl", "--episodes", "5"])
    run([py, "experiments/evaluate.py", "--smoke", "--method", "greedy"])
    print("Smoke pipeline OK")


if __name__ == "__main__":
    main()

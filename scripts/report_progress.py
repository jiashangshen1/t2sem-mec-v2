"""Print a one-line progress snapshot for auto-loop monitoring."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "auto_loop" / "state.json"
STDOUT = ROOT / "results" / "auto_loop" / "stdout.log"
PROGRESS_LOG = ROOT / "results" / "auto_loop" / "progress.log"


def _read_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"next_round": 2, "completed_rounds": [1], "max_rounds": 8}


def _ckpt_status(round_num: int) -> dict[int, bool]:
    ckpt = ROOT / f"results/checkpoints_round{round_num}" / "full"
    if not ckpt.exists():
        ckpt = ROOT / "results/checkpoints" / "full"
    seeds = [42, 123, 456]
    return {s: (ckpt / f"seed_{s}" / ".train_complete").exists() for s in seeds}


def _last_train_line() -> str:
    if not STDOUT.exists():
        return "no stdout.log"
    text = STDOUT.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if "train-full-s" in line:
            return line.strip()
    return "no training line"


def _phase_hint(round_num: int) -> str:
    summary = ROOT / f"results/round{round_num}" / "experiment_summary.csv"
    if summary.exists():
        return "round_complete"
    done = _ckpt_status(round_num)
    if all(done.values()):
        return "eval_or_baselines"
    return "training"


def main() -> None:
    state = _read_state()
    round_num = state.get("next_round", 2)
    if round_num in state.get("completed_rounds", []):
        round_num = min(round_num + 1, state.get("max_rounds", 8))

    done = _ckpt_status(round_num)
    phase = _phase_hint(round_num)
    train_line = _last_train_line()

    m = re.search(r"train-full-s(\d+):\s+(\d+)%.*?(\d+)/(\d+)", train_line)
    if m:
        ep = f"s{m.group(1)} {m.group(3)}/{m.group(4)} ({m.group(2)}%)"
    else:
        ep = train_line[:80]

    seeds_done = sum(done.values())
    msg = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Round {round_num}/{state.get('max_rounds', 8)} | "
        f"phase={phase} | seeds_done={seeds_done}/3 | "
        f"completed_rounds={state.get('completed_rounds', [])} | "
        f"latest={ep}"
    )
    print(msg)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    main()

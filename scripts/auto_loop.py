"""Autonomous multi-round train -> analyze -> optimize -> repeat."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "results" / "auto_loop" / "state.json"
LOG_PATH = ROOT / "results" / "auto_loop" / "loop.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"next_round": 2, "completed_rounds": [1], "max_rounds": 8}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def wait_for_file_done(path: Path, markers: tuple[str, ...], poll_s: int = 60) -> None:
    log(f"Waiting for {path.name} to finish...")
    last_size = -1
    stall = 0
    while True:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(m in text for m in markers):
                log(f"Detected completion in {path.name}")
                return
            size = path.stat().st_size
            if size == last_size:
                stall += 1
            else:
                stall = 0
                last_size = size
            if stall >= 30:
                log(f"WARN: log stalled 30 min at {path.name}, continuing anyway")
                return
        time.sleep(poll_s)


def analyze_round(round_num: int) -> None:
    summary = ROOT / f"results/round{round_num}" / "experiment_summary.csv"
    if not summary.exists():
        summary = ROOT / "results" / "experiment_summary.csv"
    rec_out = ROOT / "results" / "auto_loop" / f"round{round_num}_recommendations.json"
    run([sys.executable, "scripts/analyze_and_optimize.py", "--summary", str(summary.relative_to(ROOT))])
    src = ROOT / "results" / "optimization_recommendations.json"
    if src.exists():
        rec_out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def run_round(round_num: int) -> None:
    cfg = ROOT / f"configs/optimized_round{round_num}.yaml"
    if not cfg.exists() and round_num == 2:
        cfg = ROOT / "configs/optimized_round2.yaml"
    if not cfg.exists():
        cfg = ROOT / "configs/default.yaml"

    log(f"=== ROUND {round_num} START (config={cfg.name}) ===")
    cmd = [
        sys.executable,
        "experiments/run_all.py",
        "--config",
        str(cfg.relative_to(ROOT)),
        "--variants",
        "A1_full",
        "--device",
        "auto",
        "--resume",
    ]
    run(cmd)
    analyze_round(round_num)
    run(
        [
            sys.executable,
            "scripts/generate_next_config.py",
            "--round",
            str(round_num),
            "--summary",
            str((ROOT / f"results/round{round_num}" / "experiment_summary.csv").relative_to(ROOT)),
            "--base-config",
            str(cfg.relative_to(ROOT)),
        ]
    )
    log(f"=== ROUND {round_num} DONE ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--start-round", type=int, default=None)
    args = parser.parse_args()

    state = load_state()
    state["max_rounds"] = args.max_rounds
    start = args.start_round or state.get("next_round", 2)

    for r in range(start, args.max_rounds + 1):
        if r not in state.get("completed_rounds", []):
            run_round(r)
            done = state.get("completed_rounds", [1])
            if r not in done:
                done.append(r)
            state["completed_rounds"] = done
        state["next_round"] = r + 1
        save_state(state)

    log(f"Auto-loop finished through round {args.max_rounds}. State: {STATE_PATH}")


if __name__ == "__main__":
    main()

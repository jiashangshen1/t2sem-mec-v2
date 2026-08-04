#!/usr/bin/env python3
"""Watchdog for revision_run -> summary -> full ablation v4.3 pipeline (WSL/Linux).

Monitors progress, applies known auto-fixes, and restarts dead/stalled jobs with
--resume so long runs survive crashes, smoke artifacts, and log stalls.

Usage (WSL):
    /root/venvs/ml/bin/python scripts/revision_pipeline_watchdog.py
    /root/venvs/ml/bin/python scripts/revision_pipeline_watchdog.py --once
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

PY_WSL = Path("/root/venvs/ml/bin/python")
PY_CMD = str(PY_WSL if PY_WSL.is_file() else sys.executable)

REV_DIR = ROOT / "results" / "revision_v43"
ABL_DIR = ROOT / "results" / "ablation_v43"
PAPER = Path("/mnt/c/Users/12293/Desktop/T2Sem-DRL-v4-paper")

REV_LOG = REV_DIR / "run.log"
PIPE_LOG = REV_DIR / "pipeline.log"
WATCH_LOG = REV_DIR / "watchdog.log"
STATE_FILE = REV_DIR / "watchdog_state.json"
SUMMARY_CSV = REV_DIR / "revision_summary.csv"
ABL_LOG = ABL_DIR / "ablation_run.log"
ABL_CSV = ABL_DIR / "ablation_summary.csv"

EXPECTED_REV_ROWS = 28
EXPECTED_ABL_ROWS = 10
STALL_MINUTES = 50
CHECK_INTERVAL = 10 * 60
MAX_RESTARTS = 12

MONITORED_PROCS = ("revision_run.py", "ablation_run.py", "revision_pipeline_watchdog.py")


def wlog(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    REV_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"restarts": 0, "post_revision_done": False, "last_fp": [0, 0.0]}


def save_state(st: dict) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True, errors="ignore")
    except subprocess.CalledProcessError:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def find_workers() -> dict[str, list[str]]:
    return {
        "revision": [l for l in pgrep("revision_run.py") if "revision_run.py" in l],
        "ablation": [l for l in pgrep("ablation_run.py") if "ablation_run.py" in l],
    }


def log_fingerprint(paths: list[Path]) -> tuple[int, float]:
    size, mtime = 0, 0.0
    for p in paths:
        if p.is_file():
            st = p.stat()
            size += st.st_size
            mtime = max(mtime, st.st_mtime)
    return size, mtime


def tail_text(path: Path, n: int = 8) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-n:]):
        s = line.strip()
        if s and not s.startswith("train-"):
            return s[:120]
    return lines[-1][:120] if lines else ""


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return max(0, len(path.read_text(encoding="utf-8").strip().splitlines()) - 1)
    except OSError:
        return 0


def revision_complete() -> bool:
    return csv_rows(SUMMARY_CSV) >= EXPECTED_REV_ROWS


def ablation_complete() -> bool:
    if ABL_CSV.is_file() and csv_rows(ABL_CSV) >= EXPECTED_ABL_ROWS:
        return True
    if ABL_LOG.is_file():
        return "ABLATION_DONE" in ABL_LOG.read_text(encoding="utf-8", errors="ignore")
    return False


def pipeline_complete() -> bool:
    if PIPE_LOG.is_file() and "PIPELINE_DONE" in PIPE_LOG.read_text(encoding="utf-8", errors="ignore"):
        return True
    return revision_complete() and ablation_complete()


# --- Auto-fixes -----------------------------------------------------------

def fix_smoke_independent_ppo() -> bool:
    """Remove 15-ep smoke checkpoint so --resume trains full 300 ep."""
    flag = ROOT / "results/checkpoints_independent_ppo/full/seed_42/.train_complete"
    ckpt = ROOT / "results/checkpoints_independent_ppo/full/seed_42"
    if not flag.is_file():
        return False
    if re.search(r"^episodes=15\s*$", flag.read_text(encoding="utf-8"), re.M):
        wlog("FIX: delete independent_ppo seed_42 smoke checkpoint (15 ep)")
        shutil.rmtree(ckpt, ignore_errors=True)
        return True
    return False


def fix_stale_traceback_note() -> bool:
    """Log if last revision crash was the known actor.net bug (already patched)."""
    if not REV_LOG.is_file():
        return False
    text = REV_LOG.read_text(encoding="utf-8", errors="ignore")
    if "IndependentTerminalActorBank" in text and "has no attribute 'net'" in text:
        if "FIX_APPLIED: independent_ppo BC" not in text:
            with open(REV_LOG, "a", encoding="utf-8") as f:
                f.write("\n# FIX_APPLIED: independent_ppo BC routed per-terminal (ppo.py)\n")
            wlog("FIX: noted patched IndependentTerminalActorBank BC bug in run.log")
            return True
    return False


def apply_auto_fixes() -> list[str]:
    applied = []
    if fix_smoke_independent_ppo():
        applied.append("smoke_ipp_seed42")
    if fix_stale_traceback_note():
        applied.append("traceback_note")
    return applied


# --- Launchers ------------------------------------------------------------

def spawn(cmd: list[str], log_path: Path, tag: str) -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wlog(f"SPAWN {tag}: {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== watchdog spawn {tag} {datetime.now().isoformat()} ===\n")
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )


def restart_revision() -> None:
    apply_auto_fixes()
    spawn([PY_CMD, "scripts/revision_run.py", "--device", "cpu", "--resume"], REV_LOG, "revision")


def restart_ablation() -> None:
    spawn(
        [
            PY_CMD,
            "scripts/ablation_run.py",
            "--config",
            "configs/improved_v4_pertier.yaml",
            "--suite",
            "v43",
            "--episodes",
            "300",
            "--seeds",
            "42",
            "123",
            "456",
            "--eval-episodes",
            "30",
            "--device",
            "cpu",
            "--resume",
        ],
        ABL_LOG,
        "ablation",
    )


def retrain_independent_ppo_seed42(st: dict) -> None:
    fix_smoke_independent_ppo()
    st["rebuild_summary_after_ipp"] = True
    save_state(st)
    spawn(
        [PY_CMD, "scripts/revision_run.py", "--device", "cpu", "--resume", "--only", "independent_ppo"],
        REV_LOG,
        "independent_ppo_seed42",
    )


def rebuild_revision_summary() -> None:
    spawn(
        [PY_CMD, "scripts/revision_run.py", "--device", "cpu", "--resume", "--skip-train"],
        REV_LOG,
        "revision_eval_all",
    )


def needs_ipp_seed42_retrain() -> bool:
    flag = ROOT / "results/checkpoints_independent_ppo/full/seed_42/.train_complete"
    if not flag.is_file():
        return False
    return bool(re.search(r"^episodes=15\s*$", flag.read_text(encoding="utf-8"), re.M))


def run_post_revision(st: dict) -> None:
    if st.get("post_revision_done"):
        return

    workers = find_workers()

    if needs_ipp_seed42_retrain():
        wlog("POST-REV: independent_ppo seed_42 needs full retrain")
        if workers["revision"]:
            return
        retrain_independent_ppo_seed42(st)
        return

    if not revision_complete():
        if not workers["revision"]:
            wlog("POST-REV: revision incomplete, no worker -> restart")
            restart_revision()
        return

    wlog("POST-REV: running summarize + sync")
    subprocess.run([PY_CMD, "scripts/summarize_revision.py"], cwd=str(ROOT), check=False)
    if PAPER.is_dir():
        data = PAPER / "data"
        data.mkdir(parents=True, exist_ok=True)
        for src, name in [
            (SUMMARY_CSV, "revision_summary.csv"),
            (SUMMARY_CSV, "experiment_summary.csv"),
            (REV_DIR / "baseline_v43.csv", "baseline_v43.csv"),
            (REV_DIR / "REVISION_SUMMARY.md", "../REVISION_SUMMARY.md"),
        ]:
            if src.is_file():
                dst = data / name if not name.startswith("..") else PAPER / "REVISION_SUMMARY.md"
                shutil.copy2(src, dst)
        wlog(f"POST-REV: synced to {PAPER}")

    st["post_revision_done"] = True
    save_state(st)
    with open(PIPE_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n=== post_revision done {datetime.now().isoformat()} ===\n")


# --- Main check loop ------------------------------------------------------

def check_once(st: dict) -> tuple[dict, bool]:
    if pipeline_complete():
        wlog("PIPELINE_DONE -> watchdog exit")
        with open(PIPE_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== PIPELINE DONE {datetime.now().isoformat()} ===\nABLATION_DONE\n")
        if PAPER.is_dir() and ABL_CSV.is_file():
            shutil.copy2(ABL_CSV, PAPER / "data" / "ablation_summary.csv")
        return st, True

    apply_auto_fixes()
    workers = find_workers()

    if st.get("rebuild_summary_after_ipp") and not workers["revision"]:
        if revision_complete() and csv_rows(SUMMARY_CSV) >= EXPECTED_REV_ROWS:
            st["rebuild_summary_after_ipp"] = False
            save_state(st)
            wlog("POST-REV: summary rebuild complete")
        else:
            wlog("POST-REV: rebuild revision_summary.csv (eval all jobs)")
            rebuild_revision_summary()
            return st, False

    if not st.get("post_revision_done"):
        run_post_revision(st)
        workers = find_workers()

    if not st.get("post_revision_done"):
        if workers["revision"]:
            fp = log_fingerprint([REV_LOG, SUMMARY_CSV])
            last = tuple(st.get("last_fp", [0, 0.0]))
            growing = fp != last and fp[0] > last[0]
            stale_min = (time.time() - last[1]) / 60.0 if last[1] else 0.0
            wlog(
                f"CHECK revision alive rows={csv_rows(SUMMARY_CSV)}/{EXPECTED_REV_ROWS} "
                f"growing={growing} stale={stale_min:.0f}m | {tail_text(REV_LOG)}"
            )
            if last[0] > 0 and not growing and stale_min >= STALL_MINUTES:
                if st["restarts"] >= MAX_RESTARTS:
                    wlog("GIVE UP: revision stalled")
                    return st, True
                wlog(f"FIX: revision stalled {stale_min:.0f}m -> kill & restart")
                for line in workers["revision"]:
                    subprocess.run(["kill", "-9", line.split(None, 1)[0]], check=False)
                time.sleep(3)
                restart_revision()
                st["restarts"] += 1
            st["last_fp"] = list(fp)
        elif not st.get("rebuild_summary_after_ipp") and not revision_complete():
            if st["restarts"] >= MAX_RESTARTS:
                wlog(f"GIVE UP: revision dead, restarts>={MAX_RESTARTS}")
                return st, True
            if not needs_ipp_seed42_retrain():
                wlog(f"FIX: revision idle ({csv_rows(SUMMARY_CSV)} rows), restarting")
                restart_revision()
                st["restarts"] += 1

    if st.get("post_revision_done") and not ablation_complete():
        if not workers["ablation"] and not workers["revision"]:
            if st["restarts"] >= MAX_RESTARTS:
                wlog("GIVE UP: ablation dead")
                return st, True
            wlog("FIX: starting ablation suite")
            restart_ablation()
            st["restarts"] += 1
        elif workers["ablation"]:
            fp = log_fingerprint([ABL_LOG, ABL_CSV])
            abl_state = st.get("abl_fp", [0, 0.0])
            growing = fp != tuple(abl_state) and fp[0] > abl_state[0]
            stale_min = (time.time() - abl_state[1]) / 60.0 if abl_state[1] else 0.0
            wlog(
                f"CHECK ablation alive rows={csv_rows(ABL_CSV)}/{EXPECTED_ABL_ROWS} "
                f"growing={growing} stale={stale_min:.0f}m | {tail_text(ABL_LOG)}"
            )
            if abl_state[0] > 0 and not growing and stale_min >= STALL_MINUTES:
                wlog(f"FIX: ablation stalled -> kill & restart")
                for line in workers["ablation"]:
                    subprocess.run(["kill", "-9", line.split(None, 1)[0]], check=False)
                time.sleep(3)
                restart_ablation()
                st["restarts"] += 1
            st["abl_fp"] = list(fp)

    save_state(st)
    return st, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    wlog(
        f"Watchdog start interval={args.interval}s stall={STALL_MINUTES}m "
        f"max_restarts={MAX_RESTARTS}"
    )
    st = load_state()
    while True:
        try:
            st, stop = check_once(st)
        except Exception as e:
            wlog(f"CHECK ERROR: {e}")
            stop = False
        if args.once or stop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

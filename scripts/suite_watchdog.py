"""Watchdog for the paper-suite run (pareto -> mask-sensitivity -> ablation).

Every N minutes it checks whether the suite is alive and making progress. If the
suite process has died (e.g. the box slept) or its logs stalled, and the suite
has NOT finished, it relaunches scripts/run_paper_suite.ps1 -- which itself
skips already-completed steps, so a restart resumes at the failed step.

Usage:
    python scripts/suite_watchdog.py                 # 20-min checks, loops forever
    python scripts/suite_watchdog.py --interval 600  # custom seconds
    python scripts/suite_watchdog.py --once          # single check (for testing)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_PS1 = ROOT / "scripts" / "run_paper_suite.ps1"
SUITE_LOG = ROOT / "results" / "paper_suite.log"
WATCH_LOG = ROOT / "results" / "suite_watchdog.log"
STEP_LOGS = [
    ROOT / "results" / "pareto_full.log",
    ROOT / "results" / "mask_sens.log",
    ROOT / "results" / "ablation_full.log",
]
SUITE_SCRIPTS = ("pareto_sweep", "mask_sensitivity", "ablation_run")
STALL_MINUTES = 45  # no log growth this long while "alive" => stuck
MAX_RESTARTS = 8


def wlog(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(WATCH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def suite_done() -> bool:
    if not SUITE_LOG.exists():
        return False
    try:
        return "SUITE_DONE" in SUITE_LOG.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def find_suite_python() -> list[dict]:
    """Return python.exe processes whose command line runs a suite script."""
    try:
        out = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            text=True, errors="ignore", timeout=60,
        ).strip()
    except (subprocess.SubprocessError, OSError) as e:
        wlog(f"proc-list error: {e}")
        return []
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    hits = []
    for d in data:
        cmd = (d.get("CommandLine") or "")
        if any(s in cmd for s in SUITE_SCRIPTS):
            hits.append({"pid": d.get("ProcessId"), "cmd": cmd[:140]})
    return hits


def combined_fp() -> tuple[int, float]:
    """Max (size, mtime) across step logs -> progress fingerprint."""
    size, mtime = 0, 0.0
    for p in STEP_LOGS:
        if p.exists():
            st = p.stat()
            size += st.st_size
            mtime = max(mtime, st.st_mtime)
    return size, mtime


def active_step_tail() -> str:
    newest = None
    for p in STEP_LOGS:
        if p.exists() and (newest is None or p.stat().st_mtime > newest.stat().st_mtime):
            newest = p
    if newest is None:
        return ""
    try:
        lines = newest.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-6:]):
        s = line.strip()
        if s and "NativeCommandError" not in s and "CategoryInfo" not in s:
            return f"{newest.name}: {s[:110]}"
    return ""


def restart_suite() -> None:
    log_dir = ROOT / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    spawn_log = log_dir / "suite_spawn.log"
    with open(spawn_log, "a", encoding="utf-8") as lf:
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(SUITE_PS1)],
            cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
        )
    wlog("RESTART: relaunched run_paper_suite.ps1 (skips completed steps)")


def check_once(last_fp: tuple[int, float], restarts: int) -> tuple[tuple[int, float], int, bool]:
    if suite_done():
        wlog("SUITE_DONE detected -> watchdog exiting")
        return last_fp, restarts, True

    procs = find_suite_python()
    fp = combined_fp()
    growing = fp != last_fp and fp[0] > 0
    tail = active_step_tail()
    wlog(f"CHECK alive={len(procs)} log_kb={fp[0] // 1024} growing={growing} "
         f"restarts={restarts}" + (f" | {tail}" if tail else ""))

    if not procs:
        if restarts >= MAX_RESTARTS:
            wlog(f"GIVE UP: dead and restarts>={MAX_RESTARTS}")
            return fp, restarts, True
        wlog("FIX: no suite process -> restart")
        restart_suite()
        return fp, restarts + 1, False

    if last_fp[0] > 0 and not growing:
        stale_min = (time.time() - last_fp[1]) / 60.0 if last_fp[1] else 0.0
        if stale_min >= STALL_MINUTES:
            if restarts >= MAX_RESTARTS:
                wlog(f"GIVE UP: stalled and restarts>={MAX_RESTARTS}")
                return fp, restarts, True
            wlog(f"FIX: log stalled {stale_min:.0f} min -> kill & restart")
            for p in procs:
                subprocess.run(["taskkill", "/PID", str(p["pid"]), "/F"],
                               check=False, capture_output=True)
            time.sleep(5)
            restart_suite()
            return fp, restarts + 1, False

    return fp, restarts, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=20 * 60, help="seconds between checks")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    wlog(f"Watchdog started interval={args.interval}s stall={STALL_MINUTES}min max_restarts={MAX_RESTARTS}")
    last_fp = (0, 0.0)
    restarts = 0
    while True:
        try:
            last_fp, restarts, stop = check_once(last_fp, restarts)
        except Exception as e:  # never let the watchdog die on a transient error
            wlog(f"CHECK ERROR: {e}")
            stop = False
        if args.once or stop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

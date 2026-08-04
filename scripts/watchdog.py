"""Watchdog: check experiment progress every N minutes; restart if stuck/dead."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "results" / "auto_loop" / "state.json"
WATCH_LOG = ROOT / "results" / "auto_loop" / "watchdog.log"
STDOUT_LOG = ROOT / "results" / "auto_loop" / "stdout.log"
STALL_MINUTES = 35  # no log growth this long => stuck
CHECK_INTERVAL_S = 20 * 60
MIN_PYTHON_MEM_MB = 80


def wlog(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(WATCH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"next_round": 2, "completed_rounds": [1], "max_rounds": 8}


def find_training_python() -> list[dict]:
    try:
        import psutil
    except ImportError:
        return _find_python_tasklist()

    hits = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
        try:
            if p.info["name"] and "python" in p.info["name"].lower():
                cmd = " ".join(p.info["cmdline"] or [])
                if "auto_loop" in cmd or "run_all" in cmd or "experiments" in cmd:
                    mem_mb = (p.info["memory_info"].rss if p.info["memory_info"] else 0) / (1024 * 1024)
                    hits.append({"pid": p.info["pid"], "cmd": cmd[:120], "mem_mb": mem_mb})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return hits


def _find_python_tasklist() -> list[dict]:
    out = subprocess.check_output(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
        text=True,
        errors="ignore",
    )
    hits = []
    for line in out.strip().splitlines():
        if "python.exe" in line.lower():
            parts = line.replace('"', "").split(",")
            if len(parts) >= 5:
                pid = int(parts[1])
                mem_kb = int(parts[4].replace(" K", "").replace(",", ""))
                hits.append({"pid": pid, "cmd": "python.exe", "mem_mb": mem_kb / 1024})
    return hits


def log_fingerprint() -> tuple[int, float]:
    if not STDOUT_LOG.exists():
        return 0, 0.0
    st = STDOUT_LOG.stat()
    return st.st_size, st.st_mtime


def kill_training_processes() -> None:
    for p in find_training_python():
        try:
            import psutil

            proc = psutil.Process(p["pid"])
            proc.terminate()
            proc.wait(timeout=10)
            wlog(f"Killed pid={p['pid']}")
        except Exception:
            try:
                subprocess.run(["taskkill", "/PID", str(p["pid"]), "/F"], check=False, capture_output=True)
                wlog(f"Taskkill pid={p['pid']}")
            except Exception as e:
                wlog(f"Failed kill pid={p['pid']}: {e}")


def restart_auto_loop() -> None:
    state = load_state()
    start = int(state.get("next_round", 2))
    max_r = int(state.get("max_rounds", 8))
    wlog(f"Restarting auto_loop from round {start} (max={max_r})")
    log_file = ROOT / "results" / "auto_loop" / f"stdout_restart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w", encoding="utf-8") as lf:
        subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "auto_loop.py"),
                "--max-rounds",
                str(max_r),
                "--start-round",
                str(start),
            ],
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    wlog(f"Spawned auto_loop -> {log_file.name}")


def check_once(last_fp: tuple[int, float]) -> tuple[int, float]:
    state = load_state()
    procs = find_training_python()
    big = [p for p in procs if p["mem_mb"] >= MIN_PYTHON_MEM_MB]
    fp = log_fingerprint()
    size, mtime = fp

    wlog(
        f"CHECK round={state.get('next_round')} procs={len(big)} "
        f"log_kb={size//1024} completed={state.get('completed_rounds')}"
    )

    if STDOUT_LOG.exists() and size > 0:
        tail = STDOUT_LOG.read_text(encoding="utf-8", errors="ignore")[-400:]
        for line in tail.splitlines()[-3:]:
            if "train-" in line or "Skip" in line or "ROUND" in line:
                wlog(f"  tail: {line.strip()[:100]}")

    alive = len(big) > 0
    growing = fp != last_fp and size > 0

    if not alive:
        wlog("FIX: no training python process -> restart")
        restart_auto_loop()
        return fp

    if last_fp[0] > 0 and not growing:
        stale_min = (time.time() - last_fp[1]) / 60.0
        if stale_min >= STALL_MINUTES:
            wlog(f"FIX: log stalled {stale_min:.0f} min -> kill & restart")
            kill_training_processes()
            time.sleep(5)
            restart_auto_loop()
            return fp

    return fp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL_S, help="seconds between checks")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    wlog(f"Watchdog started interval={args.interval}s stall={STALL_MINUTES}min")
    last_fp = (0, 0.0)
    while True:
        try:
            last_fp = check_once(last_fp)
        except Exception as e:
            wlog(f"CHECK ERROR: {e}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

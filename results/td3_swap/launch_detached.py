"""Launch a python job fully detached (no console, survives parent exit).

Default target is the TD3 comparison; any script under the repo can be given
as argv[1] with extra args after it. Uses pythonw.exe so no console window
exists (nothing to close => no stray Ctrl+C), DETACHED_PROCESS so it is not
tied to this shell, and tries CREATE_BREAKAWAY_FROM_JOB so a parent job object
cannot reap it on exit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # t2sem-mec-v2
PYTHONW = Path(sys.executable).with_name("pythonw.exe")
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

if len(sys.argv) > 1:
    target = [sys.argv[1], *sys.argv[2:]]
else:
    target = [
        "experiments\\td3_swap.py",
        "--device",
        "cpu",
        "--log-file",
        "results/td3_swap/run_full.log",
    ]

cmd = [str(PYTHONW if PYTHONW.exists() else Path(sys.executable)), *target]

for flags, label in (
    (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, "detached+breakaway"),
    (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, "detached"),
):
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        print(f"launched pid={proc.pid} via {label}")
        break
    except OSError as exc:  # breakaway refused by the job object
        print(f"{label} refused: {exc}")
else:
    raise SystemExit("failed to launch detached run")

"""Track and normalize external baseline full-reproduction status.

This script does not claim the baseline is reproduced by default. It creates a
single status table so "Lite" and "Full" baselines are not mixed in reporting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ideal_track" / "baseline_repro_status.csv"

BASELINES = [
    ("tjcct_full", "pending"),
    ("semantic_mappo_full", "pending"),
    ("hao_tmc24_like", "pending"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-done", nargs="*", default=[])
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done_set = set(args.mark_done)
    for name, default_state in BASELINES:
        state = "done" if name in done_set else default_state
        rows.append(
            {
                "baseline": name,
                "status": state,
                "evidence_path": "",
                "note": args.note if name in done_set else "",
            }
        )

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

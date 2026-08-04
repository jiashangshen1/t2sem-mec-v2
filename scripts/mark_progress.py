"""Mark finished training runs so --resume can skip them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Completed before user stopped Cursor (from run_log.txt)
DONE = [
    ("full", 42),
    ("full", 123),
    ("full", 456),
    ("no_semantic", 42),
    ("no_semantic", 123),
    ("no_semantic", 456),
    ("single_timescale", 42),
    ("single_timescale", 123),
    ("single_timescale", 456),
    ("no_sla_penalty", 42),
    ("no_sla_penalty", 123),
    # no_sla_penalty 456 was interrupted (~104/200) — do NOT mark
]

PENDING = [
    ("no_sla_penalty", 456),
    ("uav_only", 42),
    ("uav_only", 123),
    ("uav_only", 456),
    ("fixed_mu", 42),
    ("fixed_mu", 123),
    ("fixed_mu", 456),
    ("no_contention", 42),
    ("no_contention", 123),
    ("no_contention", 456),
]


def main() -> None:
    ckpt_root = ROOT / "results" / "checkpoints"
    marked = []
    for variant, seed in DONE:
        d = ckpt_root / variant / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        marker = d / ".train_complete"
        if not marker.exists():
            marker.write_text(f"episodes=200\nmarked_by=mark_progress.py\n", encoding="utf-8")
        marked.append(f"{variant}/seed_{seed}")

    state = {
        "completed_training": [f"{v}/seed_{s}" for v, s in DONE],
        "pending_training": [f"{v}/seed_{s}" for v, s in PENDING],
        "note": "Close Cursor safely. Resume with: python scripts/run_full_pipeline.py --resume",
    }
    out = ROOT / "results" / "resume_state.json"
    out.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Marked {len(marked)} completed runs.")
    print(f"Wrote {out}")
    print(f"Pending: {len(PENDING)} training runs")


if __name__ == "__main__":
    main()

"""Detached watcher: when the TD3 swap run finishes, write the comparison report.

Polls results/td3_swap/run_full.log every minute (up to 10 h). On "DONE" it
reads summary.csv and writes FINAL_REPORT.md with the per-algorithm aggregate
(mean +- std across seeds) and the per-tier breakdown. If the log stops
growing for 20 min it writes STALL_NOTE.md instead. All writes stay inside
results/td3_swap/.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent          # .../results/td3_swap
LOG = BASE / "run_full.log"
CSV_PATH = BASE / "summary.csv"


def _safe_write(name: str, text: str) -> None:
    target = os.path.abspath(str(BASE / name))
    base = os.path.abspath(str(BASE))
    if not target.startswith(base + os.sep):
        raise ValueError("refusing to write outside results/td3_swap: " + target)
    Path(target).write_text(text, encoding="utf-8")


def _fmt(xs: list) -> str:
    vals = [float(x) for x in xs]
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


def build_report() -> str:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_seeds = max(len({r["seed"] for r in rows if r["algo"] == algo})
                 for algo in {r["algo"] for r in rows})
    lines = [
        f"# DDPG vs TD3 长尺度智能体对照（v4.3 协议，2 算法 × {n_seeds} 种子 × 300 回合）",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　数据源：results/td3_swap/summary.csv",
        "",
        "| 算法 | SLA | 时延(s) | 能耗(J) | critical | interactive | best_effort | 种子数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for algo in sorted({r["algo"] for r in rows}):
        sub = [r for r in rows if r["algo"] == algo]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                algo,
                _fmt([r["sla_rate_mean"] for r in sub]),
                _fmt([r["avg_delay_mean"] for r in sub]),
                _fmt([r["total_energy_mean"] for r in sub]),
                _fmt([r.get("sla_critical", "nan") for r in sub]),
                _fmt([r.get("sla_interactive", "nan") for r in sub]),
                _fmt([r.get("sla_best_effort", "nan") for r in sub]),
                len(sub),
            )
        )
    by_algo = {r["algo"]: [] for r in rows}
    for r in rows:
        by_algo[r["algo"]].append(r)
    lines += ["", "## 逐种子明细", ""]
    for algo, sub in sorted(by_algo.items()):
        lines.append(f"### {algo}")
        lines.append("")
        lines.append("| seed | SLA | 时延(s) | 能耗(J) | critical |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(sub, key=lambda x: int(x["seed"])):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    r["seed"],
                    f"{float(r['sla_rate_mean']):.4f}",
                    f"{float(r['avg_delay_mean']):.3f}",
                    f"{float(r['total_energy_mean']):.0f}",
                    f"{float(r.get('sla_critical', 'nan')):.4f}",
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    deadline = time.time() + 16 * 3600
    last_size = -1
    stall_deadline = time.time() + 20 * 60
    while time.time() < deadline:
        time.sleep(60)
        if not LOG.exists():
            continue
        size = LOG.stat().st_size
        if size != last_size:
            last_size = size
            stall_deadline = time.time() + 20 * 60
        elif time.time() > stall_deadline:
            _safe_write(
                "STALL_NOTE.md",
                f"log stalled at {size} bytes; run may have died. Check processes.\n",
            )
            return
        text = LOG.read_text(encoding="utf-8", errors="ignore")
        if "DONE" in text and CSV_PATH.exists():
            time.sleep(5)
            _safe_write("FINAL_REPORT.md", build_report())
            return


if __name__ == "__main__":
    main()

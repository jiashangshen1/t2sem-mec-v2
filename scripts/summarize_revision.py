"""Print revision_summary.csv as a readable markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "revision_v43" / "revision_summary.csv"
OUT = ROOT / "results" / "revision_v43" / "REVISION_SUMMARY.md"


def main() -> None:
    if not CSV.is_file():
        print(f"Missing {CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV)
    df["sla_pct"] = df["sla_rate_mean"] * 100

    lines = [
        "# T2Sem-DRL v4.3 Revision Run Summary",
        "",
        f"Source: `{CSV.relative_to(ROOT)}` ({len(df)} rows)",
        "",
        "## Per-job mean SLA (%)",
        "",
        "| Job | SLA mean | SLA std (across seeds) | Delay | Energy (J) |",
        "|-----|----------|------------------------|-------|------------|",
    ]

    for job, g in df.groupby("job", sort=False):
        lines.append(
            f"| {job} | {g['sla_pct'].mean():.2f} | {g['sla_pct'].std():.2f} | "
            f"{g['avg_delay_mean'].mean():.3f} | {g['total_energy_mean'].mean():.0f} |"
        )

    lines += ["", "## Per seed", "", "| Job | Seed | SLA % | Energy |", "|-----|------|-------|--------|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['job']} | {int(r['seed'])} | {r['sla_pct']:.2f} | {r['total_energy_mean']:.0f} |"
        )

    if "greedy" in df["job"].values:
        greedy = df[df["job"] == "greedy"]["sla_pct"].mean()
        lines += ["", f"**Greedy baseline mean:** {greedy:.2f}%", ""]
        for job in ["t2sem_full", "t2sem_single_lambda", "t2sem_single_ts", "independent_ppo"]:
            sub = df[df["job"] == job]
            if len(sub):
                lines.append(
                    f"- **{job}** vs greedy: {sub['sla_pct'].mean() - greedy:+.2f} pp"
                )

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

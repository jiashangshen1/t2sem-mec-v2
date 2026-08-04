"""Post-experiment analysis and config optimization suggestions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/experiment_summary.csv")
    args = parser.parse_args()

    path = ROOT / args.summary
    if not path.exists():
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    print("=== Experiment Summary ===\n")
    print(df.groupby(["method", "variant"])[["sla_rate_mean", "avg_delay_mean"]].mean().round(3))

    a1 = df[(df["variant"] == "A1_full") & (df["method"] == "T2Sem-DRL")]
    if not a1.empty:
        a1_sla = a1["sla_rate_mean"].mean()
    else:
        a1_sla = 0.0

    abl = df[df["method"] == "T2Sem-DRL"].copy()
    abl = abl[abl["variant"] != "A1_full"]
    deltas = []
    for v in abl["variant"].unique():
        sub = abl[abl["variant"] == v]["sla_rate_mean"].mean()
        deltas.append({"variant": v, "sla_mean": sub, "delta_vs_A1": sub - a1_sla})
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = delta_df.sort_values("delta_vs_A1")
    print("\n=== Ablation deltas vs A1 ===\n")
    if delta_df.empty:
        print("(no ablation variants in summary — A1-only run)")
    else:
        print(delta_df.round(3).to_string(index=False))

    greedy = df[(df["method"] == "greedy") & (df["variant"] == "baseline")]
    random_df = df[(df["method"] == "random") & (df["variant"] == "baseline")]
    recs = []
    if not greedy.empty and a1_sla > greedy["sla_rate_mean"].mean():
        recs.append("T2Sem-DRL beats greedy on SLA — keep current architecture.")
    elif not greedy.empty:
        recs.append("T2Sem-DRL below greedy — increase w_sla, BC demos, or train episodes.")
    if not random_df.empty and a1_sla < random_df["sla_rate_mean"].mean():
        recs.append(
            f"T2Sem-DRL below random ({a1_sla:.3f} vs {random_df['sla_rate_mean'].mean():.3f}) — boost SLA reward and training length."
        )

    worst = delta_df.iloc[0] if len(delta_df) > 0 else None
    if worst is not None and worst["delta_vs_A1"] < -0.03:
        recs.append(f"Largest ablation drop: {worst['variant']} ({worst['delta_vs_A1']:.3f}) — do not disable this component.")

    out = Path(args.summary).parent / "optimization_recommendations.json"
    if args.summary == "results/experiment_summary.csv":
        out = ROOT / "results" / "optimization_recommendations.json"
    out.write_text(json.dumps({"a1_sla": float(a1_sla), "recommendations": recs, "deltas": deltas}, indent=2), encoding="utf-8")
    print(f"\n=== Recommendations ===")
    for r in recs:
        print(f"  - {r}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

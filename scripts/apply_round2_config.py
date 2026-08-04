"""Apply round-2 optimizations based on experiment analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendations", default="results/optimization_recommendations.json")
    parser.add_argument("--output", default="configs/optimized_round2.yaml")
    args = parser.parse_args()

    rec_path = ROOT / args.recommendations
    base_path = ROOT / "configs" / "default.yaml"
    out_path = ROOT / args.output

    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    recs = []
    if rec_path.exists():
        data = json.loads(rec_path.read_text(encoding="utf-8"))
        recs = data.get("recommendations", [])
        a1_sla = float(data.get("a1_sla", 0.5))
    else:
        a1_sla = 0.5

    # Round-2 defaults: conservative improvements
    cfg["drl"]["bc_demo_episodes"] = max(int(cfg["drl"]["bc_demo_episodes"]), 30)
    cfg["drl"]["bc_train_epochs"] = max(int(cfg["drl"]["bc_train_epochs"]), 120)
    cfg["reward"]["w_sla"] = float(cfg["reward"]["w_sla"]) * (1.2 if a1_sla < 0.5 else 1.0)
    cfg["reward"]["w_fairness"] = min(float(cfg["reward"]["w_fairness"]) + 0.1, 0.5)
    cfg["drl"]["short"]["update_epochs"] = 8
    cfg["drl"]["long"]["noise_std"] = 0.06

    if any("below greedy" in r for r in recs):
        cfg["drl"]["total_long_frames"] = int(cfg["drl"]["total_long_frames"]) + 50
        cfg["task"]["arrival_rate"] = float(cfg["task"]["arrival_rate"]) * 0.95

    out_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote optimized config: {out_path}")
    for r in recs:
        print(f"  (based on) {r}")


if __name__ == "__main__":
    main()

"""P2: Energy-SLA Pareto frontier for T2Sem-DRL.

Sweeps the "energy pressure" knob (trajectory-energy weight + actor L2 + long
exploration noise). Each operating point is trained briefly and evaluated to
record (energy, SLA). The masked baselines are evaluated once as reference
points. Produces a CSV and, if matplotlib is available, a Pareto plot showing
that the method's frontier dominates the baselines.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_all import evaluate_baseline_all_seeds, evaluate_trained
from experiments.train import train_t2sem
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

# (label, w_traj_energy, action_l2_coef, long noise_std) from low->high pressure
OPERATING_POINTS = [
    ("P1_minpressure", 0.05, 0.0, 0.15),
    ("P2_low", 0.25, 0.01, 0.12),
    ("P3_v4", 0.60, 0.05, 0.08),
    ("P4_high", 1.20, 0.12, 0.05),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved_v4_pertier.yaml")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123])
    parser.add_argument("--eval-episodes", type=int, default=15)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_config(args.device, base_cfg)
    base_cfg = load_config("configs/default.yaml", args.config)
    out_dir = project_root() / "results" / "pareto"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | points={len(OPERATING_POINTS)} seeds={args.seeds} "
          f"ep={args.episodes} eval={args.eval_episodes}", flush=True)

    rows = []
    t0 = time.perf_counter()
    for idx, (label, w_traj, l2, noise) in enumerate(OPERATING_POINTS):
        cfg = copy.deepcopy(base_cfg)
        cfg["drl"]["total_long_frames"] = int(args.episodes)
        cfg["experiment"]["eval_seeds"] = list(args.seeds)
        cfg["experiment"]["eval_episodes"] = int(args.eval_episodes)
        cfg["experiment"]["checkpoint_dir"] = f"results/checkpoints_pareto/{label}"
        cfg["experiment"]["results_dir"] = f"results/pareto/{label}"
        cfg["reward"]["w_traj_energy"] = w_traj
        cfg["drl"]["long"]["action_l2_coef"] = l2
        cfg["drl"]["long"]["noise_std"] = noise

        for seed in args.seeds:
            print(f"\n[{label}] train seed={seed} (w_traj={w_traj}, l2={l2}, noise={noise})", flush=True)
            ckpt = train_t2sem(cfg, "full", seed, device)
            agg = aggregate_metrics(evaluate_trained(cfg, "full", seed, ckpt, int(args.eval_episodes), device))
            rows.append({
                "point": label, "seed": seed, "w_traj_energy": w_traj,
                "action_l2_coef": l2, "noise_std": noise,
                "sla": round(agg["sla_rate_mean"], 4),
                "delay": round(agg["avg_delay_mean"], 4),
                "energy": round(agg["total_energy_mean"], 1),
            })
            print(f"  [eval] {label} seed={seed} SLA={agg['sla_rate_mean']:.3f} "
                  f"energy={agg['total_energy_mean']:.0f}", flush=True)

    # Baselines once (independent of operating point) as reference points.
    print("\n--- Baselines (reference points) ---", flush=True)
    base_eval_cfg = copy.deepcopy(base_cfg)
    base_eval_cfg["experiment"]["eval_seeds"] = list(args.seeds)
    base_eval_cfg["experiment"]["eval_episodes"] = int(args.eval_episodes)
    for bl in ["random", "greedy", "local_only"]:
        for r in evaluate_baseline_all_seeds(base_eval_cfg, bl, int(args.eval_episodes), list(args.seeds)):
            rows.append({
                "point": bl, "seed": r["seed"], "w_traj_energy": "", "action_l2_coef": "",
                "noise_std": "", "sla": round(r["sla_rate_mean"], 4),
                "delay": round(r["avg_delay_mean"], 4), "energy": round(r["total_energy_mean"], 1),
            })

    csv_path = out_dir / "pareto_points.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}", flush=True)

    # Aggregate per point (mean over seeds) and plot.
    agg_pts = {}
    for r in rows:
        agg_pts.setdefault(r["point"], {"sla": [], "energy": []})
        agg_pts[r["point"]]["sla"].append(r["sla"])
        agg_pts[r["point"]]["energy"].append(r["energy"])
    summary = {p: (float(np.mean(v["energy"])), float(np.mean(v["sla"]))) for p, v in agg_pts.items()}
    print("\npoint                 energy        SLA")
    for p, (e, s) in summary.items():
        print(f"{p:<18}{e:>12.0f}{s:>10.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        drl_pts = [(e, s) for p, (e, s) in summary.items() if p.startswith("P")]
        drl_pts.sort()
        fig, ax = plt.subplots(figsize=(6, 4.2))
        if drl_pts:
            xs, ys = zip(*drl_pts)
            ax.plot(xs, ys, "o-", color="#1f77b4", label="T2Sem-DRL (energy-pressure sweep)", zorder=3)
        markers = {"random": "s", "greedy": "^", "local_only": "D"}
        colors = {"random": "#7f7f7f", "greedy": "#ff7f0e", "local_only": "#2ca02c"}
        for bl in ["random", "greedy", "local_only"]:
            if bl in summary:
                e, s = summary[bl]
                ax.scatter([e], [s], marker=markers[bl], color=colors[bl], s=70, label=bl, zorder=4)
        ax.set_xlabel("Total energy (per episode)")
        ax.set_ylabel("SLA satisfaction rate")
        ax.set_title("Energy-SLA Pareto frontier")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = out_dir / "pareto_frontier.png"
        fig.savefig(png, dpi=150)
        print(f"Wrote {png}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] {exc}", flush=True)

    print(f"Elapsed: {(time.perf_counter() - t0)/60:.1f} min", flush=True)
    print("PARETO_DONE", flush=True)


if __name__ == "__main__":
    main()

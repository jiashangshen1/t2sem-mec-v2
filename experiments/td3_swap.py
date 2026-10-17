"""DDPG vs TD3 long-agent swap comparison (v4.3 canonical protocol).

Question: does replacing the long-timescale DDPG with TD3 (twin critics +
delayed policy updates + target policy smoothing) change T2Sem-DRL results?

Protocol: configs/default.yaml + configs/improved_v4_pertier.yaml, variant
"full", BC warm-start, per-tier lambda lagrangian -- identical to the frozen
v4.3 revision run -- except drl.long.algo in {ddpg, td3}. All checkpoints,
logs and results are written under results/td3_swap/ so the frozen v4.3
artifacts under results/*_improved_pertier stay untouched.

Usage:
  python experiments/td3_swap.py --smoke          # 1 seed, few episodes
  python experiments/td3_swap.py                  # 3 seeds x 300 ep x 2 algos
  python experiments/td3_swap.py --skip-train     # re-evaluate only
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, train_t2sem, use_long_agent_for_variant
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

OUT_ROOT = "results/td3_swap"


def _results_path(*parts: str) -> Path:
    """Resolve under <repo>/results; refuse any path that escapes it (../ etc.)."""
    base = os.path.abspath(str(project_root() / "results"))
    target = os.path.abspath(os.path.join(base, *parts))
    if not (target == base or target.startswith(base + os.sep)):
        raise ValueError(f"path escapes results dir: {target}")
    return Path(target)


def _write_under_results(parts: list, text: str) -> Path:
    """Validate containment inside <repo>/results first, then write text."""
    target = _results_path(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def evaluate_with_tiers(cfg: dict, seed: int, ckpt_dir: Path, eval_eps: int, device: str):
    """Mirror run_all.evaluate_trained (seed*1000+ep) and also capture per-tier SLA."""
    cfg = dict(cfg)
    cfg["seed"] = seed
    env = build_env(cfg, "full")
    agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
    if (ckpt_dir / "short_ppo.pt").exists():
        agent.load(str(ckpt_dir))
    metrics = []
    tier_rates: dict[str, list[float]] = {}
    for ep in range(eval_eps):
        m = agent.evaluate_episode(seed=seed * 1000 + ep)
        metrics.append(m)
        for name, rate in env.tier_sla_rates().items():
            tier_rates.setdefault(name, []).append(float(rate))
    agg = aggregate_metrics(metrics)
    tier_mean = {k: float(np.mean(v)) for k, v in tier_rates.items()}
    return agg, tier_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=["ddpg", "td3"])
    parser.add_argument("--episodes", type=int, default=None, help="override training episodes")
    parser.add_argument("--eval-eps", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true", help="evaluate existing td3_swap checkpoints")
    parser.add_argument(
        "--log-file",
        default=None,
        help="tee stdout/stderr to this path under results/ (for console-less pythonw runs)",
    )
    args = parser.parse_args()

    if args.log_file:
        # Validate containment inside <repo>/results before opening for write.
        # Accept both "results/x.log" and "x.log" (the helper already roots at results/).
        rel = [p for p in str(args.log_file).replace("\\", "/").split("/") if p not in ("", ".")]
        if rel and rel[0] == "results":
            rel = rel[1:]
        log_path = _results_path(*rel)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream

    paths = ["configs/default.yaml", "configs/improved_v4_pertier.yaml"]
    if args.smoke:
        paths.append("configs/smoke.yaml")
    base_cfg = load_config(*paths)
    device = device_from_config(args.device, base_cfg)

    seeds = args.seeds or [int(s) for s in base_cfg["experiment"]["eval_seeds"]]
    if args.smoke and args.seeds is None:
        seeds = [seeds[0]]
    eval_eps = args.eval_eps or int(base_cfg["experiment"]["eval_episodes"])
    if args.smoke and args.eval_eps is None:
        eval_eps = 2

    rows: list[dict] = []
    t0 = time.perf_counter()
    out_root = "results/td3_swap_smoke" if args.smoke else OUT_ROOT
    for algo in args.algos:
        cfg = copy.deepcopy(base_cfg)
        cfg["drl"]["long"]["algo"] = algo
        if args.episodes is not None:
            cfg["drl"]["total_long_frames"] = args.episodes
        if args.smoke:  # keep the smoke loop fast but still exercise BC + TD3
            cfg["drl"]["bc_demo_episodes"] = 5
            cfg["drl"]["bc_train_epochs"] = 20
            cfg["drl"]["bc_min_samples"] = 32
        cfg["experiment"] = {
            **cfg["experiment"],
            "checkpoint_dir": f"{out_root}/{algo}/checkpoints",
            "log_dir": f"{out_root}/{algo}/logs",
            "results_dir": f"{out_root}/{algo}",
        }
        for seed in seeds:
            ckpt = _results_path(out_root.split("/", 1)[1], algo, "checkpoints", "full", f"seed_{seed}")
            done_marker = ckpt / ".train_complete"
            if not args.skip_train and not done_marker.exists():
                print(f"\n=== train algo={algo} seed={seed} (variant=full) ===")
                train_t2sem(cfg, "full", seed, device)
            elif done_marker.exists():
                print(f"=== resume: training already complete algo={algo} seed={seed} ===")
            print(f"=== eval algo={algo} seed={seed} ({eval_eps} eps) ===")
            agg, tiers = evaluate_with_tiers(cfg, seed, ckpt, eval_eps, device)
            row = {
                "algo": algo,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
            }
            for name in sorted(tiers):
                row[f"sla_{name}"] = tiers[name]
            rows.append(row)
            print(
                f"  {algo} seed={seed}: SLA={row['sla_rate_mean']:.4f} "
                f"delay={row['avg_delay_mean']:.3f} energy={row['total_energy_mean']:.0f} "
                f"crit={row.get('sla_critical', float('nan')):.3f}"
            )

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    csv_path = _write_under_results(["td3_swap", "summary.csv"], buf.getvalue())

    md_lines = [
        "# DDPG vs TD3 long-agent swap (v4.3 protocol)",
        "",
        f"seeds={seeds}, eval_eps={eval_eps}, episodes="
        f"{args.episodes or base_cfg['drl']['total_long_frames']}, device={device}",
        "",
        "| algo | SLA % | std (pp) | delay (s) | energy (J) | critical % | interactive % | best_effort % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algo in args.algos:
        sub = [r for r in rows if r["algo"] == algo]
        if not sub:
            continue
        md_lines.append(
            "| {} | {:.2f} | {:.2f} | {:.3f} | {:.0f} | {:.2f} | {:.2f} | {:.2f} |".format(
                algo,
                100 * float(np.mean([r["sla_rate_mean"] for r in sub])),
                100 * float(np.std([r["sla_rate_mean"] for r in sub])),
                float(np.mean([r["avg_delay_mean"] for r in sub])),
                float(np.mean([r["total_energy_mean"] for r in sub])),
                100 * float(np.mean([r.get("sla_critical", np.nan) for r in sub])),
                100 * float(np.mean([r.get("sla_interactive", np.nan) for r in sub])),
                100 * float(np.mean([r.get("sla_best_effort", np.nan) for r in sub])),
            )
        )
    _write_under_results(["td3_swap", "summary.md"], "\n".join(md_lines) + "\n")
    print(f"\nWrote {csv_path}")
    print("\n".join(md_lines))
    print(f"\nTotal elapsed: {(time.perf_counter() - t0) / 60:.1f} min")
    print("DONE")


if __name__ == "__main__":
    main()

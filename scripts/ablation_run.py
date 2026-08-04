"""P4: Targeted ablation of the two novel components + the CMDP.

Each ablation flips ONE design choice of the full method and retrains, so the
contribution of each component is isolated. Reports overall AND per-tier SLA +
energy (per-tier is the relevant axis for the CMDP/feasibility claims).

Ablations:
  full          : the complete method (reference)
  mask_off      : disable model-informed feasibility region (keep per-tier CMDP)
  single_lambda : single SLA constraint instead of per-tier dual variables
  no_cmdp       : no Lagrangian constraint at all (pure energy-weighted reward)
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

from experiments.train import build_env, train_t2sem, use_long_agent_for_variant
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

# Each ablation: (env variant, dotted reward/drl overrides). Covers the two
# novel components, the CMDP, a cumulative "pure DRL" anchor, and the
# architectural components reused from the run_all variant machinery.
ABLATIONS = {
    # --- reference + novel-component / CMDP ablations (variant=full) ---
    "full": ("full", {}),
    "mask_off": ("full", {"reward.feasibility_mask_enabled": False}),
    "single_lambda": ("full", {"reward.per_tier_lambda": False}),
    "no_cmdp": ("full", {"reward.lagrangian_enabled": False}),
    # --- cumulative anchor: strip all three algorithmic additions ---
    "pure_drl": ("full", {
        "reward.feasibility_mask_enabled": False,
        "reward.per_tier_lambda": False,
        "reward.lagrangian_enabled": False,
    }),
    # --- architectural ablations (reuse run_all variants) ---
    "no_semantic": ("no_semantic", {}),
    "single_timescale": ("single_timescale", {}),
    "no_sla_penalty": ("no_sla_penalty", {}),
    "fixed_mu": ("fixed_mu", {}),
    "no_contention": ("no_contention", {}),
}

# Cumulative build-up order for the additive story (subset of ABLATIONS).
CUMULATIVE = ["pure_drl", "no_cmdp", "single_lambda", "full"]


def apply_overrides(cfg: dict, overrides: dict) -> None:
    for dotted, val in overrides.items():
        sec, key = dotted.split(".", 1)
        cfg.setdefault(sec, {})[key] = val


def eval_drl_full(cfg: dict, variant: str, ckpt: str, seed: int, eval_eps: int, device: str):
    env = build_env(cfg, variant)
    agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant(variant))
    if (Path(ckpt) / "short_ppo.pt").exists():
        agent.load(ckpt)
    metrics, met, total = [], {}, {}
    for ep in range(eval_eps):
        m = agent.evaluate_episode(seed=seed * 1000 + ep)
        metrics.append(m)
        for n, c in env._tier_total.items():
            total[n] = total.get(n, 0) + c
        for n, c in env._tier_met.items():
            met[n] = met.get(n, 0) + c
    return aggregate_metrics(metrics), met, total


def _train_complete(ckpt_dir: Path, episodes: int) -> bool:
    flag = ckpt_dir / ".train_complete"
    if not flag.is_file():
        return False
    for line in flag.read_text(encoding="utf-8").splitlines():
        if line.startswith("episodes="):
            try:
                return int(line.split("=", 1)[1]) >= episodes
            except ValueError:
                return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved_v4_pertier.yaml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--only", nargs="*", default=None, help="subset of ablation labels")
    parser.add_argument("--cumulative", action="store_true", help="run only the additive build-up subset")
    parser.add_argument("--resume", action="store_true", help="skip train if .train_complete matches --episodes")
    parser.add_argument(
        "--suite",
        default="v43",
        help="checkpoint/results suffix (v43 -> checkpoints_ablation_v43)",
    )
    args = parser.parse_args()

    base_cfg = load_config("configs/default.yaml", args.config)
    device = device_from_config(args.device, base_cfg)
    tier_names = [t.name for t in build_env(base_cfg).sla_tiers]
    suffix = "" if args.suite in ("", "default") else f"_{args.suite}"
    out_dir = project_root() / f"results/ablation{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(ABLATIONS.items())
    if args.cumulative:
        items = [(k, ABLATIONS[k]) for k in CUMULATIVE]
    if args.only:
        items = [(k, v) for k, v in ABLATIONS.items() if k in args.only]

    print(f"Device: {device} | ablations={[k for k, _ in items]} seeds={args.seeds} "
          f"ep={args.episodes}", flush=True)

    rows = []
    t0 = time.perf_counter()
    for label, (variant, overrides) in items:
        cfg = copy.deepcopy(base_cfg)
        apply_overrides(cfg, overrides)
        cfg["drl"]["total_long_frames"] = int(args.episodes)
        cfg["experiment"]["eval_seeds"] = list(args.seeds)
        cfg["experiment"]["eval_episodes"] = int(args.eval_episodes)
        cfg["experiment"]["checkpoint_dir"] = f"results/checkpoints_ablation{suffix}/{label}"
        cfg["experiment"]["results_dir"] = f"results/ablation{suffix}/{label}"

        agg_met, agg_tot = {}, {}
        sla_seeds, energy_seeds = [], []
        for seed in args.seeds:
            ckpt_dir = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
            if args.resume and _train_complete(ckpt_dir, int(args.episodes)):
                print(f"\n[{label}] skip-train seed={seed} (complete)", flush=True)
                ckpt = str(ckpt_dir)
            else:
                print(f"\n[{label}] variant={variant} train seed={seed} overrides={overrides}", flush=True)
                ckpt = train_t2sem(cfg, variant, seed, device)
            agg, met, total = eval_drl_full(cfg, variant, ckpt, seed, int(args.eval_episodes), device)
            sla_seeds.append(agg["sla_rate_mean"])
            energy_seeds.append(agg["total_energy_mean"])
            for n in tier_names:
                agg_met[n] = agg_met.get(n, 0) + met.get(n, 0)
                agg_tot[n] = agg_tot.get(n, 0) + total.get(n, 0)
            print(f"  [eval] {label} seed={seed} SLA={agg['sla_rate_mean']:.3f} "
                  f"energy={agg['total_energy_mean']:.0f}", flush=True)

        row = {
            "ablation": label,
            "overall_sla": round(float(np.mean(sla_seeds)), 4),
            "energy": round(float(np.mean(energy_seeds)), 1),
        }
        for n in tier_names:
            row[n] = round(agg_met[n] / agg_tot[n], 4) if agg_tot.get(n, 0) else float("nan")
        rows.append(row)

    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    header = f"{'ablation':<16}{'overall':>10}{'energy':>12}" + "".join(f"{n:>14}" for n in tier_names)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['ablation']:<16}{r['overall_sla']:>10.3f}{r['energy']:>12.0f}"
              + "".join(f"{r[n]:>14.3f}" for n in tier_names))
    print("=" * len(header))
    print(f"\nWrote {csv_path}")
    print(f"Elapsed: {(time.perf_counter() - t0)/60:.1f} min", flush=True)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()

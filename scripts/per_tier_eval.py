"""P1: Per-tier SLA breakdown for the trained method vs all baselines.

Loads existing checkpoints (no retraining) and reports SLA satisfaction rate
PER SLA tier (critical / interactive / best_effort), which is the direct
evidence for the per-tier CMDP contribution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, use_long_agent_for_variant
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config

BASELINES = ["greedy", "tjcct", "random", "local_only"]


def _make_policy(name: str, env, seed: int):
    if name == "greedy":
        from src.baselines.greedy import GreedyPolicy

        return GreedyPolicy(env)
    if name == "local_only":
        from src.baselines.local_only import LocalOnlyPolicy

        return LocalOnlyPolicy(env)
    if name == "random":
        from src.baselines.random_policy import RandomPolicy

        return RandomPolicy(env, seed=seed)
    if name == "tjcct":
        from src.baselines.tjcct import TJCCTHeuristicPolicy

        return TJCCTHeuristicPolicy(env)
    raise ValueError(name)


def _accumulate(env, met: dict, total: dict) -> None:
    for name, n in env._tier_total.items():
        total[name] = total.get(name, 0) + n
    for name, n in env._tier_met.items():
        met[name] = met.get(name, 0) + n


def eval_drl(cfg: dict, seeds, eval_eps: int, device: str):
    met: dict = {}
    total: dict = {}
    for seed in seeds:
        c = dict(cfg)
        c["seed"] = seed
        env = build_env(c, "full")
        agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
        if (ckpt / "short_ppo.pt").exists():
            agent.load(str(ckpt))
        else:
            print(f"  [warn] missing checkpoint for seed={seed}: {ckpt}")
        for ep in range(eval_eps):
            agent.evaluate_episode(seed=seed * 1000 + ep)
            _accumulate(env, met, total)
    return met, total


def eval_baseline(cfg: dict, name: str, seeds, eval_eps: int, mask: bool = True):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("reward", {})["feasibility_mask_enabled"] = bool(mask)
    met: dict = {}
    total: dict = {}
    for seed in seeds:
        env = build_env(cfg)
        policy = _make_policy(name, env, seed)
        for ep in range(eval_eps):
            obs, _ = env.reset(seed=seed * 1000 + ep)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy.act(obs))
                done = term or trunc
            _accumulate(env, met, total)
    return met, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved_v4_pertier.yaml")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-mask", choices=["on", "off"], default="on",
                        help="off => framework-A vanilla baselines (no feasibility mask)")
    parser.add_argument("--skip-drl", action="store_true")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml", args.config)
    device = device_from_config(args.device, cfg)
    seeds = list(cfg["experiment"]["eval_seeds"])
    eval_eps = args.eval_episodes or int(cfg["experiment"]["eval_episodes"])
    tier_names = [t.name for t in build_env(cfg).sla_tiers]

    print(f"Per-tier SLA | seeds={seeds} eval_eps={eval_eps} device={device}")

    mask_on = args.baseline_mask == "on"
    results = {}
    if not args.skip_drl:
        print("Evaluating T2Sem-DRL (loading checkpoints)...")
        results["T2Sem-DRL"] = eval_drl(cfg, seeds, eval_eps, device)
    suffix = "" if mask_on else " (mask-off)"
    for bl in BASELINES:
        print(f"Evaluating baseline: {bl}{suffix}...")
        results[bl + suffix] = eval_baseline(cfg, bl, seeds, eval_eps, mask=mask_on)

    def rate(met, total, name):
        t = total.get(name, 0)
        return met.get(name, 0) / t if t > 0 else float("nan")

    header = f"{'method':<16}" + "".join(f"{n:>14}" for n in tier_names) + f"{'overall':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    rows = []
    for method, (met, total) in results.items():
        all_met = sum(met.values())
        all_tot = sum(total.values())
        overall = all_met / all_tot if all_tot else float("nan")
        line = f"{method:<16}" + "".join(f"{rate(met, total, n):>14.3f}" for n in tier_names) + f"{overall:>12.3f}"
        print(line)
        row = {"method": method, "overall": round(overall, 4)}
        for n in tier_names:
            row[n] = round(rate(met, total, n), 4)
            row[f"{n}_n"] = total.get(n, 0)
        rows.append(row)
    print("=" * len(header))

    fname = "per_tier_sla.csv" if mask_on else "per_tier_sla_baseline_maskoff.csv"
    out = project_root() / cfg["experiment"]["results_dir"] / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    print("PER_TIER_DONE")


if __name__ == "__main__":
    main()

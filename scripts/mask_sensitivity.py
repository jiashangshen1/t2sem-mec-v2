"""P4 (Tier 3): test-time sensitivity of the feasibility mask -- NO retraining.

Loads the already-trained full-method checkpoints and re-evaluates the SAME
policy under different mask settings (delay margin alpha, channel estimate,
similarity margin). Because the mask runs at decision time, these are pure
eval-time knobs, so this is cheap and shows that the SLA>0.8 result is driven
by the margin/channel design choices.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, use_long_agent_for_variant
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

DEFAULT_ALPHA = 0.8
DEFAULT_CHANNEL = "expected"
DEFAULT_SIM = 0.05


def _set_mask(env, alpha, channel, sim):
    env.mask_delay_margin = float(alpha)
    env.mask_channel = str(channel)
    env.mask_sim_margin = float(sim)
    env._min_mu_by_tier = {t.name: env._min_mu_for_sim(t.s_min) for t in env.sla_tiers}


def _eval(agents, alpha, channel, sim, eval_eps, tier_names):
    met, total, sla, energy = {}, {}, [], []
    for env, agent, seed in agents:
        _set_mask(env, alpha, channel, sim)
        ms = []
        for ep in range(eval_eps):
            ms.append(agent.evaluate_episode(seed=seed * 1000 + ep))
            for n, c in env._tier_total.items():
                total[n] = total.get(n, 0) + c
            for n, c in env._tier_met.items():
                met[n] = met.get(n, 0) + c
        agg = aggregate_metrics(ms)
        sla.append(agg["sla_rate_mean"])
        energy.append(agg["total_energy_mean"])
    row = {"overall_sla": round(float(np.mean(sla)), 4), "energy": round(float(np.mean(energy)), 1)}
    for n in tier_names:
        row[n] = round(met.get(n, 0) / total[n], 4) if total.get(n, 0) else float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved_v4_pertier.yaml")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml", args.config)
    device = device_from_config(args.device, cfg)
    seeds = args.seeds or list(cfg["experiment"]["eval_seeds"])
    eval_eps = int(args.eval_episodes)
    tier_names = [t.name for t in build_env(cfg).sla_tiers]

    print(f"Mask sensitivity (no retrain) | seeds={seeds} eval={eval_eps} device={device}", flush=True)

    # Build env+agent once per seed; reuse across all mask settings.
    agents = []
    for seed in seeds:
        c = dict(cfg)
        c["seed"] = seed
        env = build_env(c, "full")
        agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
        if (ckpt / "short_ppo.pt").exists():
            agent.load(str(ckpt))
        else:
            print(f"  [warn] missing checkpoint seed={seed}: {ckpt}", flush=True)
        agents.append((env, agent, seed))

    rows = []
    sweeps = (
        [("delay_margin", a, DEFAULT_CHANNEL, DEFAULT_SIM) for a in (0.6, 0.7, 0.8, 0.9, 1.0)]
        + [("channel", DEFAULT_ALPHA, ch, DEFAULT_SIM) for ch in ("los", "expected", "nlos")]
        + [("sim_margin", DEFAULT_ALPHA, DEFAULT_CHANNEL, s) for s in (0.0, 0.05, 0.1, 0.15)]
    )
    for dim, alpha, channel, sim in sweeps:
        r = _eval(agents, alpha, channel, sim, eval_eps, tier_names)
        value = {"delay_margin": alpha, "channel": channel, "sim_margin": sim}[dim]
        rows.append({"dim": dim, "value": value, "alpha": alpha, "channel": channel,
                     "sim_margin": sim, **r})
        print(f"  [{dim}={value}] SLA={r['overall_sla']:.3f} energy={r['energy']:.0f} "
              + " ".join(f"{n}={r[n]:.3f}" for n in tier_names), flush=True)

    out = project_root() / cfg["experiment"]["results_dir"] / "mask_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    print("MASK_SENS_DONE", flush=True)


if __name__ == "__main__":
    main()

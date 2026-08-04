"""S-tier reviewer-critical experiments.

1. semantic_mappo full train+eval on extra seeds {789, 1011}
2. critical_priority per-tier SLA (+ masked baselines)
3. semantic_mappo mask-off eval on canonical seeds {42, 123, 456}
4. refresh supplement_summary.csv + bootstrap stats
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, run_baseline, train_t2sem, use_long_agent_for_variant
from experiments.run_all import evaluate_trained
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

CANONICAL_SEEDS = [42, 123, 456]
EXTRA_SEEDS = [789, 1011]
ALL_SEEDS = CANONICAL_SEEDS + EXTRA_SEEDS

SEMANTIC_CONFIG = "configs/improved_v4_semantic_mappo.yaml"
CRITICAL_CONFIG = "configs/improved_v4_critical_priority.yaml"
OUT_DIR = project_root() / "results" / "supplement_v43"


def _row_key(row: dict) -> tuple[str, int]:
    return (str(row["job"]), int(row["seed"]))


def _load_csv(path: Path) -> dict[tuple[str, int], dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {_row_key(r): r for r in csv.DictReader(f)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _merge_rows(existing: dict[tuple[str, int], dict], new_rows: list[dict]) -> list[dict]:
    merged = dict(existing)
    for row in new_rows:
        merged[_row_key(row)] = row
    return sorted(merged.values(), key=lambda r: (r["job"], int(r["seed"])))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_fresh_train(ckpt: Path) -> None:
    if (ckpt / ".train_complete").is_file():
        return
    if ckpt.exists():
        _log(f"[retrain] removing incomplete checkpoint: {ckpt}")
        shutil.rmtree(ckpt)


def _eval_row(job: str, cfg: dict, variant: str, seed: int, eval_eps: int, device: str, note: str = "") -> dict:
    ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
    metrics = evaluate_trained(cfg, variant, seed, str(ckpt), eval_eps, device)
    agg = aggregate_metrics(metrics)
    return {
        "job": job,
        "variant": variant,
        "seed": seed,
        "sla_rate_mean": agg["sla_rate_mean"],
        "sla_rate_std": agg["sla_rate_std"],
        "avg_delay_mean": agg["avg_delay_mean"],
        "total_energy_mean": agg["total_energy_mean"],
        "note": note,
    }


def _accumulate(env, met: dict, total: dict) -> None:
    for name, n in env._tier_total.items():
        total[name] = total.get(name, 0) + n
    for name, n in env._tier_met.items():
        met[name] = met.get(name, 0) + n


def _tier_rate(met: dict, total: dict, name: str) -> float:
    t = total.get(name, 0)
    return met.get(name, 0) / t if t > 0 else float("nan")


def run_semantic_train_eval(seeds: list[int], eval_eps: int, device: str) -> list[dict]:
    cfg = load_config("configs/default.yaml", SEMANTIC_CONFIG)
    variant = "full"
    rows: list[dict] = []
    for seed in seeds:
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
        _ensure_fresh_train(ckpt)
        if not (ckpt / ".train_complete").is_file():
            _log(f"[train] semantic_mappo seed={seed}")
            train_t2sem(cfg, variant, seed, device)
        row = _eval_row("semantic_mappo", cfg, variant, seed, eval_eps, device, note="s_tier_full_train")
        rows.append(row)
        _log(f"[eval] semantic_mappo seed={seed} SLA={row['sla_rate_mean']:.3f}")
    return rows


def run_mask_off_eval(seeds: list[int], eval_eps: int, device: str) -> list[dict]:
    cfg = load_config("configs/default.yaml", SEMANTIC_CONFIG)
    cfg.setdefault("reward", {})["feasibility_mask_enabled"] = False
    variant = "full"
    rows: list[dict] = []
    tier_names = [t.name for t in build_env(cfg).sla_tiers]
    for seed in seeds:
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
        if not (ckpt / "short_ppo.pt").exists():
            _log(f"[skip] semantic_mappo_mask_off seed={seed} (missing ckpt)")
            continue
        env = build_env(cfg, variant)
        agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant(variant))
        agent.load(str(ckpt))
        metrics = []
        for ep in range(eval_eps):
            metrics.append(agent.evaluate_episode(seed=seed * 1000 + ep))
        agg = aggregate_metrics(metrics)
        rows.append(
            {
                "job": "semantic_mappo_mask_off",
                "variant": variant,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
                "note": "mask_off_at_eval",
            }
        )
        _log(f"[eval] semantic_mappo_mask_off seed={seed} SLA={agg['sla_rate_mean']:.3f}")
    return rows


def run_critical_per_tier(seeds: list[int], eval_eps: int, device: str) -> None:
    cfg = load_config("configs/default.yaml", CRITICAL_CONFIG)
    tier_names = [t.name for t in build_env(cfg).sla_tiers]
    out = OUT_DIR / "per_tier_critical_priority.csv"
    rows: list[dict] = []

    # DRL
    met, total = {}, {}
    for seed in seeds:
        _log(f"[per-tier] critical_priority seed={seed} ...")
        env = build_env(cfg, "full")
        agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
        ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / "full" / f"seed_{seed}"
        agent.load(str(ckpt))
        for ep in range(eval_eps):
            agent.evaluate_episode(seed=seed * 1000 + ep)
            _accumulate(env, met, total)
    overall = sum(met.values()) / sum(total.values()) if sum(total.values()) else float("nan")
    row = {"method": "critical_priority", "overall": round(overall, 4)}
    for n in tier_names:
        row[n] = round(_tier_rate(met, total, n), 4)
        row[f"{n}_n"] = total.get(n, 0)
    rows.append(row)
    _log(f"[per-tier] critical_priority overall={overall:.3f}")

    # masked greedy baseline on same config
    met, total = {}, {}
    for seed in seeds:
        _log(f"[per-tier] greedy_critical_priority seed={seed} ...")
        env = build_env(cfg)
        from src.baselines.greedy import GreedyPolicy

        policy = GreedyPolicy(env)
        for ep in range(eval_eps):
            obs, _ = env.reset(seed=seed * 1000 + ep)
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(policy.act(obs))
                done = term or trunc
            _accumulate(env, met, total)
    overall = sum(met.values()) / sum(total.values()) if sum(total.values()) else float("nan")
    row = {"method": "greedy_critical_priority", "overall": round(overall, 4)}
    for n in tier_names:
        row[n] = round(_tier_rate(met, total, n), 4)
        row[f"{n}_n"] = total.get(n, 0)
    rows.append(row)
    _log(f"[per-tier] greedy_critical_priority overall={overall:.3f}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _log(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated: semantic_train,mask_off,per_tier,bootstrap",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "supplement_summary.csv"
    existing = _load_csv(csv_path)
    device = device_from_config(args.device, load_config("configs/default.yaml", SEMANTIC_CONFIG))
    t0 = time.time()
    new_rows: list[dict] = []

    def want(name: str) -> bool:
        return args.only is None or name in args.only.split(",")

    if want("semantic_train") and not args.skip_train:
        new_rows.extend(run_semantic_train_eval(EXTRA_SEEDS, args.eval_episodes, device))
    elif want("semantic_train") and args.skip_train:
        cfg = load_config("configs/default.yaml", SEMANTIC_CONFIG)
        for seed in EXTRA_SEEDS:
            row = _eval_row("semantic_mappo", cfg, "full", seed, args.eval_episodes, device, note="s_tier_eval_only")
            new_rows.append(row)

    if want("mask_off"):
        new_rows.extend(run_mask_off_eval(CANONICAL_SEEDS, args.eval_episodes, device))

    if want("per_tier"):
        run_critical_per_tier(CANONICAL_SEEDS, args.eval_episodes, device)

    if new_rows:
        merged = _merge_rows(existing, new_rows)
        _write_csv(csv_path, merged)
        _log(f"Updated {csv_path} (+{len(new_rows)} rows, total {len(merged)})")

    if want("bootstrap"):
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bootstrap_stats.py")],
            check=False,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sync_supplement_to_paper.py")],
            check=False,
        )

    _log(f"S_TIER_DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

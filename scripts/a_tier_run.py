"""A-tier experiments (after S-tier): deployment triangle, scale-up semantic, load sweep, mask sensitivity."""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import build_env, run_baseline, use_long_agent_for_variant
from src.agents.t2sem_drl import T2SemDRL
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

CANONICAL_SEEDS = [42, 123, 456]
OUT_DIR = project_root() / "results" / "supplement_v43"

MAIN_CONFIG = "configs/improved_v4_pertier.yaml"
SEMANTIC_CONFIG = "configs/improved_v4_semantic_mappo.yaml"
CRITICAL_CONFIG = "configs/improved_v4_critical_priority.yaml"
SCALEUP_CONFIG = "configs/improved_v4_scaleup.yaml"

LOAD_RATES = [0.15, 0.25, 0.35]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    _log(f"Wrote {path} ({len(rows)} rows)")


def _eval_drl_on_cfg(
    job: str,
    cfg: dict,
    ckpt_dir: Path,
    seeds: list[int],
    eval_eps: int,
    device: str,
    note: str,
    short_algo: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    variant = "full"
    for seed in seeds:
        ckpt = ckpt_dir / variant / f"seed_{seed}"
        if not (ckpt / "short_ppo.pt").exists():
            _log(f"[skip] {job} seed={seed} missing {ckpt}")
            continue
        cfg_eval = copy.deepcopy(cfg)
        if short_algo:
            cfg_eval.setdefault("drl", {}).setdefault("short", {})["algo"] = short_algo
        env = build_env(cfg_eval, variant)
        agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant(variant))
        try:
            agent.load(str(ckpt))
        except RuntimeError as e:
            _log(f"[skip] {job} seed={seed} load failed: {e}")
            continue
        metrics = [agent.evaluate_episode(seed=seed * 1000 + ep) for ep in range(eval_eps)]
        agg = aggregate_metrics(metrics)
        rows.append(
            {
                "job": job,
                "variant": variant,
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
                "note": note,
            }
        )
        _log(f"[eval] {job} seed={seed} SLA={agg['sla_rate_mean']:.3f}")
    return rows


def run_deployment_triangle(seeds: list[int], eval_eps: int, device: str) -> list[dict]:
    """Critical-priority config: greedy, semantic transfer, t2sem transfer, trained critical_priority."""
    cfg = load_config("configs/default.yaml", CRITICAL_CONFIG)
    rows: list[dict] = []

    for seed in seeds:
        env = build_env(cfg)
        env.cfg["seed"] = seed
        agg = aggregate_metrics(run_baseline(env, "greedy", eval_eps))
        rows.append(
            {
                "job": "deploy_greedy_critical",
                "variant": "baseline",
                "seed": seed,
                "sla_rate_mean": agg["sla_rate_mean"],
                "sla_rate_std": agg["sla_rate_std"],
                "avg_delay_mean": agg["avg_delay_mean"],
                "total_energy_mean": agg["total_energy_mean"],
                "note": "critical_priority_config",
            }
        )
        _log(f"[deploy] greedy seed={seed} SLA={agg['sla_rate_mean']:.3f}")

    rows.extend(
        _eval_drl_on_cfg(
            "deploy_semantic_critical",
            cfg,
            project_root() / "results/checkpoints_semantic_mappo",
            seeds,
            eval_eps,
            device,
            "canonical_ckpt_on_critical_config",
            short_algo="independent_ppo",
        )
    )
    rows.extend(
        _eval_drl_on_cfg(
            "deploy_t2sem_critical",
            cfg,
            project_root() / "results/checkpoints_improved_pertier",
            seeds,
            eval_eps,
            device,
            "canonical_ckpt_on_critical_config",
        )
    )
    rows.extend(
        _eval_drl_on_cfg(
            "deploy_critical_priority",
            cfg,
            project_root() / "results/checkpoints_critical_priority",
            seeds,
            eval_eps,
            device,
            "critical_priority_trained",
        )
    )
    return rows


def run_scaleup_semantic(seeds: list[int], eval_eps: int, device: str) -> list[dict]:
    cfg = load_config("configs/default.yaml", SCALEUP_CONFIG)
    return _eval_drl_on_cfg(
        "scaleup_semantic_mappo",
        cfg,
        project_root() / "results/checkpoints_semantic_mappo",
        seeds,
        eval_eps,
        device,
        "canonical_ckpt_on_scaleup_topology",
        short_algo="independent_ppo",
    )


def run_load_sweep(seeds: list[int], eval_eps: int, device: str) -> None:
    rows: list[dict] = []
    base_cfg = load_config("configs/default.yaml", MAIN_CONFIG)
    ckpt_roots = {
        "t2sem_full": project_root() / "results/checkpoints_improved_pertier",
        "semantic_mappo": project_root() / "results/checkpoints_semantic_mappo",
    }

    for rate in LOAD_RATES:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("task", {})["arrival_rate"] = float(rate)
        for seed in seeds:
            env = build_env(cfg)
            env.cfg["seed"] = seed
            agg = aggregate_metrics(run_baseline(env, "greedy", eval_eps))
            rows.append(
                {
                    "arrival_rate": rate,
                    "method": "greedy",
                    "seed": seed,
                    "sla_rate_mean": agg["sla_rate_mean"],
                    "total_energy_mean": agg["total_energy_mean"],
                }
            )
            _log(f"[load] rate={rate} greedy seed={seed} SLA={agg['sla_rate_mean']:.3f}")

        for method, ckpt_root in ckpt_roots.items():
            for seed in seeds:
                ckpt = ckpt_root / "full" / f"seed_{seed}"
                if not (ckpt / "short_ppo.pt").exists():
                    continue
                cfg_eval = copy.deepcopy(cfg)
                if method == "semantic_mappo":
                    cfg_eval.setdefault("drl", {}).setdefault("short", {})["algo"] = "independent_ppo"
                env = build_env(cfg_eval, "full")
                agent = T2SemDRL(env, device=device, use_long_agent=use_long_agent_for_variant("full"))
                try:
                    agent.load(str(ckpt))
                except RuntimeError as e:
                    _log(f"[skip] load-sweep {method} seed={seed} load failed: {e}")
                    continue
                metrics = [agent.evaluate_episode(seed=seed * 1000 + ep) for ep in range(eval_eps)]
                agg = aggregate_metrics(metrics)
                rows.append(
                    {
                        "arrival_rate": rate,
                        "method": method,
                        "seed": seed,
                        "sla_rate_mean": agg["sla_rate_mean"],
                        "total_energy_mean": agg["total_energy_mean"],
                    }
                )
                _log(f"[load] rate={rate} {method} seed={seed} SLA={agg['sla_rate_mean']:.3f}")

    out = OUT_DIR / "load_sweep.csv"
    _write_csv(out, rows)

    summary: list[dict] = []
    for rate in LOAD_RATES:
        for method in ("greedy", "t2sem_full", "semantic_mappo"):
            sub = [float(r["sla_rate_mean"]) * 100 for r in rows if float(r["arrival_rate"]) == rate and r["method"] == method]
            if sub:
                summary.append(
                    {
                        "arrival_rate": rate,
                        "method": method,
                        "n": len(sub),
                        "sla_mean_pct": round(float(np.mean(sub)), 2),
                        "sla_std_pct": round(float(np.std(sub)), 2),
                    }
                )
    _write_csv(OUT_DIR / "load_sweep_summary.csv", summary)


def run_mask_sensitivity_extra(device: str) -> None:
    for cfg_name, tag in (
        (CRITICAL_CONFIG, "critical_priority"),
        (SCALEUP_CONFIG, "scaleup"),
    ):
        _log(f"[mask] running sensitivity on {tag}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "mask_sensitivity.py"),
                "--config",
                cfg_name,
                "--eval-episodes",
                "10",
                "--device",
                device,
            ],
            check=False,
        )
        src = project_root() / load_config("configs/default.yaml", cfg_name)["experiment"]["results_dir"] / "mask_sensitivity.csv"
        if src.is_file():
            dst = OUT_DIR / f"mask_sensitivity_{tag}.csv"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            _log(f"Copied {src} -> {dst}")


def merge_into_supplement(new_rows: list[dict]) -> None:
    csv_path = OUT_DIR / "supplement_summary.csv"
    existing = {(r["job"], int(r["seed"])): r for r in _load_csv(csv_path)}
    for row in new_rows:
        existing[(row["job"], int(row["seed"]))] = row
    merged = sorted(existing.values(), key=lambda r: (r["job"], int(r["seed"])))
    _write_csv(csv_path, merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument(
        "--only",
        default=None,
        help="deploy,scaleup,load,mask,bootstrap",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = device_from_config(args.device, load_config("configs/default.yaml", MAIN_CONFIG))
    t0 = time.time()
    new_rows: list[dict] = []

    def want(name: str) -> bool:
        return args.only is None or name in args.only.split(",")

    if want("deploy"):
        new_rows.extend(run_deployment_triangle(CANONICAL_SEEDS, args.eval_episodes, device))

    if want("scaleup"):
        new_rows.extend(run_scaleup_semantic(CANONICAL_SEEDS, args.eval_episodes, device))

    if new_rows:
        merge_into_supplement(new_rows)

    if want("load"):
        run_load_sweep(CANONICAL_SEEDS, args.eval_episodes, device)

    if want("mask"):
        run_mask_sensitivity_extra(device)

    if want("bootstrap"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap_stats.py")], check=False)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_supplement_to_paper.py")], check=False)

    _log(f"A_TIER_DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Supplementary experiments for reviewer concerns M4/M5/M7 + extra seeds.

Jobs:
  - semantic_mappo: decentralized PPO + shared F (P1); reuses independent_ppo ckpt if present
  - extra seeds (789, 1011): greedy + t2sem_full under canonical config
  - scaleup: greedy + t2sem_full on 16-UE / 5-UAV topology
  - critical_priority: higher critical CMDP target + tier weight
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

from experiments.train import build_env, run_baseline, train_t2sem
from experiments.run_all import evaluate_trained
from src.utils.config import load_config, project_root
from src.utils.device import device_from_config
from src.utils.metrics import aggregate_metrics

CANONICAL_SEEDS = [42, 123, 456]
EXTRA_SEEDS = [789, 1011]
ALL_SEEDS = CANONICAL_SEEDS + EXTRA_SEEDS

MAIN_CONFIG = "configs/improved_v4_pertier.yaml"

DRL_JOBS = {
    "t2sem_full": ("full", MAIN_CONFIG),
    "semantic_mappo": ("full", "configs/improved_v4_semantic_mappo.yaml"),
    "scaleup_t2sem": ("full", "configs/improved_v4_scaleup.yaml"),
    "critical_priority": ("full", "configs/improved_v4_critical_priority.yaml"),
}

CKPT_ALIAS = {
    "semantic_mappo": "results/checkpoints_independent_ppo",
}


def _row_key(row: dict) -> tuple[str, int]:
    return (str(row["job"]), int(row["seed"]))


def _load_existing_rows(csv_path: Path) -> dict[tuple[str, int], dict]:
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {_row_key(r): r for r in csv.DictReader(f)}


def _merge_rows(existing: dict[tuple[str, int], dict], new_rows: list[dict]) -> list[dict]:
    merged = dict(existing)
    for row in new_rows:
        merged[_row_key(row)] = row
    return sorted(merged.values(), key=lambda r: (r["job"], int(r["seed"])))


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load_cfg(*paths: str, smoke: bool = False) -> dict:
    if smoke:
        return load_config("configs/default.yaml", "configs/smoke.yaml", *paths)
    return load_config("configs/default.yaml", *paths)


def _resolve_ckpt(job: str, cfg: dict, variant: str, seed: int) -> Path:
    ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
    if (ckpt / "short_ppo.pt").exists():
        return ckpt
    alias_root = CKPT_ALIAS.get(job)
    if alias_root:
        alt = project_root() / alias_root / variant / f"seed_{seed}"
        if (alt / "short_ppo.pt").exists():
            return alt
    return ckpt


def _maybe_copy_alias_ckpt(job: str, cfg: dict, variant: str, seed: int) -> Path | None:
    """Copy independent_ppo checkpoint into semantic_mappo dir for reproducibility."""
    if job != "semantic_mappo":
        return None
    dst = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
    if (dst / "short_ppo.pt").exists():
        return dst
    src = project_root() / CKPT_ALIAS["semantic_mappo"] / variant / f"seed_{seed}"
    if not (src / "short_ppo.pt").exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / ".from_independent_ppo").write_text(str(src), encoding="utf-8")
    return dst


def _eval_drl(
    job: str,
    cfg: dict,
    variant: str,
    seed: int,
    eval_eps: int,
    device: str,
) -> dict | None:
    ckpt = _resolve_ckpt(job, cfg, variant, seed)
    if not (ckpt / "short_ppo.pt").exists():
        print(f"[skip-eval] {job} seed={seed} missing checkpoint: {ckpt}")
        return None
    metrics = evaluate_trained(cfg, variant, seed, str(ckpt), eval_eps, device)
    agg = aggregate_metrics(metrics)
    note = ""
    alias = project_root() / CKPT_ALIAS.get(job, "") / variant / f"seed_{seed}"
    if job in CKPT_ALIAS and ckpt.resolve() == alias.resolve():
        note = "ckpt_from_independent_ppo"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("quick", "train", "all"),
        default="all",
        help="quick=eval-only baselines+semantic; train=DRL jobs; all=both",
    )
    parser.add_argument("--only", default=None, help="comma-separated job names")
    args = parser.parse_args()

    out_dir = project_root() / "results" / ("supplement_smoke" if args.smoke else "supplement_v43")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "supplement_summary.csv"
    existing = _load_existing_rows(csv_path) if args.resume else {}
    eval_eps = 5 if args.smoke else 30
    episodes = 15 if args.smoke else None
    seeds_main = [42] if args.smoke else CANONICAL_SEEDS
    seeds_extra = [] if args.smoke else EXTRA_SEEDS
    device = device_from_config(args.device, _load_cfg(MAIN_CONFIG))

    rows: list[dict] = []
    t0 = time.time()

    def want(name: str) -> bool:
        return args.only is None or name in args.only.split(",")

    def done(job: str, seed: int) -> bool:
        return args.resume and (job, seed) in existing

    # --- Quick: extra-seed baselines on canonical config ---
    if args.phase in ("quick", "all"):
        cfg_main = _load_cfg(MAIN_CONFIG, smoke=args.smoke)
        for bl in ("greedy", "tjcct"):
            if not want(bl):
                continue
            for seed in seeds_extra:
                if done(bl, seed):
                    print(f"[skip] {bl} seed={seed} (in supplement_summary.csv)")
                    continue
                env = build_env(cfg_main)
                env.cfg["seed"] = seed
                agg = aggregate_metrics(run_baseline(env, bl, eval_eps))
                rows.append(
                    {
                        "job": bl,
                        "variant": "baseline",
                        "seed": seed,
                        "sla_rate_mean": agg["sla_rate_mean"],
                        "sla_rate_std": agg["sla_rate_std"],
                        "avg_delay_mean": agg["avg_delay_mean"],
                        "total_energy_mean": agg["total_energy_mean"],
                        "note": "extra_seed_P1",
                    }
                )
                print(f"[baseline] {bl} seed={seed} SLA={agg['sla_rate_mean']:.3f}")

        # semantic_mappo eval (reuse independent ckpt when available; never smoke-resize env)
        if want("semantic_mappo"):
            spec = DRL_JOBS["semantic_mappo"]
            cfg = _load_cfg(spec[1], smoke=False)
            if episodes:
                cfg["drl"]["total_long_frames"] = episodes
            for seed in seeds_main + seeds_extra:
                if done("semantic_mappo", seed):
                    print(f"[skip] semantic_mappo seed={seed} (in supplement_summary.csv)")
                    continue
                if not args.skip_train:
                    ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / spec[0] / f"seed_{seed}"
                    alias = project_root() / CKPT_ALIAS["semantic_mappo"] / spec[0] / f"seed_{seed}"
                    has_ckpt = (ckpt / "short_ppo.pt").exists() or (alias / "short_ppo.pt").exists()
                    if seed in EXTRA_SEEDS and not has_ckpt:
                        print(f"[skip] semantic_mappo seed={seed} (no ckpt; extra seeds optional)")
                        continue
                    need_train = not has_ckpt and not (ckpt / ".train_complete").is_file()
                    if need_train:
                        print(f"[train] semantic_mappo seed={seed}")
                        train_t2sem(cfg, spec[0], seed, device)
                    elif (alias / "short_ppo.pt").exists():
                        _maybe_copy_alias_ckpt("semantic_mappo", cfg, spec[0], seed)
                row = _eval_drl("semantic_mappo", cfg, spec[0], seed, eval_eps, device)
                if row is None:
                    continue
                rows.append(row)
                print(f"[eval] semantic_mappo seed={seed} SLA={row['sla_rate_mean']:.3f} {row['note']}")

    # --- Train phase: DRL jobs + scaleup baselines ---
    if args.phase in ("train", "all"):
        # scaleup greedy baseline
        if want("scaleup_greedy"):
            cfg_su = _load_cfg("configs/improved_v4_scaleup.yaml", smoke=False)
            for seed in seeds_main:
                if done("scaleup_greedy", seed):
                    print(f"[skip] scaleup_greedy seed={seed} (in supplement_summary.csv)")
                    continue
                env = build_env(cfg_su)
                env.cfg["seed"] = seed
                agg = aggregate_metrics(run_baseline(env, "greedy", eval_eps))
                rows.append(
                    {
                        "job": "scaleup_greedy",
                        "variant": "baseline",
                        "seed": seed,
                        "sla_rate_mean": agg["sla_rate_mean"],
                        "sla_rate_std": agg["sla_rate_std"],
                        "avg_delay_mean": agg["avg_delay_mean"],
                        "total_energy_mean": agg["total_energy_mean"],
                        "note": "scaleup_topology",
                    }
                )
                print(f"[baseline] scaleup_greedy seed={seed} SLA={agg['sla_rate_mean']:.3f}")

        train_jobs = {k: v for k, v in DRL_JOBS.items() if k != "semantic_mappo" or args.phase == "train"}
        for job_name, (variant, cfg_path) in train_jobs.items():
            if not want(job_name):
                continue
            cfg = _load_cfg(cfg_path, smoke=False)
            if args.smoke and episodes:
                cfg["drl"]["total_long_frames"] = episodes

            job_seeds = seeds_main if job_name != "t2sem_full" else seeds_main + seeds_extra

            if not args.skip_train:
                for seed in job_seeds:
                    ckpt = project_root() / cfg["experiment"]["checkpoint_dir"] / variant / f"seed_{seed}"
                    if args.resume and (ckpt / ".train_complete").is_file():
                        print(f"[skip-train] {job_name} seed={seed}")
                        continue
                    if job_name == "semantic_mappo":
                        alias = project_root() / CKPT_ALIAS["semantic_mappo"] / variant / f"seed_{seed}"
                        if (alias / "short_ppo.pt").exists():
                            _maybe_copy_alias_ckpt(job_name, cfg, variant, seed)
                            continue
                    print(f"[train] {job_name} seed={seed}")
                    train_t2sem(cfg, variant, seed, device)

            for seed in job_seeds:
                if done(job_name, seed):
                    print(f"[skip] {job_name} seed={seed} (in supplement_summary.csv)")
                    continue
                row = _eval_drl(job_name, cfg, variant, seed, eval_eps, device)
                if row is None:
                    continue
                if job_name == "scaleup_t2sem":
                    row["note"] = "scaleup_topology"
                elif job_name == "critical_priority":
                    row["note"] = "critical_priority_variant"
                elif job_name == "t2sem_full" and seed in EXTRA_SEEDS:
                    row["note"] = "extra_seed_P1"
                rows.append(row)
                print(f"[eval] {job_name} seed={seed} SLA={row['sla_rate_mean']:.3f}")

    merged = _merge_rows(existing, rows)
    if merged:
        _write_csv(csv_path, merged)
    print(f"Wrote {csv_path} ({len(merged)} rows, +{len(rows)} new) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

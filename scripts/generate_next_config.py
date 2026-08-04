"""Generate next-round config from experiment analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_next_config(
    round_num: int,
    summary_path: Path,
    rec_path: Path,
    base_cfg_path: Path,
    out_path: Path,
) -> dict:
    cfg = _load_cfg(base_cfg_path)
    next_round = round_num + 1

    a1_sla = 0.0
    random_sla = 0.0
    avg_delay = 2.0
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        a1 = df[(df["method"] == "T2Sem-DRL") & (df["variant"] == "A1_full")]
        rnd = df[(df["method"] == "random") & (df["variant"] == "baseline")]
        if not a1.empty:
            a1_sla = float(a1["sla_rate_mean"].mean())
            avg_delay = float(a1["avg_delay_mean"].mean())
        if not rnd.empty:
            random_sla = float(rnd["sla_rate_mean"].mean())

    recs = []
    if rec_path.exists():
        data = json.loads(rec_path.read_text(encoding="utf-8"))
        recs = data.get("recommendations", [])

    drl = cfg.setdefault("drl", {})
    rw = cfg.setdefault("reward", {})
    task = cfg.setdefault("task", {})
    exp = cfg.setdefault("experiment", {})

  # Per-round isolated outputs
    exp["checkpoint_dir"] = f"results/checkpoints_round{next_round}"
    exp["results_dir"] = f"results/round{next_round}"
    exp["log_dir"] = f"results/logs_round{next_round}"

    episodes = int(drl.get("total_long_frames", 200))
    w_sla = float(rw.get("w_sla", 8.0))

    if a1_sla < random_sla:
        episodes += 40
        drl["bc_demo_episodes"] = min(int(drl.get("bc_demo_episodes", 30)) + 5, 50)
        drl["bc_train_epochs"] = min(int(drl.get("bc_train_epochs", 120)) + 20, 200)
        w_sla = min(w_sla * 1.15, 15.0)
        drl["short"]["lr"] = min(float(drl["short"].get("lr", 3e-4)) * 1.1, 6e-4)
        recs.append(f"round{round_num}: SLA {a1_sla:.3f} < random {random_sla:.3f} -> +episodes +w_sla +BC")
    elif a1_sla < 0.35:
        episodes += 25
        w_sla = min(w_sla * 1.1, 12.0)
        recs.append(f"round{round_num}: low SLA {a1_sla:.3f} -> moderate boost")

    if avg_delay > 1.2:
        rw["w_energy"] = max(float(rw.get("w_energy", 0.1)) * 0.9, 0.05)
        task["arrival_rate"] = max(float(task.get("arrival_rate", 0.25)) * 0.97, 0.15)
        recs.append(f"round{round_num}: high delay {avg_delay:.2f}s -> ease load")

    if a1_sla >= 0.45:
        drl["long"]["noise_std"] = max(float(drl["long"].get("noise_std", 0.08)) * 0.9, 0.03)
        recs.append(f"round{round_num}: SLA improving -> reduce DDPG noise")

    drl["total_long_frames"] = min(episodes, 400)
    cfg["num_long_frames"] = drl["total_long_frames"]
    rw["w_sla"] = w_sla
    rw["w_fairness"] = min(float(rw.get("w_fairness", 0.3)) + 0.05, 0.5)
    drl["short"]["update_epochs"] = min(int(drl["short"].get("update_epochs", 6)) + 1, 10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")

    meta = {
        "from_round": round_num,
        "to_round": next_round,
        "a1_sla": a1_sla,
        "random_sla": random_sla,
        "avg_delay": avg_delay,
        "recommendations": recs,
        "config_path": str(out_path.relative_to(ROOT)),
    }
    meta_path = ROOT / "results" / "auto_loop" / f"round{round_num}_to_{next_round}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--recommendations", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    r = args.round
    summary = Path(args.summary or ROOT / f"results/round{r}" / "experiment_summary.csv")
    recs = Path(args.recommendations or ROOT / "results" / "auto_loop" / f"round{r}_recommendations.json")
    if not recs.exists():
        recs = ROOT / "results" / "optimization_recommendations.json"
    base = Path(
        args.base_config
        or (ROOT / f"configs/optimized_round{r}.yaml" if r > 1 else ROOT / "configs/default.yaml")
    )
    out = Path(args.output or ROOT / f"configs/optimized_round{r + 1}.yaml")

    meta = generate_next_config(r, summary, recs, base, out)
    print(f"Wrote {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

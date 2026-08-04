"""Bootstrap CI and paired tests for revision + supplement CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = m / se if se > 0 else float("inf")
    # two-sided p approx (normal for n=3 is rough)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def bootstrap_mean_diff(a: list[float], b: list[float], n_boot: int = 10000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diffs[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def summarize_job(rows: list[dict], job: str) -> dict:
    sub = [r for r in rows if r["job"] == job]
    sla = [float(r["sla_rate_mean"]) * 100 for r in sub]
    return {
        "job": job,
        "n_seeds": len(sla),
        "sla_mean_pct": round(float(np.mean(sla)), 2) if sla else float("nan"),
        "sla_std_pct": round(float(np.std(sla)), 2) if sla else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=str(ROOT / "results/revision_v43/revision_summary.csv"))
    parser.add_argument("--supplement", default=str(ROOT / "results/supplement_v43/supplement_summary.csv"))
    parser.add_argument("--output", default=str(ROOT / "results/supplement_v43/stats_summary.txt"))
    args = parser.parse_args()

    lines: list[str] = ["# Statistical summary\n"]
    rev = load_rows(Path(args.revision)) if Path(args.revision).exists() else []
    sup = load_rows(Path(args.supplement)) if Path(args.supplement).exists() else []
    all_rows = rev + sup

    for job in sorted({r["job"] for r in all_rows}):
        s = summarize_job(all_rows, job)
        lines.append(f"{s['job']}: n={s['n_seeds']} SLA={s['sla_mean_pct']}% ± {s['sla_std_pct']} pp\n")

    def merge_by_seed(job: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for r in rev:
            if r["job"] == job:
                out[int(r["seed"])] = float(r["sla_rate_mean"]) * 100
        for r in sup:
            if r["job"] == job:
                out[int(r["seed"])] = float(r["sla_rate_mean"]) * 100
        return out

    def paired_block(label: str, a: dict[int, float], b: dict[int, float]) -> None:
        shared = sorted(set(a) & set(b))
        if not shared:
            return
        va = [a[s] for s in shared]
        vb = [b[s] for s in shared]
        tstat, p = paired_ttest(va, vb)
        mean_d, lo, hi = bootstrap_mean_diff(va, vb)
        lines.append(
            f"\nPaired {label} (seeds {shared}):\n"
            f"  mean diff = {mean_d:.2f} pp, 95% bootstrap CI [{lo:.2f}, {hi:.2f}]\n"
            f"  paired t ≈ {tstat:.2f}, normal-approx p ≈ {p:.4f} (n={len(shared)}; interpret cautiously)\n"
        )

    by_seed_g = merge_by_seed("greedy")
    by_seed_t = merge_by_seed("t2sem_full")
    by_seed_m = merge_by_seed("semantic_mappo")
    by_seed_moff = merge_by_seed("semantic_mappo_mask_off")
    by_seed_su_t = merge_by_seed("scaleup_t2sem")
    by_seed_su_m = merge_by_seed("scaleup_semantic_mappo")

    paired_block("t2sem_full vs greedy", by_seed_t, by_seed_g)
    paired_block("t2sem_full vs semantic_mappo", by_seed_t, by_seed_m)
    paired_block("semantic_mappo vs semantic_mappo_mask_off", by_seed_m, by_seed_moff)
    paired_block("scaleup_t2sem vs scaleup_semantic_mappo", by_seed_su_t, by_seed_su_m)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

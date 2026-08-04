## v4.3 Baseline Freeze (Ideal Track)

This file freezes the stable submission line so the ideal-track work can evolve
without changing the evidence behind the current paper.

### Frozen baseline package

- Core config: `configs/improved_v4_pertier.yaml`
- Supplement suite:
  - `scripts/s_tier_run.py`
  - `scripts/a_tier_run.py`
  - `scripts/scaleup_semantic_run.py`
- Statistics:
  - `scripts/bootstrap_stats.py`
  - `results/supplement_v43/supplement_summary.csv`
  - `results/supplement_v43/stats_summary.txt`
- Paper source:
  - `C:/Users/12293/Desktop/T2Sem-DRL-v4-paper/draft_full.md`
  - `C:/Users/12293/Desktop/T2Sem-DRL-v4-paper/REBUTTAL_TOP6.md`

### Frozen protocol assumptions

- Canonical protocol: 3 seeds x 30 eval episodes.
- Expanded check: seeds `{42,123,456,789,1011}` for semantic comparison.
- Scale-up triangle (matched-training semantic baseline): seeds `{42,123,456}`.
- No claim of universal energy or critical-tier dominance.

### Guardrails for ideal-track changes

1. New configs/scripts must use `ideal_` prefix.
2. New result rows must use `job` names prefixed by `ideal_`.
3. Do not overwrite `supplement_v43` artifacts from the frozen line.
4. Keep all ideal-track outputs under `results/ideal_track/`.
5. Any paper-facing import from ideal-track must be explicitly marked as such.

### Compatibility target

Ideal-track implementations should remain executable through the existing
environment and metrics APIs (`MultiMECEnv`, `aggregate_metrics`) to reduce
maintenance cost and enable direct ablation against frozen v4.3.

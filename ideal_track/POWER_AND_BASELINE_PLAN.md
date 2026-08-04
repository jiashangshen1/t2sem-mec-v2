## Power and External Baseline Plan

### Seed expansion policy

- Minimum target: 8 seeds for key claims.
- Preferred target: 10 seeds (`configs/ideal_power_eval.yaml`).
- CI-width stop criterion: <= 3.0 pp for primary SLA comparison.

### Commands

1) Run ideal method on expanded seeds:

```bash
python scripts/ideal_t2semx_run.py --seeds 42 123 456 789 1011 2027 3031 4049 5051 6067
```

2) Check CI width / seed sufficiency:

```bash
python scripts/ideal_seed_power.py --csv results/ideal_track/ideal_summary.csv --target-width-pp 3.0
```

3) Initialize external baseline status board:

```bash
python scripts/ideal_baseline_repro.py
```

### Reporting rules

1. Never report a baseline as "full reproduction" unless status table marks it
   as `done` with evidence path.
2. If a baseline is partial/Lite, explicitly mark it as partial in paper tables.
3. Primary claims must use expanded seeds and CI, not only point estimates.

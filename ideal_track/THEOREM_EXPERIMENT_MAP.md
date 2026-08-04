## Theorem-Experiment Mapping Matrix (Ideal Track)

This matrix binds each theoretical claim in `THEORY_NOTE_V1.md` to executable
experiments and concrete output files.

| Theory item | Validation experiment | Script / config | Primary metrics | Output artifact |
|---|---|---|---|---|
| P1 Robust feasibility retention | compare projected vs weakly-projected action behavior under channel mismatch | `scripts/ideal_stress_suite.py` + `configs/ideal_stress_nlos.yaml` | tier SLA floor, violation ratio, delay tail | `results/ideal_track/stress_summary.csv` |
| P2 Tier-wise dual boundedness | track per-tier lambda trajectories under conflicting tier targets | `scripts/ideal_stress_suite.py` + `configs/ideal_stress_tier_conflict.yaml` | max/min lambda, oscillation amplitude, per-tier SLA | `results/ideal_track/stress_summary.csv` |
| P3 Suboptimality decomposition trend | sweep `(alpha, delta_s)` and compare SLA-energy tradeoff shape | reuse `scripts/mask_sensitivity.py` + ideal checkpoints | SLA-energy slope, feasibility volume proxy | `results/ideal_track/mask_sensitivity_ideal.csv` |
| C1 Over-conservative mask collapse | force strict margins (`alpha=0.6`, `delta_s=0.1`) | `scripts/ideal_stress_suite.py` / scenario `mask_collapse` | throughput drop, feasible target count | `results/ideal_track/stress_summary.csv` |
| C2 Single lambda conflict | disable per-tier dual and compare tier imbalance | `scripts/ideal_stress_suite.py` / scenario `single_lambda_conflict` | critical vs interactive gap | `results/ideal_track/stress_summary.csv` |
| C3 Timescale mismatch instability | amplify long noise and reduce smoothing | `scripts/ideal_stress_suite.py` / scenario `timescale_mismatch` | trajectory energy variance, SLA oscillation | `results/ideal_track/stress_summary.csv` |
| C4 Model mismatch under severe NLoS | evaluate with pessimistic channel assumption | `scripts/ideal_stress_suite.py` + `configs/ideal_stress_nlos.yaml` | SLA collapse ratio vs expected channel | `results/ideal_track/stress_summary.csv` |

### Execution order

1. Train / load `ideal_t2sem_x` checkpoints (`scripts/ideal_t2semx_run.py`).
2. Run stress suite:
   - `python scripts/ideal_stress_suite.py --method ideal_t2sem_x`
3. Repeat for baseline method:
   - `python scripts/ideal_stress_suite.py --method t2sem_full`
4. Compare rows by `scenario` in `stress_summary.csv`.

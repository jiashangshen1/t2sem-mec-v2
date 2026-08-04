# Final numeric summary

## Main comparison (mean over 3 seeds)

| job | SLA % | std pp | delay s | energy J |
|---|---:|---:|---:|---:|
| greedy | 87.77 | 0.07 | 0.122 | 3630 |
| independent_ppo | 91.41 | 0.25 | 0.145 | 20482 |
| local_only | 86.05 | 0.37 | 0.209 | 6556 |
| mobility_full | 16.39 | 13.58 | 0.870 | 53959 |
| mobility_single_ts | 1.67 | 2.90 | 2.403 | 8509 |
| random | 78.47 | 0.06 | 0.159 | 3914 |
| t2sem_full | 93.99 | 1.74 | 0.100 | 7856 |
| t2sem_single_lambda | 93.99 | 1.74 | 0.100 | 7856 |
| t2sem_single_ts | 88.85 | 0.43 | 0.145 | 4065 |
| tjcct | 86.64 | 0.07 | 0.266 | 3698 |

## Ablation v4.3

| ablation | SLA % | energy J | critical % | interactive % | best-effort % |
|---|---:|---:|---:|---:|---:|
| full | 92.17 | 15366 | 75.95 | 99.17 | 99.99 |
| mask_off | 60.39 | 61104 | 5.20 | 78.45 | 100.00 |
| single_lambda | 92.80 | 17518 | 83.78 | 94.31 | 99.99 |
| no_cmdp | 92.04 | 16418 | 77.95 | 99.26 | 99.90 |
| pure_drl | 59.91 | 57712 | 2.62 | 70.76 | 100.00 |
| no_semantic | 46.17 | 43776 | 1.80 | 36.50 | 100.00 |
| single_timescale | 90.70 | 4013 | 73.82 | 98.34 | 99.98 |
| no_sla_penalty | 92.18 | 16543 | 86.00 | 90.74 | 100.00 |
| fixed_mu | 65.72 | 45852 | 0.00 | 98.98 | 99.87 |
| no_contention | 100.00 | 56026 | 100.00 | 100.00 | 100.00 |

# T2Sem-DRL v4.3 Revision Run Summary

Source: `results/revision_v43/revision_summary.csv` (30 rows)

## Per-job mean SLA (%)

| Job | SLA mean | SLA std (across seeds) | Delay | Energy (J) |
|-----|----------|------------------------|-------|------------|
| greedy | 87.77 | 0.07 | 0.122 | 3630 |
| tjcct | 86.64 | 0.07 | 0.266 | 3698 |
| random | 78.47 | 0.06 | 0.159 | 3914 |
| local_only | 86.05 | 0.37 | 0.209 | 6556 |
| t2sem_full | 93.99 | 1.74 | 0.100 | 7856 |
| t2sem_single_lambda | 93.99 | 1.74 | 0.100 | 7856 |
| t2sem_single_ts | 88.85 | 0.43 | 0.145 | 4065 |
| independent_ppo | 91.41 | 0.25 | 0.145 | 20482 |
| mobility_full | 16.39 | 13.58 | 0.870 | 53959 |
| mobility_single_ts | 1.67 | 2.90 | 2.403 | 8509 |

## Per seed

| Job | Seed | SLA % | Energy |
|-----|------|-------|--------|
| greedy | 42 | 87.84 | 3626 |
| greedy | 123 | 87.78 | 3629 |
| greedy | 456 | 87.71 | 3634 |
| tjcct | 42 | 86.61 | 3696 |
| tjcct | 123 | 86.72 | 3696 |
| tjcct | 456 | 86.59 | 3703 |
| random | 42 | 78.53 | 3911 |
| random | 123 | 78.40 | 3914 |
| random | 456 | 78.48 | 3919 |
| local_only | 42 | 85.97 | 6549 |
| local_only | 123 | 86.45 | 6553 |
| local_only | 456 | 85.73 | 6566 |
| t2sem_full | 42 | 92.34 | 10751 |
| t2sem_full | 123 | 93.82 | 5319 |
| t2sem_full | 456 | 95.81 | 7499 |
| t2sem_single_lambda | 42 | 92.34 | 10751 |
| t2sem_single_lambda | 123 | 93.82 | 5319 |
| t2sem_single_lambda | 456 | 95.81 | 7499 |
| t2sem_single_ts | 42 | 88.46 | 4312 |
| t2sem_single_ts | 123 | 89.31 | 3980 |
| t2sem_single_ts | 456 | 88.78 | 3902 |
| independent_ppo | 42 | 91.16 | 21314 |
| independent_ppo | 123 | 91.66 | 9283 |
| independent_ppo | 456 | 91.43 | 30850 |
| mobility_full | 42 | 29.77 | 89178 |
| mobility_full | 123 | 2.63 | 10557 |
| mobility_full | 456 | 16.78 | 62141 |
| mobility_single_ts | 42 | 0.00 | 11453 |
| mobility_single_ts | 123 | 5.02 | 11345 |
| mobility_single_ts | 456 | 0.00 | 2728 |

**Greedy baseline mean:** 87.77%

- **t2sem_full** vs greedy: +6.21 pp
- **t2sem_single_lambda** vs greedy: +6.21 pp
- **t2sem_single_ts** vs greedy: +1.08 pp
- **independent_ppo** vs greedy: +3.64 pp

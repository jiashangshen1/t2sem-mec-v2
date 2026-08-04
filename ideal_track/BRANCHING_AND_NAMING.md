## Ideal Track Branching and Naming

### Branch model

- Stable submission line: `paper/v43-stable`
- Ideal track mainline: `ideal/main`
- Feature branches:
  - `ideal/theory-*`
  - `ideal/method-*`
  - `ideal/exp-*`

### File naming

- Configs: `configs/ideal_*.yaml`
- Scripts: `scripts/ideal_*.py`
- Result folders: `results/ideal_track/*`
- Paper notes: `ideal_track/*.md`

### Experiment naming

- New methods: `ideal_t2sem_x`, `ideal_t2sem_x_no_robust`, etc.
- Paired comparisons:
  - `ideal_t2sem_x vs t2sem_full`
  - `ideal_t2sem_x vs independent_ppo`

### Required metadata in result rows

Every ideal-track row in CSV should include:

- `job` (must start with `ideal_`)
- `variant`
- `seed`
- `note` (must include one of: `theory_check`, `method_ablation`, `stress_test`)

### Merge policy

Changes from ideal branches can merge into `ideal/main` only if:

1. They do not change frozen v4.3 outputs.
2. They include a reproducible command in the commit/PR description.
3. They pass at least one sanity check (`tests/test_sanity.py` or equivalent).

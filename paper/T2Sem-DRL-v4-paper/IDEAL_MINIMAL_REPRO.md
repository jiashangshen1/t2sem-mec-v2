## Minimal Repro Package (Ideal Track)

### 1) Environment

```bash
cd t2sem-mec-v2
pip install -r requirements.txt
python tests/test_sanity.py
python tests/test_ideal_t2semx.py
```

### 2) Train ideal method (3-seed smoke)

```bash
python scripts/ideal_t2semx_run.py --seeds 42 123 456 --eval-episodes 30
```

### 3) Stress suite

```bash
python scripts/ideal_stress_suite.py --method ideal_t2sem_x
python scripts/ideal_stress_suite.py --method t2sem_full
```

### 4) Power check

```bash
python scripts/ideal_seed_power.py --csv results/ideal_track/ideal_summary.csv --target-width-pp 3.0
```

### 5) Baseline repro board

```bash
python scripts/ideal_baseline_repro.py
```

### Expected outputs

- `results/ideal_track/ideal_summary.csv`
- `results/ideal_track/stress_summary.csv`
- `results/ideal_track/baseline_repro_status.csv`

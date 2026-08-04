#!/usr/bin/env bash
# Supplementary experiments: semantic MAPPO+F, extra seeds, scale-up, critical-priority, alpha validation.
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
OUT=results/supplement_v43
LOG="$OUT/supplement.log"

cd "$PROJ"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

bash scripts/launch_progress_dashboard.sh

echo "=== supplement_run WSL $(date -Is) ===" | tee "$LOG"

# Phase 1: fast eval (semantic MAPPO from independent ckpt, extra-seed baselines)
echo "--- phase quick ---" | tee -a "$LOG"
$PY scripts/supplement_run.py --device cpu --phase quick --resume 2>&1 | tee -a "$LOG"

# Alpha validation (no train)
echo "--- alpha validation ---" | tee -a "$LOG"
$PY scripts/alpha_validation.py --device cpu --eval-episodes 30 2>&1 | tee -a "$LOG"

# Phase 2: training jobs (extra seeds, scale-up, critical-priority)
echo "--- phase train ---" | tee -a "$LOG"
$PY scripts/supplement_run.py --device cpu --phase train --resume 2>&1 | tee -a "$LOG"

# Stats
echo "--- bootstrap stats ---" | tee -a "$LOG"
$PY scripts/bootstrap_stats.py 2>&1 | tee -a "$LOG"

# Sync to paper package
$PY scripts/sync_supplement_to_paper.py 2>&1 | tee -a "$LOG"

echo "SUPPLEMENT_DONE $(date -Is)" | tee -a "$LOG"

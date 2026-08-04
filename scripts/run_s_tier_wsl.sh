#!/usr/bin/env bash
# S-tier reviewer-critical experiments (semantic MAPPO extra seeds, per-tier, mask-off).
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
OUT=results/supplement_v43
LOG="$OUT/s_tier.log"

cd "$PROJ"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

bash scripts/launch_progress_dashboard.sh

echo "=== s_tier_run WSL $(date -Is) ===" | tee "$LOG"

echo "--- per-tier critical_priority (fast) ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only per_tier --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- semantic_mappo mask-off eval ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only mask_off --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- semantic_mappo train+eval seeds 789,1011 ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only semantic_train --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- bootstrap + sync ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only bootstrap 2>&1 | tee -a "$LOG"

echo "S_TIER_DONE $(date -Is)" | tee -a "$LOG"

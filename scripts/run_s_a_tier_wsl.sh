#!/usr/bin/env bash
# Full S-tier then A-tier pipeline (sequential).
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
OUT=results/supplement_v43
LOG="$OUT/s_a_tier.log"

cd "$PROJ"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

bash scripts/launch_progress_dashboard.sh

echo "=== S+A tier pipeline $(date -Is) ===" | tee "$LOG"

echo "--- S1: per-tier + mask-off ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only per_tier,mask_off --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- S2: semantic_mappo train 789,1011 ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only semantic_train --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- S3: bootstrap ---" | tee -a "$LOG"
$PY scripts/s_tier_run.py --device cpu --only bootstrap 2>&1 | tee -a "$LOG"

echo "--- A1: deployment triangle ---" | tee -a "$LOG"
$PY scripts/a_tier_run.py --device cpu --only deploy --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- A2: scale-up semantic ---" | tee -a "$LOG"
$PY scripts/a_tier_run.py --device cpu --only scaleup --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- A3: load sweep ---" | tee -a "$LOG"
$PY scripts/a_tier_run.py --device cpu --only load --eval-episodes 30 2>&1 | tee -a "$LOG"

echo "--- A4: mask sensitivity (critical + scaleup) ---" | tee -a "$LOG"
$PY scripts/a_tier_run.py --device cpu --only mask 2>&1 | tee -a "$LOG"

echo "--- A5: final bootstrap + sync ---" | tee -a "$LOG"
$PY scripts/a_tier_run.py --device cpu --only bootstrap 2>&1 | tee -a "$LOG"

echo "S_A_TIER_DONE $(date -Is)" | tee -a "$LOG"

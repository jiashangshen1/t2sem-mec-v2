#!/usr/bin/env bash
# DEPRECATED: use run_watchdog_wsl.sh (revision_pipeline_watchdog.py) instead.
# Kept for manual one-shot runs only.
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
cd "$PROJ"

REV_LOG=results/revision_v43/run.log
PIPE_LOG=results/revision_v43/pipeline.log
SUMMARY_CSV=results/revision_v43/revision_summary.csv
ABL_LOG=results/ablation_v43/ablation_run.log

mkdir -p results/revision_v43 results/ablation_v43
exec > >(tee -a "$PIPE_LOG") 2>&1

echo "=== pipeline wait start $(date -Is) ==="

# Wait until revision_run exits
while pgrep -f 'scripts/revision_run.py' >/dev/null 2>&1; do
  echo "[wait] revision_run still running $(date -Is)"
  sleep 120
done

echo "[wait] revision_run finished $(date -Is)"

# Fix independent_ppo seed_42 if smoke left 15-ep checkpoint
IPP=results/checkpoints_independent_ppo/full/seed_42/.train_complete
if [ -f "$IPP" ] && grep -q '^episodes=15' "$IPP"; then
  echo "[fix] retrain independent_ppo seed_42 (was 15-ep smoke)"
  rm -rf results/checkpoints_independent_ppo/full/seed_42
  $PY scripts/revision_run.py --device cpu --only independent_ppo 2>&1 | tee -a "$REV_LOG"
fi

if [ ! -f "$SUMMARY_CSV" ]; then
  echo "[error] missing $SUMMARY_CSV after revision" >&2
  exit 1
fi

echo "=== REVISION SUMMARY $(date -Is) ==="
$PY scripts/summarize_revision.py | tee results/revision_v43/REVISION_SUMMARY.md

# Sync to paper package
PAPER=/mnt/c/Users/12293/Desktop/T2Sem-DRL-v4-paper
if [ -d "$PAPER" ]; then
  cp -f "$SUMMARY_CSV" "$PAPER/data/revision_summary.csv"
  cp -f "$SUMMARY_CSV" "$PAPER/data/experiment_summary.csv"
  cp -f results/revision_v43/baseline_v43.csv "$PAPER/data/baseline_v43.csv" 2>/dev/null || true
  cp -f results/revision_v43/REVISION_SUMMARY.md "$PAPER/REVISION_SUMMARY.md" 2>/dev/null || true
  echo "[sync] copied CSV -> $PAPER/data/"
fi

echo "=== FULL ABLATION v4.3 start $(date -Is) ==="
echo "10 variants x 3 seeds x 300 ep (est ~12-15h CPU)"

$PY scripts/ablation_run.py \
  --config configs/improved_v4_pertier.yaml \
  --suite v43 \
  --episodes 300 \
  --seeds 42 123 456 \
  --eval-episodes 30 \
  --device cpu \
  --resume \
  2>&1 | tee -a "$ABL_LOG"

if [ -d "$PAPER" ]; then
  cp -f results/ablation_v43/ablation_summary.csv "$PAPER/data/ablation_summary.csv"
  echo "[sync] ablation_summary.csv -> $PAPER/data/"
fi

echo "=== PIPELINE DONE $(date -Is) ==="
echo "ABLATION_DONE"

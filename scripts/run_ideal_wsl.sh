#!/usr/bin/env bash
# Standard entry: ideal-track pipeline on WSL Ubuntu + CPU.
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
OUT=results/ideal_track
LOG="$OUT/run.log"

cd "$PROJ"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

echo "=== ideal_track WSL $(date -Is) ===" | tee "$LOG"
echo "python=$PY" | tee -a "$LOG"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"

if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -Command "Start-Process powercfg -ArgumentList '/change','standby-timeout-ac','0'" >/dev/null 2>&1 || true
fi

$PY tests/test_ideal_t2semx.py 2>&1 | tee -a "$LOG"

$PY scripts/ideal_t2semx_run.py --device cpu --seeds 42 123 456 2>&1 | tee -a "$LOG"

$PY scripts/ideal_stress_suite.py --method ideal_t2sem_x --device cpu 2>&1 | tee -a "$LOG"
$PY scripts/ideal_stress_suite.py --method t2sem_full --device cpu --append --bootstrap-train 2>&1 | tee -a "$LOG"

$PY scripts/ideal_t2semx_run.py --device cpu --variant no_robust --seeds 42 123 456 --append 2>&1 | tee -a "$LOG"
$PY scripts/ideal_t2semx_run.py --device cpu --variant no_credit --seeds 42 123 456 --append 2>&1 | tee -a "$LOG"

echo "=== ideal_track WSL done $(date -Is) ===" | tee -a "$LOG"

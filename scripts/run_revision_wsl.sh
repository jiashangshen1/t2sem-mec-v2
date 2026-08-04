#!/usr/bin/env bash
# Standard entry: run full revision suite on WSL Ubuntu + CPU.
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
OUT=results/revision_v43
LOG="$OUT/run.log"

cd "$PROJ"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

bash scripts/launch_progress_dashboard.sh

echo "=== revision_run WSL $(date -Is) ===" | tee -a "$LOG"
echo "python=$PY" | tee -a "$LOG"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"

# Prevent Windows sleep during long runs (best-effort via powershell.exe if available)
if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -Command "Start-Process powercfg -ArgumentList '/change','standby-timeout-ac','0'" >/dev/null 2>&1 || true
fi

exec $PY scripts/revision_run.py --device cpu --resume 2>&1 | tee -a "$LOG"
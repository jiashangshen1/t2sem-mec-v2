#!/usr/bin/env bash
# Start revision pipeline watchdog (monitors revision + ablation, auto-fix/restart).
set -euo pipefail

PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python
cd "$PROJ"
mkdir -p results/revision_v43

bash scripts/launch_progress_dashboard.sh

# Stop legacy bash waiter if still running (watchdog supersedes it)
pkill -f 'wait_revision_then_ablation.sh' 2>/dev/null || true

export PYTHONUNBUFFERED=1
echo "=== watchdog launcher $(date -Is) ===" | tee -a results/revision_v43/watchdog.log
exec "$PY" scripts/revision_pipeline_watchdog.py --interval 600 2>&1 | tee -a results/revision_v43/watchdog.log

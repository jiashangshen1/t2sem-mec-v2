#!/usr/bin/env bash
set -e
PROJ="/mnt/d/research/uav dql/code/t2sem-mec-v2"
PY=/root/venvs/ml/bin/python

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found"
  exit 1
fi

cd "$PROJ"
echo "=== WSL env ==="
uname -a
$PY -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo ""
echo "=== Benchmark 1 train episode (no BC) ==="
$PY scripts/bench_cpu_vs_gpu.py

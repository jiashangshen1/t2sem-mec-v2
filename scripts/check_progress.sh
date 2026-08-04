#!/usr/bin/env bash
ps aux | grep '[r]evision_run' || echo "NOT_RUNNING"
echo "---"
for d in logs_improved_pertier logs_independent_ppo logs_mobility_critical; do
  base="/mnt/d/research/uav dql/code/t2sem-mec-v2/results/$d/full"
  if [ -d "$base" ]; then
    echo "[$d]"
    for s in 42 123 456; do
      f="$base/seed_$s/train_metrics.csv"
      if [ -f "$f" ]; then
        ep=$(tail -1 "$f" | cut -d, -f1)
        echo "  seed_$s: ${ep}/300"
      fi
    done
  fi
done
echo "---log tail---"
tail -c 500 /mnt/d/research/uav dql/code/t2sem-mec-v2/results/revision_v43/run.log 2>/dev/null | tr '\r' '\n' | tail -5
if [ -f /mnt/d/research/uav dql/code/t2sem-mec-v2/results/revision_v43/revision_summary.csv ]; then
  echo "---COMPLETE---"
  wc -l /mnt/d/research/uav dql/code/t2sem-mec-v2/results/revision_v43/revision_summary.csv
fi

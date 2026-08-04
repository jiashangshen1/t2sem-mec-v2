#!/usr/bin/env bash
# Open Windows progress dashboard (browser + 5s watcher). Safe to call multiple times.
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
BAT="$PROJ/scripts/start_progress_watch.bat"
if command -v cmd.exe >/dev/null 2>&1 && [ -f "$BAT" ]; then
  WIN_BAT=$(wslpath -w "$BAT" 2>/dev/null || echo "D:\\research\\uav dql\\code\\t2sem-mec-v2\\scripts\\start_progress_watch.bat")
  cmd.exe /c start "" "$WIN_BAT" >/dev/null 2>&1 || true
fi

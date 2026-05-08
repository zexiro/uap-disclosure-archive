#!/usr/bin/env bash
# Container entrypoint. Run a refresh in the background on a 6-hour loop,
# and the HTTP server in the foreground. Railway treats the foreground
# process as the service; logs from both go to the same stdout.
set -euo pipefail
cd /app

# Background: initial pipeline (if volume empty) + 6-hourly refresh.
# The HTTP server starts immediately so health checks pass; the UI works
# from the bundled search-index right away. Inline previews start working
# once the volume is populated (~5–10 min on Railway's bandwidth).
(
  if [ ! -f raw/records.json ]; then
    echo "[boot] empty volume — running initial pipeline"
    bash scripts/refresh.sh || echo "[boot] initial refresh hit errors (will retry on schedule)"
  fi
  while true; do
    sleep 21600
    echo "[cron] kicking off scheduled refresh at $(date -u +%FT%TZ)"
    bash scripts/refresh.sh || echo "[cron] refresh hit errors (will retry next tick)"
  done
) &

# Foreground: HTTP server (Railway tracks this PID for health)
exec python3 scripts/serve.py

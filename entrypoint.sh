#!/usr/bin/env bash
# Container entrypoint. Run a refresh in the background on a 6-hour loop,
# and the HTTP server in the foreground. Railway treats the foreground
# process as the service; logs from both go to the same stdout.
set -euo pipefail
cd /app

# On first boot, populate the volume if it's empty.
if [ ! -f raw/records.json ]; then
  echo "[boot] empty volume — running initial pipeline (this can take ~30 minutes)"
  bash scripts/refresh.sh || echo "[boot] initial refresh hit errors (will retry on schedule)"
fi

# Background: refresh every 6 hours
(
  while true; do
    sleep 21600
    echo "[cron] kicking off scheduled refresh at $(date -u +%FT%TZ)"
    bash scripts/refresh.sh || echo "[cron] refresh hit errors (will retry next tick)"
  done
) &

# Foreground: HTTP server (Railway tracks this PID for health)
exec python3 scripts/serve.py

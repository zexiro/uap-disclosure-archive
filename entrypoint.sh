#!/usr/bin/env bash
# Container entrypoint. Run a refresh in the background on a 6-hour loop,
# and the HTTP server in the foreground. Railway treats the foreground
# process as the service; logs from both go to the same stdout.
set -euo pipefail
cd /app

# Background loop A: 6-hourly full refresh.
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

# Background loop B: hourly war.gov change-watcher.
# Lightweight HEAD/GET sweep (~1 min) that detects file adds/removes/edits
# between full refreshes. If anything changed it drops a sentinel file
# (raw/wargov_changes_pending) and we fire refresh.sh immediately so the
# UI surfaces the new content within minutes instead of hours.
(
  while true; do
    sleep 3600
    echo "[watch] hourly change-check at $(date -u +%FT%TZ)"
    python3 scripts/check_wargov_changes.py || echo "[watch] check hit errors"
    if [ -f raw/wargov_changes_pending ]; then
      echo "[watch] changes detected — firing early refresh"
      bash scripts/refresh.sh || echo "[watch] triggered refresh hit errors"
      rm -f raw/wargov_changes_pending
    fi
  done
) &

# Foreground: ASGI HTTP server (Railway tracks this PID for health).
# Single worker: embeddings.npz is loaded once per process; adding workers
# would duplicate the ~500 MB in-memory matrix. Use --workers 1 always.
exec uvicorn scripts.serve:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1

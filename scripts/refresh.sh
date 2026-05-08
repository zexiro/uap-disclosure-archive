#!/usr/bin/env bash
# Run periodically (Railway cron, or local). Detects new content from
# war.gov/UFO and runs the pipeline only if something changed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[refresh] $(date -u +%FT%TZ) starting"

# 1. Re-fetch the source CSV
mkdir -p raw/csv
prev_hash=$(shasum -a 256 raw/csv/uap-csv.csv 2>/dev/null | awk '{print $1}' || true)
./scripts/fetch.sh \
  "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv" \
  raw/csv/uap-csv.csv
new_hash=$(shasum -a 256 raw/csv/uap-csv.csv | awk '{print $1}')

if [ "$prev_hash" = "$new_hash" ] && [ -f raw/records.json ]; then
  echo "[refresh] no change in source CSV — skipping pipeline"
  exit 0
fi

echo "[refresh] CSV changed (or first run) — running full pipeline"

# 2. Pipeline
python3 scripts/parse_csv.py
python3 scripts/download.py
python3 scripts/ocr.py || true       # OCR errors aren't fatal
python3 scripts/extract_pdf_images.py || true   # extract embedded photos/sketches
python3 scripts/build_links.py
python3 scripts/build_search_index.py

# Vault: rebuild fresh so removed/renamed records don't linger
rm -rf vault/Releases vault/Index vault/README.md
python3 scripts/build_vault.py

echo "[refresh] $(date -u +%FT%TZ) done. Search index updated."

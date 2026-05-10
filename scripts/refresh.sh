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
  # CSV unchanged, but cheap incremental steps still run (idempotent + cached).
  # Lets newly-added dossier keywords or a freshly-set ANTHROPIC_API_KEY take
  # effect without requiring a bogus CSV change.
  echo "[refresh] no change in source CSV — running incremental steps only"
  python3 scripts/build_thumbs.py || true
  python3 scripts/classify_dossier_hits.py || true
  python3 scripts/build_image_embeddings.py || true  # pick up any newly-downloaded images
  exit 0
fi

echo "[refresh] CSV changed (or first run) — running full pipeline"

# 2. Pipeline
python3 scripts/parse_csv.py
python3 scripts/download.py
python3 scripts/ocr.py || true       # OCR errors aren't fatal
python3 scripts/extract_pdf_images.py || true   # extract embedded photos/sketches
python3 scripts/build_thumbs.py || true         # small JPEGs for row/grid views
python3 scripts/build_links.py
python3 scripts/extract_citations.py
python3 scripts/build_search_index.py
# Cheap-LLM relevance check on each dossier keyword hit (false-positive filter).
# Cached by hash(kw, ctx) under raw/dossier_classifications.json so subsequent
# runs only call the API for genuinely new hits. Graceful no-op without
# ANTHROPIC_API_KEY — the UI falls back to keyword-only matching.
python3 scripts/classify_dossier_hits.py || true
python3 scripts/build_image_embeddings.py || true  # CLIP visual embeddings for IMG records
python3 scripts/build_features.py
python3 scripts/build_api.py

# Multi-source sightings aggregator (NUFORC civilian + war.gov, projected to a
# unified schema with provenance tagging; correlations pass finds civilian
# reports near each official incident). NUFORC CSV is cached; subsequent
# runs are essentially free.
python3 scripts/sightings/fetch_nuforc.py || true
python3 scripts/sightings/fetch_blue_book.py || true
python3 scripts/sightings/fetch_reddit.py || true
python3 scripts/sightings/fetch_news.py || true
python3 scripts/sightings/build_unified.py || true
python3 scripts/sightings/build_correlations.py || true

# Vault: rebuild fresh so removed/renamed records don't linger
rm -rf vault/Releases vault/Index vault/README.md
python3 scripts/build_vault.py

echo "[refresh] $(date -u +%FT%TZ) done. Search index updated."

# UFO/UAP Disclosure Archive

A self-hosted, searchable mirror of the U.S. Department of War's
[UFO disclosure release](https://www.war.gov/UFO/) — every PDF, image, and
video; full-text OCR'd; an Obsidian vault; and a single-page web search UI
with content + visual relationship linking.

**[ATTRIBUTION & legal notes →](ATTRIBUTION.md)** &nbsp;·&nbsp;
[MIT licensed](LICENSE) source · public-domain US government source material

## What's in this repo

| Folder | Size | Committed? |
|---|---|---|
| `ui/` | 3 MB | ✓ — search UI + bundled `search-index.json` |
| `vault/` | 5 MB | ✓ — Obsidian vault, 162 cross-linked release notes |
| `scripts/` | 56 KB | ✓ — Python pipeline (parse → download → OCR → links → vault → index) |
| `raw/csv,records,links,text` | 6 MB | ✓ — manifests, metadata, OCR sidecars |
| `raw/docs` (PDFs) | 2.4 GB | ✗ — rebuilt by `scripts/download.py`, served from Railway volume |
| `raw/images` | 45 MB | ✗ — same |
| `raw/videos` (MP4) | 1.3 GB | ✗ — same (DVIDS) |

The repo itself is ~14 MB. The 5.6 GB of bulk media lives in a Railway
persistent volume and gets rebuilt automatically on first boot, then
refreshed every 6 hours.

## Local dev

```bash
# Run the full pipeline locally (one-off, ~30 min)
make all
# Serve the UI
make serve   # → http://localhost:8765/ui/
```

The search UI works the moment you open it (the search index is committed).
Inline document/video previews appear once `download.py` populates `raw/`.

## Deploy to Railway

The repo is wired for one-command Railway deploys:

```bash
railway init
railway up --detach -m "first deploy"
railway domain                 # mint a *.up.railway.app URL
```

Set up a persistent volume for `raw/` so the 5.6 GB doesn't get rebuilt
every redeploy:

```bash
railway volume add --service <service-name> --mount-path /app/raw
```

What happens:
1. Container starts → `entrypoint.sh` checks if `raw/records.json` exists.
2. If volume is empty → triggers `scripts/refresh.sh` (initial sync, ~30 min).
3. Foreground HTTP server (`scripts/serve.py`) is up immediately on `$PORT`.
4. Background loop runs `scripts/refresh.sh` every 6 hours — re-fetches the
   source CSV, runs the pipeline only when content changed.

`git push` to GitHub → Railway auto-redeploys. The volume persists, so
re-deploys are fast and don't lose data.

## Architecture

```
                    ┌──────────────────────────────┐
                    │   war.gov/UFO/uap-csv.csv    │  ← source of truth
                    └──────────────┬───────────────┘
                                   │ refresh.sh (every 6h)
                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                       Railway service                             │
   │                                                                   │
   │   parse_csv.py → download.py → ocr.py → build_links.py →          │
   │   build_search_index.py + build_vault.py                          │
   │                          │                                        │
   │                          ▼                                        │
   │   Persistent volume: /app/raw  (PDFs, videos, OCR text)          │
   │                                                                   │
   │   serve.py — single-page UI (MiniSearch) + static asset server    │
   └───────────────────────────────────────────────────────────────────┘
```

## Roadmap

- [x] Mirror war.gov Release 1 (162 records, 117 PDFs, 28 videos, 139 images)
- [x] Full-text OCR via pdftotext + ocrmypdf fallback
- [x] Content similarity links (TF-IDF cosine over title + blurb + extracted text)
- [x] Visual similarity links (perceptual hash on thumbnails + photos)
- [x] Obsidian vault with cross-linked notes (open in graph view)
- [x] Single-page search UI with media-type chips, filters, URL state, keyboard nav
- [x] **Multi-source aggregation** (NUFORC + Project Blue Book civilian/historical sources, with provenance tagging and geo+temporal correlations against the official archive)
- [x] **Hourly war.gov change-watcher** (HEAD/GET sweep, hash-based diff, OCR-text drift flag, fires early refresh on detected changes)
- [x] **ASGI server** (uvicorn + starlette) — async file serving + non-blocking SSE streaming, replaces ThreadingHTTPServer
- [x] **Semantic embeddings** — text via BAAI/bge-small-en-v1.5 (`ui/embeddings.npz`), images via CLIP-ViT-B-32 (`ui/image_embeddings.npz`), both ONNX/fastembed
- [x] **Named-entity extraction** — auto-extracted via spaCy en_core_web_sm: PERSON / ORG / GPE / DATE / LAW / EVENT / CODENAME plus redacted-name length-class fingerprints (`ui/entities.json`); the curated `ui/entity-gazetteer.json` remains the high-quality canonical list
- [x] **Cross-reference graph** — TF-IDF + pHash similarity (`raw/links.json`), plus auto-extracted in-PDF case-number citations as an explicit edge graph (`ui/citations.json`)
- [x] **RAG / "Ask the archive" with citations** — `/api/ask` SSE-streamed, hybrid dense+lexical retrieval, server-side citation enforcement
- [x] **Map + timeline views** — Leaflet basemap with marker-clustered incidents, vis-timeline scrubber over corpus + 16 milestones, both provenance-aware (civilian records visually distinct at layout level)
- [x] **User-submitted corrections + annotations** — `POST /api/corrections` with field allowlist, hashed-IP rate limit, append-only JSONL store at `vault/corrections/`; UI affordance in detail panel
- [ ] **Live collaborative analysis surface** *(in progress)* — anonymous per-session presence, common chat channel, ability to tag and share files / pages / photos / filter URLs in the chat, follow-mode that mirrors another visitor's view (so groups can investigate together in real time). Privacy: free-text inputs (search box, ask widget) are scrubbed from broadcast to avoid leaks.

## Disclosure Globe vision

The next iteration of the project pivots from a single-source mirror to a
**multi-source aggregator** — a tactical-command-centre interactive globe
that pulls together declassified UAP/UFO documentation across:

- **PURSUE / war.gov** (this mirror — Release 1+)
- **AARO** (All-domain Anomaly Resolution Office)
- **NUFORC** (National UFO Reporting Center, civilian)
- **MUFON** (Mutual UFO Network)
- **Project Blue Book** (USAF, 1947–1969)
- **FBI Vault** UFO collection
- **NARA** (National Archives)
- **UK MoD** released files
- **French GEIPAN** (CNES civilian sightings)

Single searchable interactive archive with a Cesium-based 3D globe at
`/`, hover/click on incident markers, slide-in dossiers, decade-scale
timeline scrubber, and a tactical HUD.

Working title: **Disclosure Globe**. Sibling repo: `../disclosure-globe`
(Next.js 15 + Cesium + Tailwind v4). When the prototype is solid, the
search UI here will move under `/archive` and the globe becomes the
front door at `/`.

When subsequent disclosure tranches drop, the cron auto-detects the CSV
change and runs the pipeline. Cross-tranche relationship links update
automatically.

## Files of interest

- [`scripts/parse_csv.py`](scripts/parse_csv.py) — CSV → records.json
- [`scripts/download.py`](scripts/download.py) — concurrent fetcher with idempotent skip
- [`scripts/ocr.py`](scripts/ocr.py) — pdftotext + ocrmypdf fallback
- [`scripts/build_links.py`](scripts/build_links.py) — TF-IDF + pHash similarity
- [`scripts/build_search_index.py`](scripts/build_search_index.py) — JSON for the UI
- [`scripts/build_vault.py`](scripts/build_vault.py) — Obsidian markdown
- [`ui/index.html`](ui/index.html) — single-page search UI (MiniSearch)

#!/usr/bin/env python3
"""Build search-index.json consumed by ui/index.html.

For each record we pack:
  id, title, agency, type, dates, location, blurb, sources,
  text (joined extracted text, truncated to ~100k chars per record so the
        full payload stays manageable),
  thumbs[], primary[], video.
"""
import json
import re
from pathlib import Path

from build_thumbs import thumb_path_for

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
TEXT = RAW / "text"
RECORDS = json.loads((RAW / "records.json").read_text())
LINKS_PATH = RAW / "links.json"
LINKS = json.loads(LINKS_PATH.read_text()) if LINKS_PATH.exists() else {}

DOSSIERS_PATH = ROOT / "ui" / "dossiers.json"
DOSSIERS = json.loads(DOSSIERS_PATH.read_text()) if DOSSIERS_PATH.exists() else []
# Pre-compile each pattern individually so we can attribute hits back to the
# specific keyword that fired — needed for the UI's match-details display.
_DOSSIER_PATTERNS: list[tuple[str, list[tuple[str, re.Pattern]]]] = []
for _d in DOSSIERS:
    pats: list[tuple[str, re.Pattern]] = []
    for raw in _d.get("keywords_ci", []):
        pats.append((raw, re.compile(raw, re.IGNORECASE)))
    for raw in _d.get("keywords_cs", []):
        pats.append((raw, re.compile(raw)))
    _DOSSIER_PATTERNS.append((_d["id"], pats))

CTX_BEFORE = 50
CTX_AFTER = 60
MAX_HITS_PER_DOSSIER = 8


def dossier_hits_for(text: str) -> dict[str, list[dict]]:
    """Per-dossier list of {kw, pat, ctx} for the first occurrence of each
    matching keyword. Empty dict if no dossier matches."""
    if not text:
        return {}
    out: dict[str, list[dict]] = {}
    for did, pats in _DOSSIER_PATTERNS:
        hits: list[dict] = []
        for raw, rx in pats:
            m = rx.search(text)
            if not m:
                continue
            s = max(0, m.start() - CTX_BEFORE)
            e = min(len(text), m.end() + CTX_AFTER)
            ctx = text[s:e].replace("\n", " ").replace("  ", " ").strip()
            hits.append({"kw": m.group(), "pat": raw, "ctx": ctx})
            if len(hits) >= MAX_HITS_PER_DOSSIER:
                break
        if hits:
            out[did] = hits
    return out


def small_thumb(src_rel: str) -> str:
    """Path to generated thumb if it exists on disk, else "" (UI falls back to full-size)."""
    if not src_rel:
        return ""
    dst = thumb_path_for(src_rel)
    if dst is None or not dst.exists():
        return ""
    return str(dst.relative_to(ROOT))

# Group extracted-image entries by their src_pdf so we can surface them per record
EXTRACTED_PATH = RAW / "extracted_images.json"
EXTRACTED_BY_PDF: dict[str, list[str]] = {}
if EXTRACTED_PATH.exists():
    for e in json.loads(EXTRACTED_PATH.read_text()):
        EXTRACTED_BY_PDF.setdefault(e["src_pdf"], []).append(e["file"])
    # Sort by filename so order is stable (page-then-seq)
    for k in EXTRACTED_BY_PDF:
        EXTRACTED_BY_PDF[k].sort()

PER_REC_TEXT_CAP = 100_000


def text_for(rec):
    chunks = []
    for lp in rec.get("primary_local", []):
        if lp.endswith(".pdf"):
            t = TEXT / (Path(lp).stem + ".txt")
            if t.exists():
                chunks.append(t.read_text(errors="replace"))
    if not chunks:
        return ""
    joined = "\n\n".join(chunks)
    if len(joined) > PER_REC_TEXT_CAP:
        joined = joined[:PER_REC_TEXT_CAP] + "\n\n[…truncated…]"
    return joined


docs = []
records_by_id: dict[str, dict] = {}
for r in RECORDS:
    extracted: list[str] = []
    for lp in r.get("primary_local", []):
        if lp.endswith(".pdf"):
            extracted.extend(EXTRACTED_BY_PDF.get(lp, []))
    records_by_id[r["id"]] = r
    # Pick the source image the row/grid will display; same logic as the UI fallback chain.
    thumbs = r.get("thumbnail_local", []) or []
    primary_imgs = [p for p in r.get("primary_local", []) or [] if p.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))]
    row_src = (thumbs[0] if thumbs else (primary_imgs[0] if primary_imgs else ""))
    rec_text = text_for(r)
    rec_hits = dossier_hits_for(rec_text)
    rec_dossiers = list(rec_hits.keys())
    docs.append({
        "id": r["id"],
        "title": r["title"],
        "agency": r["agency"],
        "type": r["type"].strip(),
        "release_date": r["release_date"],
        "incident_date": r["incident_date"],
        "incident_location": r["incident_location"],
        "redaction": r["redaction"],
        "blurb": r["blurb"],
        "source_url": r.get("pdf_image_link", ""),
        "primary_local": r.get("primary_local", []),
        "thumbnail_local": r.get("thumbnail_local", []),
        "thumb_small": small_thumb(row_src),
        "video_local": r.get("video_local", ""),
        "dvids_video_id": r.get("dvids_video_id", ""),
        "text": rec_text,
        "similar_text":  LINKS.get(r["id"], {}).get("similar_text",  []),
        "similar_image": LINKS.get(r["id"], {}).get("similar_image", []),
        "extracted_images": extracted,
        "dossiers": rec_dossiers,
        "dossier_hits": rec_hits,
    })
    # Stash parent dossiers so synthetic IMG records can inherit them
    r["_dossiers"] = rec_dossiers

# Synthesize one IMG record per extracted image so the Images filter
# surfaces them at the top level (not just hidden in their parent's gallery).
# Each synthetic record inherits its parent's text/blurb/agency so search
# still hits them via parent metadata; click opens a detail view focused
# on the single image.
extracted_records = 0
# Map raw/docs/<stem>.pdf -> records that have it as primary_local
pdf_to_record_id: dict[str, str] = {}
for r in RECORDS:
    for lp in r.get("primary_local", []):
        if lp.endswith(".pdf"):
            pdf_to_record_id[lp] = r["id"]

if EXTRACTED_PATH.exists():
    for entry in json.loads(EXTRACTED_PATH.read_text()):
        parent_id = pdf_to_record_id.get(entry["src_pdf"])
        if not parent_id:
            continue
        parent = records_by_id[parent_id]
        page = entry.get("page", "?")
        seq = Path(entry["file"]).stem.rsplit("_", 1)[-1]
        synthetic_id = f"{parent_id}__img_p{page}_{seq}"
        docs.append({
            "id": synthetic_id,
            "title": f"{parent['title']} — image (page {page})",
            "agency": parent["agency"],
            "type": "IMG",
            "release_date": parent["release_date"],
            "incident_date": parent["incident_date"],
            "incident_location": parent["incident_location"],
            "redaction": parent["redaction"],
            "blurb": f"Embedded image extracted from page {page} of {parent['title']}. " + (parent.get("blurb", "") or ""),
            "source_url": parent.get("pdf_image_link", ""),
            "primary_local": [entry["file"]],
            "thumbnail_local": [entry["file"]],
            "thumb_small": small_thumb(entry["file"]),
            "video_local": "",
            "dvids_video_id": "",
            "text": "",
            "similar_text":  [],
            "similar_image": [],
            "extracted_images": [],
            "dossiers": parent.get("_dossiers", []),
            "extracted_from_id": parent_id,
            "extracted_page": page,
        })
        extracted_records += 1

# ─── Project Blue Book (USAF, 1947-1969) ──────────────────────────────
# Pull in NICAP-mirrored "Unknown" cases as additional records, tagged
# with provenance="official_us_military". They have no PDFs/OCR text, so
# the text-heavy features (dossiers, links, related) quietly do nothing —
# but they appear in search/filter, render in the detail panel, and pick
# up civilian (NUFORC) correlations when nearby in space+time.
BLUE_BOOK_PATH = RAW / "sightings" / "blue_book" / "sightings.json"
blue_book_records = 0
if BLUE_BOOK_PATH.exists():
    bb = json.loads(BLUE_BOOK_PATH.read_text())
    for r in bb:
        case_no = r.get("source_id", "")
        # Use underscore form in search-index ids (URL-safer than colons).
        ui_id = f"blue_book_{case_no}"
        loc = r.get("location", {}) or {}
        docs.append({
            "id": ui_id,
            "title": r.get("title", f"USAF Blue Book case #{case_no}"),
            "agency": "USAF Project Blue Book",
            "type": "Case File",
            "release_date": "1969-12-17",  # Blue Book program closeout
            "incident_date": r.get("raw", {}).get("date_text", ""),
            "incident_location": loc.get("name", ""),
            "redaction": "",
            "blurb": r.get("summary", ""),
            "source_url": r.get("source_url", ""),
            "primary_local": [],
            "thumbnail_local": [],
            "thumb_small": "",
            "video_local": "",
            "dvids_video_id": "",
            "text": "",
            "similar_text":  [],
            "similar_image": [],
            "extracted_images": [],
            "dossiers": [],
            "dossier_hits": {},
            "provenance": "official_us_military",
            "provenance_source": "blue_book",
            "case_number": case_no,
        })
        blue_book_records += 1

out = ROOT / "ui" / "search-index.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(docs, ensure_ascii=False))
print(
    f"Wrote {out} — {len(docs)} records "
    f"({extracted_records} synthetic IMG, {blue_book_records} Blue Book), "
    f"{sum(len(d['text']) for d in docs):,} text chars total"
)

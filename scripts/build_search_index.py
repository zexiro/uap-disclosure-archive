#!/usr/bin/env python3
"""Build search-index.json consumed by ui/index.html.

For each record we pack:
  id, title, agency, type, dates, location, blurb, sources,
  text (joined extracted text, truncated to ~100k chars per record so the
        full payload stays manageable),
  thumbs[], primary[], video.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
TEXT = RAW / "text"
RECORDS = json.loads((RAW / "records.json").read_text())
LINKS_PATH = RAW / "links.json"
LINKS = json.loads(LINKS_PATH.read_text()) if LINKS_PATH.exists() else {}

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
        "video_local": r.get("video_local", ""),
        "dvids_video_id": r.get("dvids_video_id", ""),
        "text": text_for(r),
        "similar_text":  LINKS.get(r["id"], {}).get("similar_text",  []),
        "similar_image": LINKS.get(r["id"], {}).get("similar_image", []),
        "extracted_images": extracted,
    })

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
            "video_local": "",
            "dvids_video_id": "",
            "text": "",
            "similar_text":  [],
            "similar_image": [],
            "extracted_images": [],
            "extracted_from_id": parent_id,
            "extracted_page": page,
        })
        extracted_records += 1

out = ROOT / "ui" / "search-index.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(docs, ensure_ascii=False))
print(f"Wrote {out} — {len(docs)} records ({extracted_records} synthetic IMG from extracted PDF images), {sum(len(d['text']) for d in docs):,} text chars total")

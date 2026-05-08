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
for r in RECORDS:
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
    })

out = ROOT / "ui" / "search-index.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(docs, ensure_ascii=False))
print(f"Wrote {out} — {len(docs)} records, {sum(len(d['text']) for d in docs):,} text chars total")

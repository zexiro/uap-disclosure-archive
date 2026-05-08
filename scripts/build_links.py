#!/usr/bin/env python3
"""Compute relationship links between records.

Two layers:
  1. Text similarity — TF-IDF over (title + blurb + extracted text), cosine.
     Captures conceptual neighbours: shared FBI cases, same incident location,
     same mission-report series, etc.
  2. Image similarity — perceptual hash (pHash) over thumbnails + primary
     images. Captures near-duplicates: same photo cropped/recompressed, same
     scene from a different angle in some pairs.

Output: raw/links.json
{
  "<record_id>": {
    "similar_text":  [{"id": "...", "score": 0.42, "title": "..."}],
    "similar_image": [{"id": "...", "score": 12,   "title": "..."}]
  },
  ...
}
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
TEXT = RAW / "text"
RECORDS = json.loads((RAW / "records.json").read_text())
LINKS_PATH = RAW / "links.json"

TOP_N_TEXT = 6
MIN_TEXT_SCORE = 0.10
TOP_N_IMAGE = 4
MAX_IMG_HASH_DISTANCE = 16  # 0 = identical pHash; ≤8 = near-dup; ≤16 = similar


def text_for(rec):
    """Concatenate title + blurb + all extracted PDF text for one record."""
    chunks = [rec.get("title", ""), rec.get("blurb", ""), rec.get("agency", ""),
              rec.get("incident_location", "")]
    for lp in rec.get("primary_local", []):
        if lp.endswith(".pdf"):
            t = TEXT / (Path(lp).stem + ".txt")
            if t.exists():
                chunks.append(t.read_text(errors="replace"))
    return "\n".join(chunks)


def compute_text_links():
    docs_text = [text_for(r) for r in RECORDS]
    if not any(d.strip() for d in docs_text):
        return {r["id"]: [] for r in RECORDS}

    vec = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs_text)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)

    links = {}
    for i, rec in enumerate(RECORDS):
        order = np.argsort(-sim[i])
        out = []
        for j in order[: TOP_N_TEXT * 3]:
            score = float(sim[i, j])
            if score < MIN_TEXT_SCORE:
                break
            other = RECORDS[j]
            # Avoid linking a record to a near-duplicate of itself (e.g. same blurb)
            out.append({
                "id": other["id"],
                "title": other["title"],
                "type": other["type"].strip(),
                "score": round(score, 3),
            })
            if len(out) >= TOP_N_TEXT:
                break
        links[rec["id"]] = out
    return links


def compute_image_links():
    """For each record, compute pHash over (a) its primary + thumbnail images
    and (b) any images extracted from its source PDF (raw/extracted_images.json),
    so a record's hash pool includes the photos hidden inside its FBI scan etc.
    Records whose hashes are close are linked."""
    # Map src_pdf path -> record_index, so extracted images attach to their parent record
    pdf_to_rec: dict[str, int] = {}
    for i, rec in enumerate(RECORDS):
        for lp in rec.get("primary_local", []):
            if lp.endswith(".pdf"):
                pdf_to_rec[lp] = i

    hashes = []  # list[(record_index, hash, src_path)]
    for i, rec in enumerate(RECORDS):
        candidates = list(rec.get("primary_local", [])) + list(rec.get("thumbnail_local", []))
        for lp in candidates:
            if not lp.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                continue
            full = ROOT / lp
            if not full.exists():
                continue
            try:
                with Image.open(full) as im:
                    im = im.convert("RGB")
                    h = imagehash.phash(im, hash_size=16)
                hashes.append((i, h, lp))
            except (UnidentifiedImageError, OSError, ValueError):
                continue

    # Add embedded images extracted from PDFs — index file already has pHash strings
    extracted_path = RAW / "extracted_images.json"
    extracted_count = 0
    if extracted_path.exists():
        for entry in json.loads(extracted_path.read_text()):
            rec_idx = pdf_to_rec.get(entry["src_pdf"])
            if rec_idx is None:
                continue
            try:
                h = imagehash.hex_to_hash(entry["phash"])
            except (ValueError, KeyError):
                continue
            hashes.append((rec_idx, h, entry["file"]))
            extracted_count += 1
        print(f"  + {extracted_count} hashes from extracted PDF images")

    links = {r["id"]: [] for r in RECORDS}
    if not hashes:
        return links

    # Build per-record best score against every other record
    by_record: dict[int, dict[int, int]] = {i: {} for i in range(len(RECORDS))}
    for a in range(len(hashes)):
        ia, ha, _ = hashes[a]
        for b in range(a + 1, len(hashes)):
            ib, hb, _ = hashes[b]
            if ia == ib:
                continue
            d = ha - hb  # Hamming distance
            if d <= MAX_IMG_HASH_DISTANCE:
                cur_a = by_record[ia].get(ib, 999)
                cur_b = by_record[ib].get(ia, 999)
                if d < cur_a:
                    by_record[ia][ib] = d
                if d < cur_b:
                    by_record[ib][ia] = d

    for i, rec in enumerate(RECORDS):
        ranked = sorted(by_record[i].items(), key=lambda kv: kv[1])[:TOP_N_IMAGE]
        out = []
        for j, dist in ranked:
            other = RECORDS[j]
            out.append({
                "id": other["id"],
                "title": other["title"],
                "type": other["type"].strip(),
                "distance": int(dist),
            })
        links[rec["id"]] = out
    return links


def main():
    print("Computing text similarity (TF-IDF cosine)…")
    text_links = compute_text_links()
    print(f"  records with ≥1 text link: {sum(1 for v in text_links.values() if v)}")

    print("Computing image similarity (pHash)…")
    image_links = compute_image_links()
    print(f"  records with ≥1 image link: {sum(1 for v in image_links.values() if v)}")

    out = {}
    for rec in RECORDS:
        rid = rec["id"]
        out[rid] = {
            "similar_text": text_links.get(rid, []),
            "similar_image": image_links.get(rid, []),
        }
    LINKS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {LINKS_PATH}")


if __name__ == "__main__":
    main()

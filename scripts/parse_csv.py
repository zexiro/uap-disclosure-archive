#!/usr/bin/env python3
"""Normalize uap-csv.csv into records.json + downloads.tsv.

records.json — one object per release item, normalized columns.
downloads.tsv — url<TAB>local_path lines for the downloader to consume.
"""
import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
CSV_PATH = RAW / "csv" / "uap-csv.csv"
RECORDS_PATH = RAW / "records.json"
DOWNLOADS_PATH = RAW / "downloads.tsv"


def slugify(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")[:160] or "untitled"


def split_links(cell: str):
    """Split a cell into URLs. URLs may contain literal spaces (war.gov leaves
    them unencoded), so split on pipes / newlines / consecutive whitespace
    AROUND http(s) boundaries — not on every whitespace char."""
    if not cell:
        return []
    s = cell.strip()
    # Normalize separators between URLs to a single pipe
    s = re.sub(r"\s*\|\s*", "|", s)
    s = re.sub(r"\r?\n+", "|", s)
    # Multiple URLs concatenated by whitespace? Only split when whitespace
    # is followed by "http"
    s = re.sub(r"\s+(?=https?://)", "|", s)
    parts = [p.strip() for p in s.split("|")]
    return [p for p in parts if p.startswith("http")]


def local_path_for(url: str, kind: str) -> Path:
    """Map a URL to a path under raw/<kind>/ preserving the war.gov filename.

    URLs may contain unencoded spaces, so we slugify on the unquoted basename.
    """
    name = unquote(urlparse(url).path).rsplit("/", 1)[-1] or "file"
    # Preserve extension explicitly (slugify might mangle it)
    if "." in name:
        stem, _, ext = name.rpartition(".")
        name = slugify(stem) + "." + slugify(ext)
    else:
        name = slugify(name)
    return RAW / kind / name


def main():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    records = []
    downloads = []  # list[(url, local_path)]
    seen_urls = set()

    def queue(url: str, kind: str) -> str:
        url = url.strip()
        if not url.startswith("http"):
            return ""
        path = local_path_for(url, kind)
        if url not in seen_urls:
            seen_urls.add(url)
            downloads.append((url, str(path.relative_to(ROOT))))
        return str(path.relative_to(ROOT))

    for raw_row in rows[1:]:
        # Pad to header width
        row = (raw_row + [""] * len(header))[: len(header)]
        rec = {
            "release_date": row[1].strip(),
            "title": row[2].strip().replace("\n", " ").strip(),
            "type": row[3].strip(),
            "blurb": row[6].strip(),
            "dvids_video_id": row[7].strip(),
            "video_title": row[8].strip(),
            "agency": row[9].strip(),
            "incident_date": row[11 - 1].strip() if len(row) > 10 else "",
            "incident_location": row[11].strip(),
        }
        # Re-fix: header positions
        rec["incident_date"] = row[10].strip()
        rec["incident_location"] = row[11].strip()
        rec["pdf_image_link"] = row[12].strip()
        rec["modal_image"] = row[13].strip()
        rec["redaction"] = row[0].strip()
        rec["video_pairing"] = row[4].strip()
        rec["pdf_pairing"] = row[5].strip()

        if not rec["title"]:
            continue

        rec["id"] = slugify(rec["title"])

        # Normalize URLs and queue downloads
        # Kind is inferred per-URL from extension (not record type), because
        # VID records often link to a *paired* PDF mission report.
        def kind_for(u: str) -> str:
            ul = u.lower()
            if ul.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                return "images"
            if ul.endswith((".mp4", ".mov", ".webm", ".m4v")):
                return "videos"
            return "docs"  # PDFs and unknowns

        primary_links = split_links(rec["pdf_image_link"])
        # Heuristic fixup: if a primary link looks malformed (no extension,
        # contains "+M" garbage), try to derive it from the modal_image
        # thumbnail URL (which follows the pattern .../thumbnail/<basename>.jpg
        # → .../<basename>.pdf for FBI items).
        thumb_links = split_links(rec["modal_image"])
        cleaned_primary = []
        for u in primary_links:
            looks_bad = ("+M" in u) or (u.lower().rstrip("/").rsplit("/", 1)[-1].count(".") == 0)
            if looks_bad and thumb_links:
                t = thumb_links[0]
                if "/thumbnail/" in t and t.lower().endswith((".jpg", ".jpeg", ".png")):
                    derived = t.replace("/thumbnail/", "/")
                    derived = re.sub(r"\.(jpg|jpeg|png)$", ".pdf", derived, flags=re.I)
                    cleaned_primary.append(derived)
                    continue
            cleaned_primary.append(u)
        primary_links = cleaned_primary
        rec["primary_local"] = []
        for u in primary_links:
            lp = queue(u, kind_for(u))
            if lp:
                rec["primary_local"].append(lp)

        modal_links = split_links(rec["modal_image"])
        rec["thumbnail_local"] = []
        for u in modal_links:
            lp = queue(u, "images")
            if lp:
                rec["thumbnail_local"].append(lp)

        # Pairings can also reference URLs
        for pair_field in ("video_pairing", "pdf_pairing"):
            for u in split_links(rec[pair_field]):
                kind = "videos" if pair_field == "video_pairing" else "docs"
                lp = queue(u, kind)
                if lp:
                    rec.setdefault("paired_local", []).append(lp)

        records.append(rec)

    # Disambiguate duplicate IDs (when two records have the same title)
    seen_ids: dict[str, int] = {}
    for r in records:
        base = r["id"]
        n = seen_ids.get(base, 0) + 1
        seen_ids[base] = n
        if n > 1:
            r["id"] = f"{base}__{n}"
            r["title"] = f"{r['title']} ({n})"

    RECORDS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    with DOWNLOADS_PATH.open("w") as f:
        for url, local in downloads:
            f.write(f"{url}\t{local}\n")

    print(f"records: {len(records)}")
    print(f"unique downloads: {len(downloads)}")
    by_kind = {}
    for _, p in downloads:
        kind = p.split("/")[1]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"by kind: {by_kind}")


if __name__ == "__main__":
    main()

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

# war.gov release tranches by their CSV "Release Date" value.
RELEASE_BY_DATE = {
    "5/8/26": "release_1",
    "5/22/26": "release_2",
    "6/12/26": "release_3",
    "7/10/26": "release_4",
}


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
    header = [h.strip() for h in rows[0]]

    # Column lookup by header name (war.gov added a "Featured" column in
    # release 4, which shifted every positional index — resolve by name).
    def col(*names: str) -> int:
        for n in names:
            if n in header:
                return header.index(n)
        return -1

    C_RED = col("Redaction")
    C_REL = col("Release Date")
    C_TITLE = col("Title")
    C_TYPE = col("Type")
    C_VPAIR = col("Video Pairing")
    C_PPAIR = col("PDF Pairing")
    C_BLURB = col("Description Blurb")
    C_DVIDS = col("DVIDS Video ID")
    C_VTITLE = col("Video Title")
    C_AGENCY = col("Agency")
    C_IDATE = col("Incident Date")
    C_ILOC = col("Incident Location")
    C_LINK = col("PDF | Image Link")
    C_MODAL = col("Modal Image")
    C_FEATURED = col("Featured")

    def cell(row, idx):
        return row[idx].strip() if 0 <= idx < len(row) else ""

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
            "release_date": cell(row, C_REL),
            "title": cell(row, C_TITLE).replace("\n", " ").strip(),
            "type": cell(row, C_TYPE),
            "blurb": cell(row, C_BLURB),
            "dvids_video_id": cell(row, C_DVIDS),
            "video_title": cell(row, C_VTITLE),
            "agency": cell(row, C_AGENCY),
            "incident_date": cell(row, C_IDATE),
            "incident_location": cell(row, C_ILOC),
        }
        rec["pdf_image_link"] = cell(row, C_LINK)
        rec["modal_image"] = cell(row, C_MODAL)
        rec["redaction"] = cell(row, C_RED)
        rec["video_pairing"] = cell(row, C_VPAIR)
        rec["pdf_pairing"] = cell(row, C_PPAIR)
        rec["featured"] = cell(row, C_FEATURED)
        rec["release"] = RELEASE_BY_DATE.get(rec["release_date"], "")

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

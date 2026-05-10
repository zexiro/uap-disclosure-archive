#!/usr/bin/env python3
"""Pull recent UAP/UFO news headlines from Google News RSS.

GDELT was the original target but rate-limits aggressively (1 req/5s with
ambient throttling) and its results are noisier. Google News RSS gives us
~100 articles per query with title, link, pubDate, and source — enough
for a "recent media chatter" surface alongside the official archive.

Each query returns the freshest results matching the search terms — so we
fan out across a few different phrasings to broaden coverage. Duplicates
(same article URL across queries) are deduped on the way out.

Output:
  raw/sightings/news/raw/<slugified_query>.xml  (raw RSS)
  raw/sightings/news/sightings.json              (normalized)
"""
from __future__ import annotations

import datetime as dt
import email.utils
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.schema import PROVENANCE_FOR_SOURCE, verification_for  # noqa: E402

OUT_DIR = ROOT / "raw" / "sightings" / "news"
RAW_DIR = OUT_DIR / "raw"
NORMALIZED = OUT_DIR / "sightings.json"

# Different phrasings catch overlapping but not-identical article sets.
# Keep the list short — each query returns up to 100 items, dedup is
# cheap, and Google News gives diminishing returns past ~5 queries.
QUERIES = [
    'UAP "unidentified aerial phenomenon"',
    "UFO sighting",
    "Pentagon UAP",
    "AARO UAP",
    "UAP disclosure",
]

USER_AGENT = "disclosure-archive/1.0 (multi-source UAP aggregator)"
RSS_BASE = "https://news.google.com/rss/search"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "query"


def fetch_query(q: str) -> str:
    raw_path = RAW_DIR / f"{slugify(q)}.xml"
    if raw_path.exists():
        print(f"[news] cached query {q!r}: {raw_path.relative_to(ROOT)}")
        return raw_path.read_text(encoding="utf-8")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = f"{RSS_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", errors="replace")
    raw_path.write_text(body, encoding="utf-8")
    print(f"[news] fetched {q!r} → {raw_path.relative_to(ROOT)} ({len(body):,} bytes)")
    time.sleep(2.0)  # be nice to Google
    return body


def parse_rss(xml_text: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"[news] RSS parse error: {e}")
        return out
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        guid = (item.findtext("guid") or link).strip()
        # <source> isn't a Python-friendly tag; handled separately
        source_el = item.find("source")
        source_name = (source_el.text if source_el is not None else "") or ""
        # description is HTML; just take the visible source name
        out.append({
            "title": html.unescape(title),
            "link": link,
            "guid": guid,
            "pub_date": pub,
            "source_name": html.unescape(source_name).strip(),
        })
    return out


def parse_pub_date(raw: str) -> tuple[str, str]:
    """RFC 2822 (Mon, 09 May 2026 14:23:00 GMT)."""
    if not raw:
        return "", "unknown"
    try:
        d = email.utils.parsedate_to_datetime(raw)
        return d.isoformat(), "minute"
    except (TypeError, ValueError):
        return "", "unknown"


def normalize_article(art: dict, fetched_at: str) -> dict | None:
    title = art.get("title", "").strip()
    link = art.get("link", "").strip()
    if not title or not link:
        return None
    occurred, precision = parse_pub_date(art.get("pub_date", ""))
    # Strip the trailing " - <Source>" Google appends to titles for cleaner display.
    src = art.get("source_name", "").strip()
    display_title = title
    if src and title.endswith(f" - {src}"):
        display_title = title[: -(len(src) + 3)].strip()

    prov = PROVENANCE_FOR_SOURCE["news"]
    rec = {
        "id": f"news:{abs(hash(art['guid']))}",
        "source": "news",
        "source_id": art["guid"],
        "source_url": link,                  # google news redirector; the real publisher URL is one hop in
        "fetched_at": fetched_at,
        "provenance": prov,
        "verification_status": verification_for(prov),
        "title": display_title[:240],
        "summary": display_title[:280],      # Google News doesn't ship article bodies
        "occurred_at": occurred,             # publication time
        "occurred_at_precision": precision,
        "media": [],
        "raw": {
            "outlet": src,
            "rss_pub_date": art.get("pub_date", ""),
            "guid": art.get("guid", ""),
        },
    }
    return {k: v for k, v in rec.items() if v not in ("", [], {}, 0)}


def main() -> int:
    fetched_at = dt.date.today().isoformat()
    seen_guids: set[str] = set()
    sightings: list[dict] = []

    for q in QUERIES:
        body = fetch_query(q)
        for art in parse_rss(body):
            if not art["guid"] or art["guid"] in seen_guids:
                continue
            seen_guids.add(art["guid"])
            rec = normalize_article(art, fetched_at)
            if rec:
                sightings.append(rec)

    NORMALIZED.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED.open("w") as f:
        json.dump(sightings, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[news] normalized {len(sightings)} unique articles → {NORMALIZED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Merge every source into raw/sightings/sightings.json.

Currently consumes:
  - raw/records.json                        (war.gov, projected as official_us_military)
  - raw/sightings/nuforc/sightings.json     (NUFORC civilian)
  - raw/sightings/blue_book/sightings.json  (USAF Project Blue Book "Unknown" cases)
  - raw/sightings/reddit/sightings.json     (r/UFOs, r/UFOB, r/HighStrangeness — media_unverified)
  - raw/sightings/news/sightings.json       (Google News RSS — media_unverified)

To add a new source: write its fetcher under scripts/sightings/, normalize
to the schema in scripts/sightings/schema.py, and add a load step here.

Locations from sources that don't ship lat/lng are geocoded via
scripts/sightings/geocode.py with a file-backed cache, so subsequent runs
don't hit the network. The cache is committed.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.geocode import geocode  # noqa: E402
from sightings.schema import PROVENANCE_FOR_SOURCE, verification_for  # noqa: E402

WARGOV_RECORDS = ROOT / "raw" / "records.json"
NUFORC_SIGHTINGS = ROOT / "raw" / "sightings" / "nuforc" / "sightings.json"
BLUE_BOOK_SIGHTINGS = ROOT / "raw" / "sightings" / "blue_book" / "sightings.json"
REDDIT_SIGHTINGS = ROOT / "raw" / "sightings" / "reddit" / "sightings.json"
NEWS_SIGHTINGS = ROOT / "raw" / "sightings" / "news" / "sightings.json"
OUT = ROOT / "raw" / "sightings" / "sightings.json"

CURRENT_YEAR_2 = dt.date.today().year % 100  # 26 in 2026


def parse_wargov_date(raw: str) -> tuple[str, str]:
    """war.gov `incident_date` is M/D/YY, M/D/YYYY, or M/D/YYYY-M/D/YYYY.
    For ranges we anchor on the start date. 2-digit years > today's
    2-digit year are treated as 19xx (so "85" → 1985, "24" → 2024)."""
    if not raw or raw == "N/A":
        return "", "unknown"
    raw = raw.strip()
    if "-" in raw and len(raw) > 10:
        raw = raw.split("-", 1)[0].strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", raw)
    if not m:
        return "", "unknown"
    mo, da, yr = (int(x) for x in m.groups())
    if yr < 100:
        yr = 1900 + yr if yr > CURRENT_YEAR_2 else 2000 + yr
    try:
        return dt.date(yr, mo, da).isoformat(), "day"
    except ValueError:
        return "", "unknown"


def project_wargov(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    fetched_at = dt.date.today().isoformat()
    geo_misses = 0
    for r in records:
        loc_raw = (r.get("incident_location") or "").strip()
        location: dict = {}
        if loc_raw and loc_raw != "N/A":
            location["name"] = loc_raw
            geo = geocode(loc_raw)
            if geo:
                location.update(geo)
            else:
                geo_misses += 1

        occurred, precision = parse_wargov_date(r.get("incident_date", ""))

        media: list[dict] = []
        if r.get("pdf_image_link"):
            media.append({"type": "document", "url": r["pdf_image_link"]})
        for local in r.get("primary_local") or []:
            media.append({"type": "document", "local_path": local})
        for thumb in r.get("thumbnail_local") or []:
            media.append({"type": "photo", "local_path": thumb})

        prov = PROVENANCE_FOR_SOURCE["wargov"]
        rec = {
            "id": f"wargov:{r['id']}",
            "source": "wargov",
            "source_id": r["id"],
            "source_url": r.get("pdf_image_link", ""),
            "fetched_at": fetched_at,
            "provenance": prov,
            "verification_status": verification_for(prov),
            "title": r.get("title", "") or r["id"],
            "summary": (r.get("blurb", "") or "")[:280],
            "text": r.get("blurb", "") or "",
            "occurred_at": occurred,
            "occurred_at_precision": precision,
            "location": location,
            "media": media,
            "raw": {
                "agency": r.get("agency", ""),
                "release_date": r.get("release_date", ""),
                "type": r.get("type", ""),
            },
        }
        out.append({k: v for k, v in rec.items() if v not in ("", [], {})})
    print(f"[unified] projected {len(out)} war.gov records ({geo_misses} location strings unresolved)")
    return out


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[unified] {path.relative_to(ROOT)} missing — skipping")
        return []
    return json.loads(path.read_text())


def geocode_in_place(records: list[dict], label: str) -> None:
    """Add lat/lng/country to each record's location dict by geocoding
    its `name`. Skips records that already have lat/lng or no name."""
    misses = 0
    hits = 0
    for r in records:
        loc = r.get("location") or {}
        name = (loc.get("name") or "").strip()
        if not name or "lat" in loc:
            continue
        geo = geocode(name)
        if geo:
            loc.update(geo)
            r["location"] = loc
            hits += 1
        else:
            misses += 1
    print(f"[unified] geocoded {label}: {hits} resolved, {misses} unresolved")


def main() -> int:
    wargov = project_wargov(load_json(WARGOV_RECORDS))
    nuforc = load_json(NUFORC_SIGHTINGS)
    print(f"[unified] loaded {len(nuforc):,} NUFORC sightings")
    blue_book = load_json(BLUE_BOOK_SIGHTINGS)
    print(f"[unified] loaded {len(blue_book)} Blue Book sightings (geocoding…)")
    geocode_in_place(blue_book, "blue_book")
    reddit = load_json(REDDIT_SIGHTINGS)
    print(f"[unified] loaded {len(reddit):,} Reddit posts")
    news = load_json(NEWS_SIGHTINGS)
    print(f"[unified] loaded {len(news):,} news articles")

    sightings = wargov + blue_book + nuforc + reddit + news
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(sightings, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[unified] wrote {len(sightings):,} sightings → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

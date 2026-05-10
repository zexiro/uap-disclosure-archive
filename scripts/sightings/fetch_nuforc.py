#!/usr/bin/env python3
"""Pull NUFORC civilian sightings into the unified schema.

Source: planetsig/ufo-reports — geocoded, time-standardized NUFORC dump.
80,332 reports, 1906-2014, with lat/lng pre-resolved. NUFORC's own site
forbids redistribution; this is the most-cited public mirror. License is
unspecified upstream, treated as fair-use research data here.

Coverage stops at 2014 — newer civilian reports come from Reddit / news
sources in a sibling fetcher. NOT a substitute for a fresh NUFORC scrape.

Output:
  raw/sightings/nuforc/raw/ufo-scrubbed-geocoded-time-standardized.csv
  raw/sightings/nuforc/sightings.json
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.schema import PROVENANCE_FOR_SOURCE, verification_for  # noqa: E402

OUT_DIR = ROOT / "raw" / "sightings" / "nuforc"
RAW_CSV = OUT_DIR / "raw" / "ufo-scrubbed-geocoded-time-standardized.csv"
NORMALIZED = OUT_DIR / "sightings.json"

CSV_URL = (
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/"
    "csv-data/ufo-scrubbed-geocoded-time-standardized.csv"
)

REPORT_LINK_BASE = "https://nuforc.org/databank/"

# Headerless CSV from planetsig — column order is fixed.
COLUMNS = (
    "datetime",
    "city",
    "state",
    "country",
    "shape",
    "duration_seconds",
    "duration_text",
    "comments",
    "date_posted",
    "latitude",
    "longitude",
)


def fetch_csv() -> Path:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"[nuforc] using cached {RAW_CSV.relative_to(ROOT)}")
        return RAW_CSV
    print(f"[nuforc] downloading {CSV_URL}")
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "disclosure-archive/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    RAW_CSV.write_bytes(data)
    print(f"[nuforc] wrote {len(data):,} bytes → {RAW_CSV.relative_to(ROOT)}")
    return RAW_CSV


def parse_datetime(raw: str) -> tuple[str, str]:
    """planetsig uses M/D/YYYY HH:MM. Returns (iso, precision)."""
    raw = (raw or "").strip()
    if not raw:
        return "", "unknown"
    for fmt, precision in (
        ("%m/%d/%Y %H:%M", "minute"),
        ("%m/%d/%Y", "day"),
    ):
        try:
            return dt.datetime.strptime(raw, fmt).isoformat(), precision
        except ValueError:
            continue
    return "", "unknown"


def parse_duration(seconds: str, text: str) -> int:
    try:
        s = int(float(seconds))
        if s > 0:
            return s
    except (ValueError, TypeError):
        pass
    return 0


def clean_text(s: str) -> str:
    """planetsig escapes HTML entities (&#44; &amp;) — undo that."""
    if not s:
        return ""
    return html.unescape(s).strip()


def location_name(city: str, state: str, country: str) -> str:
    parts = [p.strip() for p in (city, state, country) if p and p.strip()]
    return ", ".join(parts).title() if parts else ""


def normalize_row(row: list[str], idx: int, fetched_at: str) -> dict | None:
    if len(row) != len(COLUMNS):
        return None
    rec = dict(zip(COLUMNS, row))

    occurred, precision = parse_datetime(rec["datetime"])
    summary = clean_text(rec["comments"])
    if not summary and not occurred:
        return None

    location: dict = {}
    name = location_name(rec["city"], rec["state"], rec["country"])
    if name:
        location["name"] = name
    try:
        lat = float(rec["latitude"])
        lng = float(rec["longitude"])
        # planetsig has some 0,0 stragglers — drop those, they're noise
        if not (lat == 0 and lng == 0):
            location["lat"] = lat
            location["lng"] = lng
    except (ValueError, TypeError):
        pass
    if rec["country"]:
        location["country"] = rec["country"].strip().lower()

    source_id = f"{idx:06d}"
    out = {
        "id": f"nuforc:{source_id}",
        "source": "nuforc",
        "source_id": source_id,
        "source_url": REPORT_LINK_BASE,
        "fetched_at": fetched_at,
        "provenance": PROVENANCE_FOR_SOURCE["nuforc"],
        "verification_status": verification_for(PROVENANCE_FOR_SOURCE["nuforc"]),
        "title": _title_from(rec, summary, occurred),
        "summary": summary[:280],
        "text": summary,
        "occurred_at": occurred,
        "occurred_at_precision": precision,
        "location": location,
        "shape": (rec["shape"] or "").strip().lower(),
        "duration_seconds": parse_duration(rec["duration_seconds"], rec["duration_text"]),
        "media": [],
        "raw": {
            "duration_text": clean_text(rec["duration_text"]),
            "date_posted": rec["date_posted"],
        },
    }
    # Strip empty values to keep the JSON tight.
    return {k: v for k, v in out.items() if v not in ("", [], {}, 0)}


def _title_from(rec: dict, summary: str, occurred: str) -> str:
    shape = (rec["shape"] or "").strip().lower()
    where = location_name(rec["city"], rec["state"], rec["country"])
    when = occurred[:10] if occurred else ""
    bits = []
    if shape:
        bits.append(shape.capitalize())
    if where:
        bits.append(f"over {where}")
    if when:
        bits.append(f"({when})")
    title = " ".join(bits).strip()
    if title:
        return title
    return (summary[:80] + "…") if len(summary) > 80 else (summary or "NUFORC sighting")


def main() -> int:
    csv_path = fetch_csv()
    fetched_at = dt.date.today().isoformat()

    sightings: list[dict] = []
    skipped = 0
    with csv_path.open(encoding="utf-8", errors="replace") as f:
        for idx, row in enumerate(csv.reader(f), start=1):
            rec = normalize_row(row, idx, fetched_at)
            if rec is None:
                skipped += 1
                continue
            sightings.append(rec)

    NORMALIZED.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED.open("w") as f:
        json.dump(sightings, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[nuforc] normalized {len(sightings):,} sightings ({skipped} skipped) → {NORMALIZED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

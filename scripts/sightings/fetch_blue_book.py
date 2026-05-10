#!/usr/bin/env python3
"""Pull Project Blue Book "Unknown" cases into the unified schema.

Source: NICAP — National Investigations Committee on Aerial Phenomena.
Their plain-text list at nicap.org/bluebook/bluelist.htm enumerates the
564 documented USAF Project Blue Book cases (out of the canonical 701
"unidentified") with case number, date, and location.

These are the cases the U.S. Air Force *itself* could not explain after
investigation — every entry here is officially "UNKNOWN" per Blue Book's
final disposition, and so they're flagged as official_us_military
provenance with a bbu_unknown=True raw flag for any future filtering.

Coverage: 1947–1969 (Blue Book ran 1952–1969 but inherited earlier
Project SIGN/GRUDGE cases). All US, US territories, or US-bases-overseas.

Output:
  raw/sightings/blue_book/raw/bluelist.html
  raw/sightings/blue_book/sightings.json
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.schema import PROVENANCE_FOR_SOURCE, verification_for  # noqa: E402

OUT_DIR = ROOT / "raw" / "sightings" / "blue_book"
RAW_HTML = OUT_DIR / "raw" / "bluelist.html"
NORMALIZED = OUT_DIR / "sightings.json"

SOURCE_URL = "https://www.nicap.org/bluebook/bluelist.htm"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1)}

# Per-line shape:  "<case#> <date_text>{padding}<location>"
LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Z][a-z]+(?:[\s\-,/A-Za-z\d]*\d{4})|[A-Z][a-z]+\s+\d{4})\s{2,}(.+?)\s*$"
)


def fetch_html() -> str:
    RAW_HTML.parent.mkdir(parents=True, exist_ok=True)
    if RAW_HTML.exists():
        print(f"[blue_book] using cached {RAW_HTML.relative_to(ROOT)}")
        return RAW_HTML.read_text(encoding="utf-8", errors="replace")
    print(f"[blue_book] downloading {SOURCE_URL}")
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "disclosure-archive/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read().decode("utf-8", errors="replace")
    RAW_HTML.write_text(data, encoding="utf-8")
    print(f"[blue_book] wrote {len(data):,} chars → {RAW_HTML.relative_to(ROOT)}")
    return data


def extract_pre(html: str) -> str:
    m = re.search(r"<pre>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    if not m:
        raise SystemExit("[blue_book] no <pre> block found in source HTML")
    body = m.group(1)
    # NICAP uses <br> instead of newlines inside the <pre>.
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>]+>", "", body)  # strip any other inline tags
    return body


def parse_date(raw: str) -> tuple[str, str]:
    """Returns (iso_date, precision). Falls back to year-only when only a
    month or month range is given."""
    raw = raw.strip().rstrip(",")
    # Pull the year from the end (4 digits).
    ym = re.search(r"\b(\d{4})\b\s*$", raw)
    if not ym:
        return "", "unknown"
    year = int(ym.group(1))
    head = raw[:ym.start()].strip().rstrip(",").strip()

    # Try "Month DD" — take the first integer encountered as the day.
    month_match = re.match(r"([A-Z][a-z]+)", head)
    if not month_match:
        return f"{year:04d}-01-01", "year"
    month = MONTHS.get(month_match.group(1))
    if not month:
        return f"{year:04d}-01-01", "year"

    rest = head[month_match.end():]
    day_match = re.search(r"\b(\d{1,2})\b", rest)
    if not day_match:
        # "October 1947" → month-precision (we still emit a date so
        # downstream tools have something — precision flag carries the truth).
        try:
            return dt.date(year, month, 1).isoformat(), "month"
        except ValueError:
            return f"{year:04d}-01-01", "year"

    day = int(day_match.group(1))
    try:
        return dt.date(year, month, day).isoformat(), "day"
    except ValueError:
        return dt.date(year, month, 1).isoformat(), "month"


def parse_lines(text: str) -> list[tuple[str, str, str]]:
    """Returns list of (case_no, date_text, location)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        case_no, date_text, loc = m.groups()
        # Drop "Case missing" stub lines etc.
        if loc.lower().startswith("case missing") or "case missing" in loc.lower():
            continue
        out.append((case_no, date_text.strip(), loc.strip()))
    return out


def normalize(case_no: str, date_text: str, location: str, fetched_at: str) -> dict:
    iso, precision = parse_date(date_text)
    prov = PROVENANCE_FOR_SOURCE["blue_book"]
    rec = {
        "id": f"blue_book:{case_no}",
        "source": "blue_book",
        "source_id": case_no,
        "source_url": SOURCE_URL,
        "fetched_at": fetched_at,
        "provenance": prov,
        "verification_status": verification_for(prov),
        "title": f"USAF Blue Book case #{case_no} — {location} ({date_text})",
        "summary": (
            f"USAF Project Blue Book officially classified this incident as "
            f"UNKNOWN after investigation. Case #{case_no}. {location}, {date_text}."
        ),
        "occurred_at": iso,
        "occurred_at_precision": precision,
        "location": {"name": location},
        "raw": {
            "case_number": case_no,
            "date_text": date_text,
            "bbu_unknown": True,  # USAF final disposition
            "program": "USAF Project Blue Book (1952-1969)",
        },
    }
    return {k: v for k, v in rec.items() if v not in ("", [], {})}


def main() -> int:
    html = fetch_html()
    text = extract_pre(html)
    raw_cases = parse_lines(text)
    print(f"[blue_book] parsed {len(raw_cases)} case rows")

    fetched_at = dt.date.today().isoformat()
    sightings = [normalize(c, d, loc, fetched_at) for c, d, loc in raw_cases]

    NORMALIZED.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED.open("w") as f:
        json.dump(sightings, f, ensure_ascii=False, indent=2)
    print(f"[blue_book] normalized {len(sightings)} sightings → {NORMALIZED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

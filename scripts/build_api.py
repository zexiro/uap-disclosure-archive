#!/usr/bin/env python3
"""Emit a documented public JSON API + Atom feed alongside the static UI.

Run after build_search_index.py + build_features.py. Outputs:

  ui/api/v1/index.json         — API discovery doc
  ui/api/v1/records.json       — full corpus (single payload, sans 'text' to
                                  keep size manageable)
  ui/api/v1/records/<id>.json  — per-record JSON (full, including text)
  ui/api/v1/incidents.json     — curated incidents
  ui/feed.xml                  — Atom feed of records, sorted by release_date
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
INCIDENTS_PATH = ROOT / "ui" / "incidents.json"
API_DIR = ROOT / "ui" / "api" / "v1"
PER_RECORD_DIR = API_DIR / "records"
FEED_PATH = ROOT / "ui" / "feed.xml"

SITE_URL = "https://uapdisclosuremirror.com"
SITE_TITLE = "UFO/UAP Disclosure Archive"
SITE_DESC = "Permanent mirror of the U.S. Department of War's UAP disclosure release."


def parse_release_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%m/%d/%y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def main():
    docs = json.loads(INDEX_PATH.read_text())
    incidents = {}
    if INCIDENTS_PATH.exists():
        incidents = json.loads(INCIDENTS_PATH.read_text()).get("incidents", {})

    API_DIR.mkdir(parents=True, exist_ok=True)
    PER_RECORD_DIR.mkdir(parents=True, exist_ok=True)

    # Discovery doc
    discovery = {
        "name": SITE_TITLE,
        "version": "v1",
        "description": SITE_DESC,
        "tranche": "release-1-2026-05-08",
        "endpoints": {
            "records":      "/ui/api/v1/records.json",
            "record_by_id": "/ui/api/v1/records/{id}.json",
            "incidents":    "/ui/api/v1/incidents.json",
            "feed":         "/ui/feed.xml",
        },
        "license": "CC-BY-4.0 attribution to Disclosure Archive; underlying records are U.S. Government works (public domain).",
    }
    (API_DIR / "index.json").write_text(json.dumps(discovery, indent=2, ensure_ascii=False))

    # Full corpus (without 'text' — that's heavy and lives on per-record endpoints)
    light = []
    for d in docs:
        d2 = {k: v for k, v in d.items() if k != "text"}
        light.append(d2)
    (API_DIR / "records.json").write_text(json.dumps(light, ensure_ascii=False))
    print(f"  wrote /ui/api/v1/records.json ({len(light)} records)")

    # Per-record (full, with text)
    written = 0
    for d in docs:
        # Avoid creating directory traversal — slugified IDs are filesystem-safe
        # but defend in depth.
        rid = d["id"].replace("/", "_").replace("..", "_")
        (PER_RECORD_DIR / f"{rid}.json").write_text(json.dumps(d, ensure_ascii=False))
        written += 1
    print(f"  wrote /ui/api/v1/records/<id>.json ({written} files)")

    # Incidents
    (API_DIR / "incidents.json").write_text(json.dumps(incidents, indent=2, ensure_ascii=False))

    # Atom feed — records sorted by release_date (newest first, fall back to id)
    enriched = []
    for d in docs:
        if d.get("type") == "IMG" and "__img_p" in d.get("id", ""):
            continue   # Skip synthetic image rows in feed
        rd = parse_release_date(d.get("release_date", ""))
        enriched.append((rd or datetime(1970, 1, 1, tzinfo=timezone.utc), d))
    enriched.sort(key=lambda kv: kv[0], reverse=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8"?>')
    parts.append('<feed xmlns="http://www.w3.org/2005/Atom">')
    parts.append(f"  <title>{xml_escape(SITE_TITLE)}</title>")
    parts.append(f"  <subtitle>{xml_escape(SITE_DESC)}</subtitle>")
    parts.append(f'  <link href="{SITE_URL}/ui/feed.xml" rel="self"/>')
    parts.append(f'  <link href="{SITE_URL}/"/>')
    parts.append(f"  <id>{SITE_URL}/ui/feed.xml</id>")
    parts.append(f"  <updated>{now}</updated>")
    for ts, d in enriched[:200]:
        rid = d["id"]
        title = xml_escape(d.get("title", "(untitled)"))
        url = f"{SITE_URL}/ui/#doc={rid}"
        summary_chunks = []
        if d.get("blurb"):
            summary_chunks.append(d["blurb"][:600])
        if d.get("incident_location"):
            summary_chunks.append(f"Location: {d['incident_location']}")
        if d.get("incident_date"):
            summary_chunks.append(f"Incident date: {d['incident_date']}")
        if d.get("incident_id"):
            summary_chunks.append(f"Incident: {d['incident_id']}")
        summary = xml_escape("\n\n".join(summary_chunks))
        parts.append("  <entry>")
        parts.append(f"    <id>{SITE_URL}/ui/api/v1/records/{xml_escape(rid)}.json</id>")
        parts.append(f"    <title>{title}</title>")
        parts.append(f'    <link href="{xml_escape(url)}"/>')
        parts.append(f"    <updated>{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>")
        parts.append(f"    <author><name>{xml_escape(d.get('agency', 'U.S. Government'))}</name></author>")
        parts.append(f"    <summary>{summary}</summary>")
        parts.append("  </entry>")
    parts.append("</feed>")
    FEED_PATH.write_text("\n".join(parts))
    print(f"  wrote /ui/feed.xml ({min(len(enriched), 200)} entries)")


if __name__ == "__main__":
    main()

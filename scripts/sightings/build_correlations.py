#!/usr/bin/env python3
"""Find civilian sightings near each official record.

For every record with provenance="official_*" that has a parseable
date AND lat/lng, find civilian sightings (provenance != official_*)
within DISTANCE_KM and ±DAYS_WINDOW. Emits raw/sightings/correlations.json
keyed by official record id.

Output shape (one entry per official record that had any matches):
  {
    "wargov:<id>": {
      "official": { …trimmed copy of the official record… },
      "matches": [
        { "id": "nuforc:000123", "distance_km": 23.4, "delta_days": -2,
          "summary": "...", "occurred_at": "...", "shape": "...",
          "location": {...} },
        ...
      ]
    },
    ...
  }

The temporal window widens for low-precision dates: a war.gov record
dated only "year" gets ±365 days; "month" gets ±30; otherwise DAYS_WINDOW.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIGHTINGS = ROOT / "raw" / "sightings" / "sightings.json"
OUT = ROOT / "raw" / "sightings" / "correlations.json"
UI_OUT = ROOT / "ui" / "correlations.json"

DISTANCE_KM = 150.0   # 150km ≈ 1.35° at equator — generous, civilian reports are city-coarse
DAYS_WINDOW = 30      # default temporal window
MAX_PER_RECORD = 50   # cap per official record to keep file size sane


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def parse_date(iso: str) -> dt.date | None:
    if not iso:
        return None
    try:
        return dt.date.fromisoformat(iso[:10])
    except ValueError:
        return None


def window_days(precision: str) -> int:
    return {"year": 365, "decade": 365 * 5, "month": 30}.get(precision, DAYS_WINDOW)


def has_geo(rec: dict) -> bool:
    loc = rec.get("location") or {}
    return "lat" in loc and "lng" in loc


def is_official(rec: dict) -> bool:
    return (rec.get("provenance") or "").startswith("official_")


def main() -> int:
    sightings = json.loads(SIGHTINGS.read_text())
    print(f"[corr] loaded {len(sightings):,} sightings")

    official = [s for s in sightings if is_official(s) and has_geo(s) and parse_date(s.get("occurred_at", ""))]
    civilian = [s for s in sightings if not is_official(s) and has_geo(s) and parse_date(s.get("occurred_at", ""))]
    print(f"[corr] datable+geocoded: {len(official)} official, {len(civilian):,} civilian")

    # Pre-compute civilian fields for the inner loop
    civ_idx = []
    for c in civilian:
        d = parse_date(c["occurred_at"])
        loc = c["location"]
        civ_idx.append((c, d, loc["lat"], loc["lng"]))

    correlations: dict[str, dict] = {}
    for off in official:
        off_date = parse_date(off["occurred_at"])
        off_loc = off["location"]
        win = window_days(off.get("occurred_at_precision") or "day")

        candidates: list[tuple[float, int, dict]] = []
        # Bounding box prefilter: 1° lat ≈ 111km
        deg_lat = DISTANCE_KM / 111.0
        deg_lng = DISTANCE_KM / (111.0 * max(0.1, math.cos(math.radians(off_loc["lat"]))))

        for c, c_date, c_lat, c_lng in civ_idx:
            if abs(c_lat - off_loc["lat"]) > deg_lat:
                continue
            if abs(c_lng - off_loc["lng"]) > deg_lng:
                continue
            delta_days = (c_date - off_date).days
            if abs(delta_days) > win:
                continue
            km = haversine_km(off_loc["lat"], off_loc["lng"], c_lat, c_lng)
            if km > DISTANCE_KM:
                continue
            candidates.append((km, delta_days, c))

        if not candidates:
            continue

        # Closest in space first, then closest in time
        candidates.sort(key=lambda x: (x[0], abs(x[1])))
        matches = []
        for km, delta_days, c in candidates[:MAX_PER_RECORD]:
            matches.append({
                "id": c["id"],
                "source": c["source"],
                "distance_km": round(km, 1),
                "delta_days": delta_days,
                "occurred_at": c.get("occurred_at", ""),
                "title": c.get("title", ""),
                "summary": (c.get("summary") or "")[:200],
                "shape": c.get("shape", ""),
                "location": c.get("location", {}),
            })

        correlations[off["id"]] = {
            "official": {
                "id": off["id"],
                "source": off["source"],
                "title": off.get("title", ""),
                "occurred_at": off.get("occurred_at", ""),
                "location": off.get("location", {}),
                "provenance": off["provenance"],
            },
            "match_count": len(candidates),
            "matches": matches,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(correlations, f, ensure_ascii=False, indent=2)

    # Mirror to ui/ — the static UI fetches it directly. Re-key by the id
    # the UI uses in search-index.json: war.gov drops the "wargov:" prefix
    # entirely; Blue Book uses underscore form ("blue_book:12" → "blue_book_12")
    # because some UI plumbing trips on colons in record ids.
    def _ui_key(cid: str) -> str:
        if cid.startswith("wargov:"):
            return cid.split(":", 1)[1]
        return cid.replace(":", "_", 1)

    ui_correlations = {_ui_key(cid): data for cid, data in correlations.items()}
    UI_OUT.parent.mkdir(parents=True, exist_ok=True)
    with UI_OUT.open("w") as f:
        json.dump(ui_correlations, f, ensure_ascii=False, separators=(",", ":"))

    total_matches = sum(c["match_count"] for c in correlations.values())
    print(
        f"[corr] {len(correlations)}/{len(official)} official records have civilian matches "
        f"(total {total_matches:,} matches) → {OUT.relative_to(ROOT)} + {UI_OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

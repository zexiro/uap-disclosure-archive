"""Tiny file-cached geocoder over Nominatim.

Used to resolve war.gov `incident_location` strings to lat/lng. Caches
to raw/sightings/geocache.json so repeated pipeline runs don't re-hit
Nominatim. Respects the 1-req/sec policy.

Locations like "Moon" or "Low Earth Orbit" obviously can't geocode —
those return None and the caller stores the name without coordinates.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "raw" / "sightings" / "geocache.json"

USER_AGENT = "disclosure-archive/1.0 (https://github.com/lwhmby/disclosure)"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Free-text locations that can't be geocoded — skip without an API call.
SKIPLIST = {
    "n/a", "", "moon", "low earth orbit", "leo", "space", "orbit",
    "indo-pacom", "centcom", "northcom", "eucom", "africom", "southcom",
    "various", "unknown", "classified", "redacted",
}

_cache: dict[str, dict | None] | None = None


def _load_cache() -> dict[str, dict | None]:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        _cache = json.loads(CACHE_PATH.read_text())
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=2, sort_keys=True, ensure_ascii=False))


def geocode(name: str, *, polite_delay: float = 1.1) -> dict | None:
    """Returns {"lat": float, "lng": float, "country": str} or None."""
    if not name:
        return None
    key = name.strip()
    if key.lower() in SKIPLIST:
        return None
    cache = _load_cache()
    if key in cache:
        return cache[key]

    params = {"q": key, "format": "json", "limit": "1", "addressdetails": "1"}
    url = f"{NOMINATIM}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"[geocode] failed for {key!r}: {e}")
        cache[key] = None
        _save_cache()
        time.sleep(polite_delay)
        return None

    result: dict | None
    if data:
        item = data[0]
        addr = item.get("address", {}) or {}
        result = {
            "lat": float(item["lat"]),
            "lng": float(item["lon"]),
            "country": (addr.get("country_code") or "").lower(),
        }
    else:
        result = None
    cache[key] = result
    _save_cache()
    time.sleep(polite_delay)
    return result

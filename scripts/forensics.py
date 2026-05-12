#!/usr/bin/env python3
"""
Build-time image forensics — Wave 2 of IMAGE_FORENSICS_PLAN.md.

For every image under raw/images/, write a metadata sidecar entry to
ui/image_forensics.json. The lightbox loads this lazily and surfaces it
as a "metadata" pane behind the FX toolbar.

Per-image output:
- size_bytes, dimensions, sha256, phash
- jpeg_markers + marker_fingerprint (APP0/APP1/... segment names)
- exif: camera, lens, software, datetime, GPS, orientation, tag_count
- edit_signatures: detected editor / scanner software strings
- celestial: sun + moon altitude/azimuth + moon phase (only when GPS + datetime present)
- notes: human-readable observations ("no camera metadata — likely a scan", etc.)

Pure stdlib + PIL + imagehash (already deps). No new requirements,
no ephemeris files — celestial math uses analytical formulae
(Meeus / NOAA) accurate to ~0.3°, more than enough to answer
"was the sun above the horizon when this was taken".
"""
from __future__ import annotations
import json
import hashlib
import math
import struct
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from PIL import Image, ExifTags
import imagehash

ROOT = Path(__file__).resolve().parent.parent
# We scan both the primary image directory and the per-PDF extracted images
# directory. Sidecars are written one per image to ui/forensics/<name>.json
# so the lightbox only fetches the file it needs on demand — a single
# bundled JSON would be ~12 MB and gate the FX panel on a heavy parse.
IMG_DIRS = [
    ROOT / "raw" / "images",
    ROOT / "raw" / "images_extracted",
]
OUT_DIR = ROOT / "ui" / "forensics"
INDEX_OUT = ROOT / "ui" / "forensics_index.json"

# Editor / scanner software signatures we'll surface as edit hints.
# Matched case-insensitively against EXIF Software / ProcessingSoftware
# and the JPEG comment block.
EDITOR_PATTERNS = [
    "adobe photoshop", "photoshop", "lightroom", "camera raw",
    "affinity photo", "gimp", "pixelmator", "acorn",
    "imagemagick", "graphicsmagick",
    "preview", "macos preview",
    "scanner", "scansoft", "kodak", "epson scan", "canoscan",
    "abbyy", "finereader",
    "save for web", "exported", "exiftool",
]

EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}


# ── JPEG segment walker ───────────────────────────────────────────────
def jpeg_markers(path: Path, max_bytes: int = 256_000) -> list[str]:
    """Return APPn segment identifier strings in order of appearance."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return []
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xDA:  # SOS — image data starts
            break
        if 0xE0 <= marker <= 0xEF:
            size = struct.unpack(">H", data[i + 2 : i + 4])[0]
            payload = data[i + 4 : i + 2 + size]
            head = payload.split(b"\x00", 1)[0].decode("latin1", errors="replace")[:30]
            out.append(f"APP{marker - 0xE0}({head or '?'})")
            i += 2 + size
            continue
        i += 1
    return out


def jpeg_comment(path: Path, max_bytes: int = 256_000) -> str | None:
    """Read the COM (0xFE) segment if present — sometimes carries editor info."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return None
    i = 0
    n = len(data)
    while i < n - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xDA:
            return None
        if marker == 0xFE:
            size = struct.unpack(">H", data[i + 2 : i + 4])[0]
            return data[i + 4 : i + 2 + size].decode("latin1", errors="replace").strip()
        if 0xE0 <= marker <= 0xEF:
            size = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + size
            continue
        i += 1
    return None


# ── EXIF helpers ──────────────────────────────────────────────────────
def _rational_to_float(r) -> float:
    try:
        if isinstance(r, tuple) and len(r) == 2:
            return r[0] / r[1] if r[1] else 0.0
        return float(r)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _dms_to_deg(dms) -> float:
    try:
        d, m, s = (_rational_to_float(x) for x in dms)
        return d + m / 60 + s / 3600
    except Exception:
        return 0.0


def _parse_exif_datetime(s: str) -> datetime | None:
    """EXIF format: 'YYYY:MM:DD HH:MM:SS' (no timezone). Assume UTC."""
    try:
        return datetime.strptime(s.strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def extract_exif(img: Image.Image) -> dict:
    """Pull a normalised EXIF summary out of a PIL image."""
    raw = img._getexif() or {}
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    out: dict = {
        "tag_count": len(tags),
        "camera": None,
        "lens": None,
        "software": None,
        "datetime": None,
        "gps": None,
        "orientation": tags.get("Orientation"),
        "resolution_dpi": None,
    }
    make = (tags.get("Make") or "").strip()
    model = (tags.get("Model") or "").strip()
    if make or model:
        out["camera"] = (f"{make} {model}").strip()
    if tags.get("LensModel"):
        out["lens"] = str(tags["LensModel"]).strip()
    if tags.get("Software"):
        out["software"] = str(tags["Software"]).strip()
    dt = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if dt:
        parsed = _parse_exif_datetime(str(dt))
        if parsed:
            out["datetime"] = parsed.isoformat()
    xres = tags.get("XResolution")
    yres = tags.get("YResolution")
    if xres and yres:
        out["resolution_dpi"] = [round(_rational_to_float(xres), 1), round(_rational_to_float(yres), 1)]
    # GPS
    gps_raw = tags.get("GPSInfo")
    if gps_raw:
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
        lat = gps.get("GPSLatitude")
        lon = gps.get("GPSLongitude")
        if lat and lon:
            lat_deg = _dms_to_deg(lat)
            lon_deg = _dms_to_deg(lon)
            if gps.get("GPSLatitudeRef") in ("S", "s"):
                lat_deg = -lat_deg
            if gps.get("GPSLongitudeRef") in ("W", "w"):
                lon_deg = -lon_deg
            out["gps"] = [round(lat_deg, 6), round(lon_deg, 6)]
    return out


# ── Edit signatures ───────────────────────────────────────────────────
def detect_edit_signatures(exif: dict, comment: str | None) -> list[str]:
    """Match longest patterns first so "Adobe Photoshop" suppresses the
    redundant bare "Photoshop" hit it contains."""
    haystack = " ".join(
        [
            str(exif.get("software") or ""),
            str(exif.get("camera") or ""),
            comment or "",
        ]
    ).lower()
    found: list[str] = []
    for pat in sorted(EDITOR_PATTERNS, key=lambda p: -len(p)):
        if pat not in haystack:
            continue
        # Skip if a longer pattern we already matched contains this one.
        if any(pat in f.lower() for f in found):
            continue
        found.append(pat.title())
    return found


# ── Celestial position (analytical, no ephemeris) ─────────────────────
# Both formulae are simplified Meeus — enough to answer "above or below
# the horizon" and orient a shadow direction. Accuracy ±0.3° for sun,
# ±1° for moon (good enough for civil purposes).
def _julian_day(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def _equatorial_to_horizontal(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float, dt: datetime) -> tuple[float, float]:
    """Convert RA/Dec to altitude/azimuth at observer's location."""
    jd = _julian_day(dt)
    t = (jd - 2451545.0) / 36525
    # Greenwich Mean Sidereal Time (degrees)
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + t * t * (0.000387933 - t / 38710000)
    gmst %= 360
    lst = (gmst + lon_deg) % 360
    ha = (lst - ra_deg) % 360
    if ha > 180:
        ha -= 360
    ha_r = math.radians(ha)
    dec_r = math.radians(dec_deg)
    lat_r = math.radians(lat_deg)
    sin_alt = math.sin(dec_r) * math.sin(lat_r) + math.cos(dec_r) * math.cos(lat_r) * math.cos(ha_r)
    alt = math.degrees(math.asin(max(-1, min(1, sin_alt))))
    cos_az = (math.sin(dec_r) - math.sin(math.radians(alt)) * math.sin(lat_r)) / max(1e-9, math.cos(math.radians(alt)) * math.cos(lat_r))
    az = math.degrees(math.acos(max(-1, min(1, cos_az))))
    if math.sin(ha_r) > 0:
        az = 360 - az
    return alt, az


def sun_position(lat: float, lon: float, dt: datetime) -> dict:
    jd = _julian_day(dt)
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360  # mean longitude
    g = math.radians((357.528 + 0.9856003 * n) % 360)  # mean anomaly
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))  # ecliptic longitude
    eps = math.radians(23.439 - 0.0000004 * n)  # obliquity
    ra = math.degrees(math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))) % 360
    dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
    alt, az = _equatorial_to_horizontal(ra, dec, lat, lon, dt)
    return {"altitude_deg": round(alt, 2), "azimuth_deg": round(az, 2), "above_horizon": alt > -0.833}


def moon_position(lat: float, lon: float, dt: datetime) -> dict:
    """Low-precision Meeus formula. Phase is fraction illuminated (0..1)."""
    jd = _julian_day(dt)
    t = (jd - 2451545.0) / 36525
    # Mean elements (Meeus chapter 47, low-precision)
    L_prime = (218.3164477 + 481267.88123421 * t) % 360
    D = (297.8501921 + 445267.1114034 * t) % 360
    M = (357.5291092 + 35999.0502909 * t) % 360
    M_prime = (134.9633964 + 477198.8675055 * t) % 360
    F = (93.272095 + 483202.0175233 * t) % 360
    # Major periodic terms only (degrees)
    Dr, Mr, Mpr, Fr = (math.radians(x) for x in (D, M, M_prime, F))
    lon_corr = (
        6.289 * math.sin(Mpr)
        - 1.274 * math.sin(Mpr - 2 * Dr)
        + 0.658 * math.sin(2 * Dr)
        - 0.186 * math.sin(Mr)
    )
    lat_corr = 5.128 * math.sin(Fr) + 0.281 * math.sin(Mpr + Fr) - 0.278 * math.sin(Mpr - Fr)
    moon_lon = math.radians((L_prime + lon_corr) % 360)
    moon_lat = math.radians(lat_corr)
    eps = math.radians(23.439 - 0.0000004 * (jd - 2451545.0))
    ra = math.degrees(math.atan2(
        math.sin(moon_lon) * math.cos(eps) - math.tan(moon_lat) * math.sin(eps),
        math.cos(moon_lon),
    )) % 360
    dec = math.degrees(math.asin(
        math.sin(moon_lat) * math.cos(eps) + math.cos(moon_lat) * math.sin(eps) * math.sin(moon_lon)
    ))
    alt, az = _equatorial_to_horizontal(ra, dec, lat, lon, dt)
    # Phase angle (Sun-Moon elongation), then illuminated fraction
    elong = (L_prime - (280.460 + 0.9856474 * (jd - 2451545.0))) % 360
    phase_frac = (1 - math.cos(math.radians(elong))) / 2
    phase_name = _phase_name(elong)
    return {
        "altitude_deg": round(alt, 2),
        "azimuth_deg": round(az, 2),
        "above_horizon": alt > 0,
        "phase": round(phase_frac, 3),
        "phase_name": phase_name,
    }


def _phase_name(elong_deg: float) -> str:
    e = elong_deg % 360
    if e < 22.5: return "new"
    if e < 67.5: return "waxing crescent"
    if e < 112.5: return "first quarter"
    if e < 157.5: return "waxing gibbous"
    if e < 202.5: return "full"
    if e < 247.5: return "waning gibbous"
    if e < 292.5: return "last quarter"
    if e < 337.5: return "waning crescent"
    return "new"


# ── Per-image driver ──────────────────────────────────────────────────
def process_image(path: Path) -> dict | None:
    try:
        size_bytes = path.stat().st_size
        with Image.open(path) as img:
            w, h = img.size
            exif = extract_exif(img)
            ph = str(imagehash.phash(img))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as e:
        print(f"[forensics] {path.name}: {e}", file=sys.stderr)
        return None

    markers = jpeg_markers(path)
    fingerprint = "+".join(sorted({m.split("(", 1)[1].rstrip(")") for m in markers})) or "raw"
    comment = jpeg_comment(path)
    signatures = detect_edit_signatures(exif, comment)

    notes: list[str] = []
    if exif["tag_count"] == 0:
        notes.append("no EXIF metadata — possibly stripped or never recorded")
    elif not exif["camera"] and not exif["software"]:
        notes.append("EXIF present but no camera or software field — likely a scan")
    if signatures:
        notes.append(f"editor/scanner signature: {', '.join(signatures)}")
    if not exif["gps"]:
        notes.append("no GPS — location cannot be verified from metadata")

    celestial = None
    if exif["gps"] and exif["datetime"]:
        try:
            dt = datetime.fromisoformat(exif["datetime"])
            lat, lon = exif["gps"]
            celestial = {
                "sun": sun_position(lat, lon, dt),
                "moon": moon_position(lat, lon, dt),
                "observer": {"lat": lat, "lon": lon, "datetime": exif["datetime"]},
            }
            if not celestial["sun"]["above_horizon"]:
                notes.append("sun below horizon at recorded time/place — taken at night")
            else:
                notes.append(
                    f"sun at altitude {celestial['sun']['altitude_deg']}° azimuth {celestial['sun']['azimuth_deg']}° at recorded time/place"
                )
        except Exception as e:
            print(f"[forensics] celestial calc failed for {path.name}: {e}", file=sys.stderr)

    return {
        "size_bytes": size_bytes,
        "dimensions": [w, h],
        "sha256": sha,
        "phash": ph,
        "jpeg_markers": markers,
        "marker_fingerprint": fingerprint,
        "jpeg_comment": comment,
        "exif": exif,
        "edit_signatures": signatures,
        "celestial": celestial,
        "notes": notes,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for d in IMG_DIRS:
        if not d.exists():
            print(f"[forensics] {d} doesn't exist; skipping")
            continue
        paths.extend(sorted(d.glob("*.jpg")))
        paths.extend(sorted(d.glob("*.png")))

    # Index lists which images have a sidecar — the lightbox checks here
    # before issuing a fetch so missing-data is a zero-network case.
    index: dict[str, dict] = {}
    written = 0
    for p in paths:
        rec = process_image(p)
        if rec is None:
            continue
        sidecar_name = p.name + ".json"
        (OUT_DIR / sidecar_name).write_text(json.dumps(rec, sort_keys=True, separators=(",", ":")))
        index[p.name] = {
            "sha256": rec["sha256"],
            "phash": rec["phash"],
            "has_camera": bool(rec["exif"]["camera"]),
            "has_gps": bool(rec["exif"]["gps"]),
            "has_signatures": bool(rec["edit_signatures"]),
            "has_celestial": rec["celestial"] is not None,
        }
        written += 1
    INDEX_OUT.write_text(json.dumps(index, sort_keys=True))
    print(f"[forensics] wrote {written} sidecars → {OUT_DIR.relative_to(ROOT)}/")
    print(f"[forensics] wrote index → {INDEX_OUT.relative_to(ROOT)} ({len(index)} entries)")

    # Tally on the index.
    have_camera = sum(1 for r in index.values() if r["has_camera"])
    have_gps = sum(1 for r in index.values() if r["has_gps"])
    have_signature = sum(1 for r in index.values() if r["has_signatures"])
    print(f"[forensics]   {have_camera} with camera, {have_gps} with GPS, {have_signature} with editor signatures")

    # Sweep stale sidecars: any file under OUT_DIR not in the index list now.
    expected = {p + ".json" for p in index.keys()}
    for f in OUT_DIR.glob("*.json"):
        if f.name not in expected:
            f.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())

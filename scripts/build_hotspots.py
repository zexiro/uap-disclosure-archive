#!/usr/bin/env python3
"""
build_hotspots.py — Getis-Ord Gi* hotspot analysis on H3 hexbins.

Loads all incidents with valid lat/lng from ui/correlations.json, buckets
them into H3 hexagons at resolution 4 (continent-scale) and resolution 6
(regional), then computes Getis-Ord Gi* statistics separately for:
  - all sources
  - official (war.gov, AARO, FBI, Blue Book)
  - civilian (NUFORC)

Uses PySAL esda.G_Local with libpysal.weights.KNN (k=8).
Outputs ui/hotspots.json.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import h3
except ImportError:
    print("ERROR: pip install 'h3>=4.0'", file=sys.stderr)
    sys.exit(1)

try:
    import libpysal
    from libpysal.weights import KNN
    from esda.getisord import G_Local
except ImportError:
    print("ERROR: pip install esda libpysal", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent.parent
CORRELATIONS_PATH = ROOT / "ui" / "correlations.json"
OUTPUT_PATH = ROOT / "ui" / "hotspots.json"

OFFICIAL_SOURCES = {"wargov", "aaro", "fbi", "blue_book", "bluebook"}
RESOLUTIONS = [4, 6]


def load_points():
    """Return list of dicts with lat, lng, source, is_official."""
    data = json.loads(CORRELATIONS_PATH.read_text())
    points = []

    for key, entry in data.items():
        # Official record
        rec = entry.get("official", {})
        loc = rec.get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")
        source = rec.get("source", "")
        if lat is not None and lng is not None:
            points.append({
                "lat": float(lat),
                "lng": float(lng),
                "source": source,
                "is_official": source.lower().replace("-", "_") in OFFICIAL_SOURCES,
            })

        # NUFORC civilian matches
        for match in entry.get("matches", []):
            ml = match.get("location", {})
            mlat, mlng = ml.get("lat"), ml.get("lng")
            msource = match.get("source", "nuforc")
            if mlat is not None and mlng is not None:
                points.append({
                    "lat": float(mlat),
                    "lng": float(mlng),
                    "source": msource,
                    "is_official": msource.lower().replace("-", "_") in OFFICIAL_SOURCES,
                })

    return points


def bucket_points(points, resolution):
    """Map each point to an H3 hex cell, return dict hex_id → list of points."""
    buckets = defaultdict(list)
    for pt in points:
        cell = h3.latlng_to_cell(pt["lat"], pt["lng"], resolution)
        buckets[cell].append(pt)
    return buckets


def gi_star(hex_ids, counts, k=8):
    """
    Compute Getis-Ord Gi* for the given hex ids and counts array.

    Returns (z_scores, p_values) arrays aligned with hex_ids.
    Falls back to zeros if fewer than k+1 hexes.
    """
    n = len(hex_ids)
    if n < k + 1:
        return np.zeros(n), np.ones(n)

    # Build coordinate array from H3 centroids
    coords = np.array([h3.cell_to_latlng(h) for h in hex_ids], dtype=float)
    # coords rows: [lat, lng]

    try:
        w = KNN.from_array(coords, k=min(k, n - 1))
        w.transform = "R"  # row-standardise
        y = np.array(counts, dtype=float)
        g = G_Local(y, w, transform="R", star=True, permutations=0)
        z = np.array(g.Zs)
        # Two-tailed p-values from standard normal
        from scipy import stats as scipy_stats
        p = 2 * (1 - scipy_stats.norm.cdf(np.abs(z)))
        return z, p
    except Exception as e:
        print(f"  [warn] Gi* failed ({e}), returning zeros", file=sys.stderr)
        return np.zeros(n), np.ones(n)


def build_layer(hex_buckets, filter_fn=None):
    """
    Build a hotspot layer.

    hex_buckets: dict hex_id → list of all points in that hex.
    filter_fn:   callable(point) → bool selecting which points count.
    Returns list of dicts with hex, z, p, count, centroid.
    """
    if filter_fn is None:
        filter_fn = lambda _: True

    hex_ids = []
    counts = []
    for hex_id, pts in hex_buckets.items():
        c = sum(1 for p in pts if filter_fn(p))
        hex_ids.append(hex_id)
        counts.append(c)

    if not hex_ids:
        return []

    z_arr, p_arr = gi_star(hex_ids, counts)

    results = []
    for i, hex_id in enumerate(hex_ids):
        lat, lng = h3.cell_to_latlng(hex_id)
        results.append({
            "hex": hex_id,
            "z": round(float(z_arr[i]), 4),
            "p": round(float(p_arr[i]), 6),
            "count": int(counts[i]),
            "centroid": [round(lat, 5), round(lng, 5)],
        })

    return results


def print_summary(label, layer_data):
    """Print count of significant hexes and top-3 hottest."""
    sig = [r for r in layer_data if r["p"] < 0.05 and r["count"] > 0]
    total = len(layer_data)
    print(f"  {label}: {len(sig)} sig / {total} total hexes")
    top3 = sorted(layer_data, key=lambda r: r["z"], reverse=True)[:3]
    for r in top3:
        lat, lng = r["centroid"]
        print(f"    hex={r['hex']}  z={r['z']:.2f}  p={r['p']:.4f}  "
              f"n={r['count']}  @ ({lat:.2f}, {lng:.2f})")


def main():
    print("[hotspots] loading incidents …")
    points = load_points()
    print(f"[hotspots] {len(points)} points with valid lat/lng")

    output = {}

    for res in RESOLUTIONS:
        print(f"\n[hotspots] resolution {res} …")
        buckets = bucket_points(points, res)
        print(f"  {len(buckets)} hexes occupied")

        all_layer      = build_layer(buckets)
        official_layer = build_layer(buckets, filter_fn=lambda p: p["is_official"])
        civilian_layer = build_layer(buckets, filter_fn=lambda p: not p["is_official"])

        res_key = f"h3_resolution_{res}"
        output[res_key] = {
            "all":      all_layer,
            "official": official_layer,
            "civilian": civilian_layer,
        }

        print(f"\n  === Resolution {res} smoke test ===")
        print_summary("all",      all_layer)
        print_summary("official", official_layer)
        print_summary("civilian", civilian_layer)

    OUTPUT_PATH.write_text(json.dumps(output, separators=(",", ":")))
    print(f"\n[hotspots] wrote {OUTPUT_PATH}")

    # Print final summary counts for deliverable
    print("\n=== Significant hotspot counts (p < 0.05) ===")
    for res in RESOLUTIONS:
        rk = f"h3_resolution_{res}"
        for layer in ["all", "official", "civilian"]:
            data = output[rk][layer]
            sig = sum(1 for r in data if r["p"] < 0.05 and r["count"] > 0)
            print(f"  res{res} / {layer:8s}: {sig}")

    print("\n=== Top-3 hottest hexes (all sources, resolution 4) ===")
    top3 = sorted(output["h3_resolution_4"]["all"], key=lambda r: r["z"], reverse=True)[:3]
    for r in top3:
        lat, lng = r["centroid"]
        print(f"  {r['hex']}  z={r['z']:.2f}  n={r['count']}  @ ({lat:.2f}, {lng:.2f})")


if __name__ == "__main__":
    main()

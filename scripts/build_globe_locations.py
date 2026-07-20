"""Build ui/geo/locations.json — incident_location string → [lat, lng].

Resolves every distinct `incident_location` in records-lite.json via the
file-cached Nominatim geocoder (scripts/sightings/geocode.py), plus manual
overrides for regions Nominatim can't place (oceans, US regions, defunct
states). Strings that genuinely have no earthly location (Moon, orbit,
combatant commands) are omitted — the globe skips them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.geocode import geocode  # noqa: E402

OUT = ROOT / "ui" / "geo" / "locations.json"

# Regions / defunct states / seas that Nominatim fails or misplaces.
MANUAL = {
    "Atlantic Ocean": [30.0, -45.0],
    "North Atlantic Ocean": [45.0, -30.0],
    "South China Sea": [12.0, 114.0],
    "Yellow Sea": [35.0, 123.0],
    "Gulf of America": [27.0, -90.0],
    "Eastern United States": [37.0, -80.0],
    "Midwestern United States": [41.5, -93.0],
    "Northeastern United States": [43.0, -72.0],
    "Southeastern United States": [33.0, -84.0],
    "Southern United States": [32.0, -90.0],
    "Western United States": [40.0, -113.0],
    "Westen United States": [40.0, -113.0],
    "New Mexico": [34.5, -106.0],
    "Texas": [31.0, -100.0],
    "Virginia": [37.5, -78.7],
    "USSR": [55.75, 37.62],
    "Ladakh, Nepal | Sikkim, India | Bhutam": [34.0, 77.6],
    # Raw lat/lng strings embedded in the location field.
    "12-17'N 155-35' W (Pacific)": [12.283, -155.583],
    "14'-17\" N 69'57\" E": [14.283, 69.95],
    "19' N 172\" E (Pacific)": [19.0, 172.0],
    "34'55\" N 164'05\" E (Pacific)": [34.917, 164.083],
    "35'-50' N 125'40' W (Pacific)": [35.833, -125.667],
    "37' 25' N 132' 25' E (Sea of Japan)": [37.417, 132.417],
    "40'-00\" N 175'54\" W (Pacific)": [40.0, -175.9],
    "41'N-35'W (Atlantic)": [41.0, -35.0],
    # Relative-to-place descriptions → the anchor place.
    "15 miles south of Houghton": [46.9, -88.57],
    "20 miles south of Quartsite, Arizona": [33.37, -114.23],
    "200 miles east of Dover (Atlantic)": [39.0, -71.5],
    "60 miles east of St. Louis, Missouri": [38.63, -89.1],
    "Near Owensboro, Kentucky": [37.77, -87.11],
    "North of Goodland, Kansas": [39.6, -101.71],
    "South of Fort Worth, Texas": [32.4, -97.32],
    "South of Kyushu, Japan": [30.5, 131.0],
    "WSW of Biloxi, Mississippi": [30.39, -89.3],
    # Misspellings / historic names in the source records.
    "Air radar, Bermuda": [32.3, -64.75],
    "Andrews AFB, Washington, D.C.": [38.81, -76.87],
    "Annandle, Virginia": [38.83, -77.2],
    "Bentor Harbor, Michigan": [42.12, -86.45],
    "Boston-Provincetown, Massachusetts": [42.0, -70.5],
    "Bunker Hill AFB , Indiana": [40.65, -86.15],
    "Cheffy Creek, New York": [42.9, -74.9],
    "Chicago, lllinois": [41.88, -87.63],
    "Clasgow and Opheim, Montana": [48.2, -106.64],
    "Clinton, Iowa--Littleton, Illinois": [41.5, -90.5],
    "Cold Bay AFS, Alaska": [55.2, -162.72],
    "Copemiah, Michigan": [44.48, -85.92],
    "Cortex, Florida": [27.47, -82.69],
    "Denver, California [sic]": [39.74, -104.99],
    "Elberton, Alabama": [30.41, -87.6],
    "Erding Air Depot, Germany": [48.3, 11.91],
    "Fairfield-Suisan AFB, California": [38.26, -121.93],
    "Forth Smith, Arkansas": [35.39, -94.4],
    "Galesburg, Moline, Illinois": [41.3, -90.5],
    "Goose AFB, Labrador": [53.32, -60.42],
    "Haneda AFB, Japan": [35.55, 139.78],
    "Japan, Korea Area": [36.0, 129.0],
    "Kadena APB, Okinawa": [26.36, 127.77],
    "Killeen Base, Camp Hood, Texas": [31.13, -97.78],
    "Ladd AFB, Alaska": [64.83, -147.61],
    "Lagarfiot River, Iceland": [65.17, -14.5],
    "Lake Charles AFB, Louisiana": [30.21, -93.14],
    "Lake Kishkonoug, Wisconsin": [42.86, -88.94],
    "Langley APB, Virginia": [37.08, -76.36],
    "Larson AFB, Washington": [47.2, -119.32],
    "Leveland, Texas": [33.59, -102.38],
    "Lock Raven Dam, Maryland": [39.43, -76.54],
    "Lockbourne AFB, Ohio": [39.81, -82.93],
    "Marrakech, Morrocco": [31.63, -7.99],
    "Marrowhode Lake, Tennessee": [36.32, -86.78],
    "Merced, Calffornia": [37.3, -120.48],
    "Merrick, Long Island, New York": [40.66, -73.55],
    "Miramar NAS, California": [32.87, -117.14],
    "Misawa AFB, Japan": [40.68, 141.37],
    "Miyako Jima Air Station, Japan": [24.78, 125.28],
    "Muroc AFB, California": [34.91, -117.88],
    "Nagoya, Honshu, Japan": [35.17, 136.9],
    "Nanyika, Kenya, Africa": [0.02, 37.07],
    "Neffesville, Pennsylvania": [40.1, -76.3],
    "Nouasseur, French Morocco": [33.37, -7.59],
    "Oldtown, Florida": [29.6, -82.98],
    "Olmsted AFB, Pennsylvania": [40.19, -76.76],
    "Onawa, lowa": [42.03, -96.1],
    "Operation Mainbrace": [56.0, 3.0],
    "Point Muga, California": [34.12, -119.12],
    "Pope AFB, North Carolina": [35.17, -79.01],
    "Queen Annes City, Maryland": [39.0, -76.0],
    "Rabat, French Morocco": [34.02, -6.84],
    "Ramstein AFB, Germany": [49.44, 7.6],
    "Redstone Arsenal, Georgia": [34.68, -86.65],
    "Richards, Gebaur AB, Missouri": [38.84, -94.56],
    "Rodio, New Mexico": [31.83, -109.03],
    "Scotch Plain, New Jersey": [40.65, -74.4],
    "Selfridge AFB, Michigan": [42.61, -82.83],
    "Simiutak, Greenland": [60.68, -46.57],
    "South Brookville, Maine": [44.35, -68.75],
    "St. Calen, Switzerland": [47.42, 9.37],
    "Suffolk County AFB, New York": [40.84, -72.65],
    "Telephone Ridge, Oregon": [44.0, -120.5],
    "Texahoma, Oklahoma": [36.5, -101.78],
    "Washington, D.C.; Andrews AFB": [38.81, -76.87],
    "Wheelus AFB, Tripoli": [32.9, 13.19],
    "Witchita, Kansas": [37.69, -97.34],
    "Yuma Test Station, Arizona": [32.86, -114.4],
}

# No coordinates by nature — never plot.
UNPLOTTABLE = {
    "Moon", "Low Earth Orbit", "Low-Earth Orbit", "Cislunar Space",
    "N/A", "Various",
}


def main() -> None:
    docs = json.loads((ROOT / "ui" / "records-lite.json").read_text())
    locs = sorted({
        (d.get("incident_location") or "").strip()
        for d in docs
        if (d.get("incident_location") or "").strip()
    })

    out: dict[str, list[float]] = {}
    skipped: list[str] = []
    for loc in locs:
        if loc in UNPLOTTABLE:
            skipped.append(loc)
            continue
        if loc in MANUAL:
            out[loc] = MANUAL[loc]
            continue
        hit = geocode(loc)
        if hit:
            out[loc] = [round(hit["lat"], 4), round(hit["lng"], 4)]
        else:
            skipped.append(loc)

    OUT.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True))
    print(f"wrote {OUT} — {len(out)}/{len(locs)} locations resolved")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Splink probabilistic record linkage across war.gov + NUFORC + Blue Book + FBI.

Cross-source only: we trust per-source ordering, so we never deduplicate
within the same source.  High-precision bias: false-positive cost of linking
a Pentagon document to a NUFORC civilian report is high; we tune thresholds
conservatively.

Outputs:
  ui/dedup_clusters.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SEARCH_INDEX = ROOT / "ui" / "search-index.json"
GEOCACHE = ROOT / "raw" / "sightings" / "geocache.json"
BLUE_BOOK_SIGHTINGS = ROOT / "raw" / "sightings" / "blue_book" / "sightings.json"
OUT = ROOT / "ui" / "dedup_clusters.json"

# ── Threshold ────────────────────────────────────────────────────────────────
# Match threshold.  Conservative enough to avoid false positives but tuned
# to the reality of this corpus (many records have no location, so geo is
# often a neutral signal; date + source-specific title cues carry more weight).
#
# Bands (calibrated on manual inspection):
#   0.75+ : very likely same incident (date within 1 day, same state/region)
#   0.65–0.75 : probable same incident (date within 7 days, vague location)
#   <0.65 : coin-toss, skip
MATCH_THRESHOLD = 0.65

# ── Date parsing ─────────────────────────────────────────────────────────────
MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

CURRENT_YEAR_2 = dt.date.today().year % 100


def parse_date(raw: str) -> str | None:
    """Return ISO YYYY-MM-DD string or None."""
    if not raw or raw in ("N/A", ""):
        return None
    raw = raw.strip()
    # ISO already
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # M/D/YY or M/D/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", raw)
    if m:
        mo, da, yr = (int(x) for x in m.groups())
        if yr < 100:
            yr = 1900 + yr if yr > CURRENT_YEAR_2 else 2000 + yr
        try:
            return dt.date(yr, mo, da).isoformat()
        except ValueError:
            return None
    # "June 24, 1947" etc.
    m2 = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", raw)
    if m2:
        mo_name, da, yr = m2.group(1).lower(), int(m2.group(2)), int(m2.group(3))
        if mo_name in MONTH_NAMES:
            try:
                return dt.date(yr, MONTH_NAMES[mo_name], da).isoformat()
            except ValueError:
                return None
    # "June 1947" — month precision, use mid-month
    m3 = re.match(r"^([A-Za-z]+)\s+(\d{4})$", raw)
    if m3:
        mo_name, yr = m3.group(1).lower(), int(m3.group(2))
        if mo_name in MONTH_NAMES:
            return f"{yr:04d}-{MONTH_NAMES[mo_name]:02d}-15"
    return None


# ── Geocache ──────────────────────────────────────────────────────────────────

def load_geocache() -> dict[str, dict | None]:
    if GEOCACHE.exists():
        return json.loads(GEOCACHE.read_text())
    return {}


def geocode_from_cache(location: str, cache: dict) -> tuple[float | None, float | None]:
    """Return (lat, lng) or (None, None)."""
    if not location:
        return None, None
    # Exact match
    hit = cache.get(location)
    if hit and isinstance(hit, dict):
        return hit.get("lat"), hit.get("lng")
    # Case-insensitive prefix match
    loc_lower = location.lower()
    for k, v in cache.items():
        if k.lower() == loc_lower and v and isinstance(v, dict):
            return v.get("lat"), v.get("lng")
    return None, None


# ── Record loading ─────────────────────────────────────────────────────────────

def source_for_agency(agency: str) -> str:
    mapping = {
        "USAF Project Blue Book": "blue_book",
        "FBI": "fbi",
        "Department of War": "wargov",
        "Department of State": "wargov",
        "NASA": "nasa",
    }
    return mapping.get(agency, "other")


def load_records(geocache: dict) -> list[dict]:
    """
    Load and normalise records from search-index.json plus blue_book sightings
    (which have lat/lng from the build_correlations pipeline).

    Returns a list of dicts with fields:
      record_id, source, incident_date, incident_location, title, lat, lng
    """
    idx = json.loads(SEARCH_INDEX.read_text())
    records_raw = idx if isinstance(idx, list) else idx.get("records", idx)

    # Build lat/lng supplement from blue_book sightings (already geocoded)
    bb_geo: dict[str, tuple[float, float]] = {}
    if BLUE_BOOK_SIGHTINGS.exists():
        bb_sightings = json.loads(BLUE_BOOK_SIGHTINGS.read_text())
        for s in bb_sightings:
            loc = s.get("location") or {}
            if "lat" in loc and "lng" in loc:
                # sighting id is "blue_book:12" → map to search index "blue_book_12"
                sid = s.get("id", "")
                if sid.startswith("blue_book:"):
                    idx_id = "blue_book_" + sid.split(":", 1)[1]
                    bb_geo[idx_id] = (loc["lat"], loc["lng"])

    out: list[dict] = []
    for r in records_raw:
        rec_id = r.get("id", "")
        agency = r.get("agency", "")
        source = source_for_agency(agency)

        raw_date = r.get("incident_date", "") or ""
        raw_loc = r.get("incident_location", "") or ""

        iso_date = parse_date(raw_date)
        if not iso_date:
            continue  # can't link without a date

        # Geocode: prefer bb_geo supplement, then geocache
        lat, lng = None, None
        if rec_id in bb_geo:
            lat, lng = bb_geo[rec_id]
        elif raw_loc and raw_loc != "N/A":
            lat, lng = geocode_from_cache(raw_loc, geocache)

        title = r.get("title", "") or rec_id

        out.append({
            "record_id": rec_id,
            "source": source,
            "incident_date": iso_date,
            "incident_location": raw_loc if raw_loc != "N/A" else "",
            "title": title[:200],
            "lat": lat,
            "lng": lng,
        })

    return out


# ── Haversine ────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# ── Jaro-Winkler (pure Python) ───────────────────────────────────────────────

def jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    match_dist = max(l1, l2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * l1
    s2_matches = [False] * l2
    matches = 0
    transpositions = 0
    for i in range(l1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, l2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if not matches:
        return 0.0
    k = 0
    for i in range(l1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    j = jaro(s1, s2)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return j + prefix * p * (1 - j)


def title_sim(t1: str, t2: str) -> float:
    """Token-level Jaro-Winkler similarity between two titles."""
    stop = {"the", "a", "an", "of", "in", "at", "on", "and", "or", "case",
            "usaf", "blue", "book", "ufo", "uap", "report", "sighting",
            "over", "near", "incident"}
    tokens1 = [w for w in re.findall(r"\w+", t1.lower()) if w not in stop and len(w) > 2]
    tokens2 = [w for w in re.findall(r"\w+", t2.lower()) if w not in stop and len(w) > 2]
    if not tokens1 or not tokens2:
        return 0.0
    # Average max JW for each token in t1 against all tokens in t2
    scores = []
    for tok in tokens1:
        best = max((jaro_winkler(tok, t) for t in tokens2), default=0.0)
        scores.append(best)
    return sum(scores) / len(scores)


# ── Manual Splink-style linkage (no EM training needed at this scale) ─────────
#
# We use Splink for its Bayesian scoring and clustering, but because our corpus
# is small (~700 linkable records) and we have clear domain intuition for the
# Bayes factors, we supply deterministic priors directly rather than running
# EM training (which needs thousands of pairs and would overfit here).
#
# Strategy:
#   1. Enumerate candidate cross-source pairs via blocking rules
#   2. Score each pair with a hand-tuned weighted sum
#   3. Apply match threshold
#   4. Connected-components clustering over surviving pairs


def date_score(d1: str, d2: str) -> float:
    """0.0–1.0 date proximity score."""
    try:
        dt1 = dt.date.fromisoformat(d1)
        dt2 = dt.date.fromisoformat(d2)
        delta = abs((dt2 - dt1).days)
        if delta == 0:
            return 1.0
        if delta <= 7:
            return 0.9
        if delta <= 30:
            return 0.7
        if delta <= 90:
            return 0.4
        if delta <= 365:
            return 0.1
        return 0.0
    except ValueError:
        return 0.0


def geo_score(lat1, lng1, lat2, lng2) -> float:
    """0.0–1.0 geographic proximity score.  Returns None when both sides
    lack coordinates (caller treats as missing, not zero)."""
    if None in (lat1, lng1, lat2, lng2):
        return None  # type: ignore[return-value]
    dist = haversine_km(float(lat1), float(lng1), float(lat2), float(lng2))
    if dist <= 10:
        return 1.0
    if dist <= 50:
        return 0.9
    if dist <= 150:
        return 0.6
    if dist <= 500:
        return 0.2
    return 0.0


def location_string_score(loc1: str, loc2: str) -> float | None:
    """Fuzzy string match on location names.  Returns None when both missing."""
    if not loc1 and not loc2:
        return None
    if not loc1 or not loc2:
        return 0.5  # one side unknown → weak match
    # Normalize: strip AFB/AFS/AAF suffixes for better matching
    def norm(s):
        s = re.sub(r"\b(AFB|AFS|AAF|AAB|NAS|RAF|MCAS|Army Air Field|Air Force Base)\b", "", s, flags=re.I)
        return re.sub(r"[,\s]+", " ", s.lower()).strip()
    l1, l2 = norm(loc1), norm(loc2)
    if l1 == l2:
        return 1.0
    # State-level match: last token often = state
    tokens1 = l1.split()
    tokens2 = l2.split()
    if tokens1 and tokens2 and tokens1[-1] == tokens2[-1]:
        # Same state → partial credit, but use full JW for city match
        return max(0.7, jaro_winkler(l1, l2))
    return jaro_winkler(l1, l2)


def pair_match_probability(r1: dict, r2: dict) -> float:
    """
    Weighted combination of signals → match probability in [0, 1].

    High-precision design:
    - Date is the dominant signal (UAP incidents are clustered by flap periods)
    - Geo is combined from lat/lng (precise) and string (fallback)
    - When geo is completely unavailable for both sides, date+title carry more
    - Title similarity is a weak tiebreaker (Blue Book titles mention city/state)

    Scoring philosophy:
      exact_date + same_region  → ~0.85  (very likely link)
      exact_date + no_location  → ~0.65  (possible, warrants flagging)
      7-day window + same_state → ~0.75  (probable)
      30-day + no_location      → ~0.50  (below threshold, skip)
    """
    if r1["source"] == r2["source"]:
        return 0.0

    d = date_score(r1["incident_date"], r2["incident_date"])
    if d == 0.0:
        return 0.0

    g_latlng = geo_score(r1["lat"], r1["lng"], r2["lat"], r2["lng"])
    g_str = location_string_score(r1["incident_location"], r2["incident_location"])

    # Combine geo signals
    if g_latlng is not None and g_str is not None:
        # Both available: lat/lng wins
        g = 0.65 * g_latlng + 0.35 * g_str
        geo_available = True
    elif g_latlng is not None:
        g = g_latlng
        geo_available = True
    elif g_str is not None:
        g = g_str
        geo_available = True
    else:
        # No location data on either side — neutral
        g = 0.5
        geo_available = False

    t = title_sim(r1["title"], r2["title"])

    if geo_available:
        # Standard weighting: date 50%, geo 35%, title 15%
        score = 0.50 * d + 0.35 * g + 0.15 * t
    else:
        # No geo — date 65%, title 35% (can't rely on location)
        score = 0.65 * d + 0.35 * t

    return round(score, 4)


# ── Blocking rules ─────────────────────────────────────────────────────────────
#
# To avoid O(n²) comparison of all ~700 records, we block on year.
# Within each year bucket we compare all cross-source pairs.

def build_year_buckets(records: list[dict]) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {}
    for r in records:
        try:
            yr = int(r["incident_date"][:4])
        except (ValueError, TypeError, IndexError):
            continue
        buckets.setdefault(yr, []).append(r)
    return buckets


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("[dedup] Loading geocache…")
    geocache = load_geocache()
    print(f"[dedup]   {len(geocache):,} geocache entries ({sum(1 for v in geocache.values() if v):,} non-null)")

    print("[dedup] Loading + normalising records…")
    records = load_records(geocache)
    print(f"[dedup]   {len(records):,} linkable records (have parseable date)")

    source_counts: dict[str, int] = {}
    for r in records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    print(f"[dedup]   By source: {source_counts}")

    print(f"[dedup] Building year buckets + scoring cross-source pairs…")
    buckets = build_year_buckets(records)

    # Splink would score pairs; we compute manually and store as edge list
    uf = UnionFind()
    # Track per-pair scores for cluster member scores
    edge_scores: dict[tuple[str, str], float] = {}

    total_pairs = 0
    linked_pairs = 0

    for yr, bucket in sorted(buckets.items()):
        if len(bucket) < 2:
            continue
        # Compare all cross-source pairs in this year bucket
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                r1, r2 = bucket[i], bucket[j]
                if r1["source"] == r2["source"]:
                    continue
                total_pairs += 1
                prob = pair_match_probability(r1, r2)
                if prob >= MATCH_THRESHOLD:
                    linked_pairs += 1
                    uf.union(r1["record_id"], r2["record_id"])
                    key = (r1["record_id"], r2["record_id"])
                    edge_scores[key] = prob

    print(f"[dedup]   Evaluated {total_pairs:,} cross-source candidate pairs")
    print(f"[dedup]   {linked_pairs:,} pairs above threshold {MATCH_THRESHOLD}")

    # Build clusters from union-find
    # Only include records that are part of a multi-member cluster
    record_by_id = {r["record_id"]: r for r in records}

    cluster_members: dict[str, list[str]] = {}
    for r in records:
        root = uf.find(r["record_id"])
        cluster_members.setdefault(root, []).append(r["record_id"])

    # Filter to clusters with ≥2 members AND cross-source
    multi = {root: members for root, members in cluster_members.items() if len(members) >= 2}

    cross_source_clusters = {}
    for root, members in multi.items():
        member_sources = {record_by_id[m]["source"] for m in members if m in record_by_id}
        if len(member_sources) >= 2:
            cross_source_clusters[root] = members

    print(f"[dedup]   Total clusters (≥2 members): {len(multi):,}")
    print(f"[dedup]   Cross-source clusters: {len(cross_source_clusters):,}")

    # Build output JSON
    def consensus_date(members: list[str]) -> str:
        dates = sorted(
            record_by_id[m]["incident_date"]
            for m in members
            if m in record_by_id and record_by_id[m]["incident_date"]
        )
        return dates[0] if dates else ""

    def consensus_location(members: list[str]) -> str:
        locs = [
            record_by_id[m]["incident_location"]
            for m in members
            if m in record_by_id and record_by_id[m]["incident_location"]
        ]
        # Return most-common non-empty
        if not locs:
            return ""
        counts: dict[str, int] = {}
        for l in locs:
            counts[l] = counts.get(l, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def member_score(rec_id: str, root: str, members: list[str]) -> float:
        # Score = max edge probability this member is part of
        scores = []
        for m in members:
            if m == rec_id:
                continue
            key1 = (rec_id, m)
            key2 = (m, rec_id)
            scores.append(edge_scores.get(key1, edge_scores.get(key2, 0.0)))
        return round(max(scores, default=0.0), 4)

    clusters_list = []
    by_record: dict[str, str] = {}
    cluster_idx = 1

    for root, members in sorted(cross_source_clusters.items()):
        cid = f"c_{cluster_idx:04d}"
        cluster_idx += 1
        member_entries = []
        for m in sorted(members):
            if m not in record_by_id:
                continue
            r = record_by_id[m]
            score = member_score(m, root, members)
            member_entries.append({
                "record_id": m,
                "source": r["source"],
                "match_score": score if score > 0 else round(MATCH_THRESHOLD, 4),
            })
            by_record[m] = cid
        if len(member_entries) < 2:
            continue
        clusters_list.append({
            "cluster_id": cid,
            "members": member_entries,
            "consensus_date": consensus_date(members),
            "consensus_location": consensus_location(members),
        })

    output = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "total_records_evaluated": len(records),
        "match_threshold": MATCH_THRESHOLD,
        "clusters": clusters_list,
        "by_record": by_record,
    }

    OUT.write_text(json.dumps(output, indent=2))
    print(f"[dedup] Wrote {len(clusters_list):,} cross-source clusters → {OUT}")

    # ── Smoke test output ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Total cross-source clusters: {len(clusters_list)}")
    print(f"Records in clusters: {len(by_record)}")
    print()
    print("Top 3 cross-source clusters:")
    for c in clusters_list[:3]:
        print(f"  {c['cluster_id']}  date={c['consensus_date']}  loc={c['consensus_location'][:40]}")
        for m in c["members"]:
            print(f"    [{m['source']}] {m['record_id'][:60]}  score={m['match_score']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

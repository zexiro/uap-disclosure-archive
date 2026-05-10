#!/usr/bin/env python3
"""Augment ui/search-index.json with feature fields, then emit incidents.json
and graph-layout.json consumed by ui/index.html.

Run after build_search_index.py — it overwrites ui/search-index.json in place
with extra per-record fields:

  craft_shape    str     keyword-detected (best-effort) shape: tic-tac, disc,
                         triangle, sphere, cigar, oval, cylinder, chevron,
                         light. "" if no confident match.
  incident_id    str     keyword-detected reference to a known incident
                         (see INCIDENTS below). "" if no match.
  tranche_id     str     version of the source release this record came from.
  last_verified  str     ISO date when build was run (proxy for "source link
                         was reachable as of").

Outputs (alongside the existing ui/search-index.json):

  ui/incidents.json     curated incident definitions
  ui/graph-layout.json  pre-computed 2D coordinates for the full corpus graph
"""
import json
import re
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
INCIDENTS_PATH = ROOT / "ui" / "incidents.json"
LAYOUT_PATH = ROOT / "ui" / "graph-layout.json"
TODAY_INDEX_PATH = ROOT / "ui" / "today_index.json"

TRANCHE_ID = "release-1-2026-05-08"
LAST_VERIFIED = date.today().isoformat()


# ─── Craft shape patterns (first match wins) ──────────────────────────
SHAPE_PATTERNS = [
    ("tic-tac",  re.compile(r"\btic[\s-]?tacs?\b", re.I)),
    ("triangle", re.compile(r"\btriangular|triangle(?:[\s-]?shaped)?\b", re.I)),
    ("disc",     re.compile(r"\b(disc|disk|saucer|flying\s+saucer|disc[\s-]?shaped|disk[\s-]?shaped)\b", re.I)),
    ("sphere",   re.compile(r"\b(spheres?|spherical|orbs?|sphere[\s-]?shaped)\b", re.I)),
    ("cigar",    re.compile(r"\bcigar(?:[\s-]?shaped)?\b", re.I)),
    ("oval",     re.compile(r"\boval(?:[\s-]?shaped)?|elliptical\b", re.I)),
    ("cylinder", re.compile(r"\bcylind(?:rical|er)\b", re.I)),
    ("chevron",  re.compile(r"\bchevron(?:[\s-]?shaped)?|boomerang\b", re.I)),
    ("light",    re.compile(r"\b(bright\s+lights?|glowing\s+objects?|illuminated)\b", re.I)),
]


# ─── Curated incidents ────────────────────────────────────────────────
# id → metadata + match patterns (regex against blurb + first 16KB of text).
# Patterns are conservative: a record is tagged with the FIRST incident whose
# pattern fires. Order matters — most specific first.
INCIDENTS = [
    {
        "id": "tic-tac-2004",
        "name": "Tic Tac (USS Nimitz, 2004)",
        "date": "2004-11-14",
        "location": "Off the coast of San Diego, CA",
        "summary": "Multi-day UAP encounter by the USS Nimitz Carrier Strike Group. Cmdr. David Fravor's F/A-18F engaged a tic-tac-shaped object that displayed extreme acceleration and no visible propulsion.",
        "patterns": [r"\btic[\s-]?tac\b", r"\bU?SS\s+Nimitz\b", r"\bFravor\b", r"\bPrincipi\b"],
    },
    {
        "id": "gimbal-2015",
        "name": "Gimbal (USS Theodore Roosevelt, 2015)",
        "date": "2015-01-21",
        "location": "Off the coast of Jacksonville, FL",
        "summary": "FLIR ATFLIR pod video from a Super Hornet showing an apparently rotating, hovering object against the wind.",
        "patterns": [r"\bgimbal\b", r"\bATFLIR\b"],
    },
    {
        "id": "go-fast-2015",
        "name": "Go Fast (USS Theodore Roosevelt, 2015)",
        "date": "2015-01-21",
        "location": "Off the East Coast of the U.S.",
        "summary": "FLIR video of a small object speeding low over the ocean as tracked by an F/A-18.",
        "patterns": [r"\bgo[\s-]?fast\b"],
    },
    {
        "id": "flir1-2004",
        "name": "FLIR1 (USS Nimitz, 2004)",
        "date": "2004-11-14",
        "location": "Off the coast of San Diego, CA",
        "summary": "Black-and-white infrared track from VFA-41 — same encounter as the Tic Tac sightings.",
        "patterns": [r"\bFLIR\s*1\b", r"\bFLIR-1\b"],
    },
    {
        "id": "aguadilla-2013",
        "name": "Aguadilla (CBP, 2013)",
        "date": "2013-04-25",
        "location": "Aguadilla, Puerto Rico",
        "summary": "Customs and Border Protection thermal video of a low-altitude object that appeared to enter the ocean and split.",
        "patterns": [r"\bAguadilla\b", r"\bRafael\s+Hernandez\b"],
    },
    {
        "id": "eastern-seaboard-2014-2019",
        "name": "Eastern Seaboard sightings (2014–2019)",
        "date": "2014-01-01",
        "location": "U.S. East Coast",
        "summary": "Sustained observations of UAP by Navy aviators training along the East Coast through the 2010s.",
        "patterns": [r"\beastern\s+seaboard\b", r"\bRyan\s+Graves\b"],
    },
    {
        "id": "mosul-orb-2016",
        "name": "Mosul Orb (2016)",
        "date": "2016-04-16",
        "location": "Mosul, Iraq",
        "summary": "Surveillance footage of a sphere-shaped UAP loitering over a populated area.",
        "patterns": [r"\bMosul\s+orb\b", r"\bMosul\b.{0,40}\borb\b"],
    },
    {
        "id": "kabul-2021",
        "name": "Kabul UAP (2021)",
        "date": "2021-08-29",
        "location": "Kabul, Afghanistan",
        "summary": "MQ-9 Reaper footage of a small reflective sphere over Kabul during the U.S. evacuation.",
        "patterns": [r"\bKabul\b.{0,30}\bUAP\b", r"\bKabul\b.{0,30}\bsphere\b"],
    },
    {
        "id": "phoenix-lights-1997",
        "name": "Phoenix Lights (1997)",
        "date": "1997-03-13",
        "location": "Phoenix, AZ",
        "summary": "Mass-witness V-formation of lights observed over Arizona; one of the most-reported civilian UAP events.",
        "patterns": [r"\bPhoenix\s+Lights\b"],
    },
    {
        "id": "stephenville-2008",
        "name": "Stephenville (2008)",
        "date": "2008-01-08",
        "location": "Stephenville, TX",
        "summary": "Multi-witness UAP sighting in central Texas with corroborating radar tracks reported by MUFON.",
        "patterns": [r"\bStephenville\b"],
    },
    {
        "id": "roswell-1947",
        "name": "Roswell Incident (1947)",
        "date": "1947-07-08",
        "location": "Roswell, NM",
        "summary": "Recovery of debris near Roswell Army Air Field initially announced as a 'flying disc' before retraction.",
        "patterns": [r"\bRoswell\b"],
    },
    {
        "id": "kenneth-arnold-1947",
        "name": "Kenneth Arnold sighting (1947)",
        "date": "1947-06-24",
        "location": "Mt. Rainier, WA",
        "summary": "Civilian pilot's observation of nine objects 'skipping like saucers' — origin of the term 'flying saucer'.",
        "patterns": [r"\bKenneth\s+Arnold\b"],
    },
    {
        "id": "project-blue-book",
        "name": "Project Blue Book",
        "date": "1952-03-01",
        "location": "Wright-Patterson AFB, OH",
        "summary": "USAF UAP investigation program 1952–1969; ~12,600 reports catalogued.",
        "patterns": [r"\bProject\s+Blue\s*Book\b", r"\bBlue\s*Book\b"],
    },
    {
        "id": "project-sign-grudge",
        "name": "Projects Sign / Grudge",
        "date": "1948-01-22",
        "location": "Wright-Patterson AFB, OH",
        "summary": "USAF predecessor programs to Blue Book that handled the earliest UAP investigation files.",
        "patterns": [r"\bProject\s+Sign\b", r"\bProject\s+Grudge\b"],
    },
    {
        "id": "battle-of-la-1942",
        "name": "Battle of Los Angeles (1942)",
        "date": "1942-02-25",
        "location": "Los Angeles, CA",
        "summary": "Anti-aircraft barrage fired at unidentified objects over LA in the wake of Pearl Harbor.",
        "patterns": [r"\bBattle\s+of\s+Los\s+Angeles\b"],
    },
    {
        "id": "washington-dc-1952",
        "name": "Washington D.C. flap (1952)",
        "date": "1952-07-19",
        "location": "Washington, D.C.",
        "summary": "Two consecutive weekends of radar/visual UAP encounters over D.C. that prompted CIA's Robertson Panel.",
        "patterns": [r"\bWashington\s+(?:D\.?C\.?|flap|saucer)\b"],
    },
    {
        "id": "rendlesham-1980",
        "name": "Rendlesham Forest (1980)",
        "date": "1980-12-26",
        "location": "Suffolk, UK",
        "summary": "U.S. Air Force personnel reported lights and an object near RAF Bentwaters/Woodbridge.",
        "patterns": [r"\bRendlesham\b", r"\bBentwaters\b", r"\bWoodbridge\b"],
    },
    {
        "id": "skinwalker-ranch",
        "name": "Skinwalker Ranch / AAWSAP",
        "date": "2008-01-01",
        "location": "Uintah Basin, UT",
        "summary": "Site of long-running anomaly research backed by AAWSAP / Robert Bigelow.",
        "patterns": [r"\bSkinwalker\b", r"\bAAWSAP\b"],
    },
    {
        "id": "aatip-program",
        "name": "AATIP / Advanced Aerospace Threat Identification Program",
        "date": "2007-01-01",
        "location": "DoD",
        "summary": "Pentagon program 2007–2012 that catalogued UAP encounters; precursor to UAPTF and AARO.",
        "patterns": [r"\bAATIP\b"],
    },
    {
        "id": "uaptf",
        "name": "Unidentified Aerial Phenomena Task Force",
        "date": "2020-08-04",
        "location": "DoD",
        "summary": "DoD task force established 2020; folded into AARO in 2022.",
        "patterns": [r"\bUAPTF\b", r"\bUAP\s+Task\s+Force\b"],
    },
    {
        "id": "aaro",
        "name": "All-domain Anomaly Resolution Office",
        "date": "2022-07-15",
        "location": "DoD",
        "summary": "DoD office responsible for UAP investigation since 2022; publisher of historical record reviews.",
        "patterns": [r"\bAARO\b", r"\bAll[\s-]?domain\s+Anomaly\s+Resolution\b"],
    },
    {
        "id": "drone-swarms-2024",
        "name": "Northeast drone swarms (2024)",
        "date": "2024-11-15",
        "location": "NJ / NY / PA",
        "summary": "Mass civilian and law-enforcement reports of unidentified drone-like objects across the U.S. Northeast.",
        "patterns": [r"\bdrone\s+swarms?\b", r"\bNew\s+Jersey\s+drones?\b"],
    },
    {
        "id": "uss-omaha-2019",
        "name": "USS Omaha sphere encounter (2019)",
        "date": "2019-07-15",
        "location": "Off the coast of San Diego, CA",
        "summary": "Multiple unidentified spherical objects observed and recorded by USS Omaha and other Navy ships.",
        "patterns": [r"\bU?SS\s+Omaha\b"],
    },
    {
        "id": "f22-events-2023",
        "name": "North American shootdowns (Feb 2023)",
        "date": "2023-02-04",
        "location": "U.S. / Canada airspace",
        "summary": "F-22 engagements of a high-altitude balloon and three subsequent unidentified objects over North America.",
        "patterns": [r"\bChinese\s+balloon\b", r"\bspy\s+balloon\b", r"\bhigh[\s-]?altitude\s+balloon\b"],
    },
]


def detect_shape(blurb: str, text: str) -> str:
    corpus = (blurb or "") + " " + (text or "")[:16000]
    for name, rx in SHAPE_PATTERNS:
        if rx.search(corpus):
            return name
    return ""


def detect_incident(blurb: str, text: str) -> str:
    corpus = (blurb or "") + " " + (text or "")[:16000]
    for inc in INCIDENTS:
        for pat in inc["patterns"]:
            if re.search(pat, corpus, re.I):
                return inc["id"]
    return ""


def force_layout(node_ids: list[str], edges: list[tuple[int, int, float]],
                 iterations: int = 220) -> np.ndarray:
    """Fruchterman-Reingold-ish 2D layout. node_ids is the order; edges are
    (i, j, weight) into that ordering. Returns (n, 2) coords."""
    n = len(node_ids)
    if n == 0:
        return np.zeros((0, 2))
    rng = np.random.default_rng(42)  # deterministic
    pos = rng.uniform(-1, 1, size=(n, 2)) * 100
    if n == 1:
        return pos
    W, H = 1400.0, 900.0
    k = (W * H / n) ** 0.5
    temp = W / 8
    for it in range(iterations):
        # Repulsion (vectorised)
        diff = pos[:, None, :] - pos[None, :, :]                     # (n, n, 2)
        dist = np.linalg.norm(diff, axis=-1) + 1e-3                  # (n, n)
        rep_mag = (k * k) / dist                                      # (n, n)
        np.fill_diagonal(rep_mag, 0)
        rep_dir = diff / dist[..., None]                              # (n, n, 2)
        disp = (rep_dir * rep_mag[..., None]).sum(axis=1)             # (n, 2)
        # Attraction
        for (i, j, w) in edges:
            d = pos[i] - pos[j]
            dn = float(np.linalg.norm(d) + 1e-3)
            f = (dn * dn) / k * (0.4 + (w or 0))
            unit = d / dn
            disp[i] -= unit * f
            disp[j] += unit * f
        # Step + cool
        m = np.linalg.norm(disp, axis=1) + 1e-3
        step = np.minimum(m, temp)
        pos += disp / m[:, None] * step[:, None]
        temp = max(1.0, temp * 0.965)
    return pos


def main():
    docs = json.loads(INDEX_PATH.read_text())
    n_total = len(docs)

    # Augment per-record fields
    shape_count = 0
    incident_count = 0
    for d in docs:
        if not d.get("craft_shape"):
            s = detect_shape(d.get("blurb", ""), d.get("text", ""))
            if s:
                d["craft_shape"] = s
                shape_count += 1
            else:
                d["craft_shape"] = ""
        if not d.get("incident_id"):
            iid = detect_incident(d.get("blurb", ""), d.get("text", ""))
            if iid:
                d["incident_id"] = iid
                incident_count += 1
            else:
                d["incident_id"] = ""
        d["tranche_id"] = TRANCHE_ID
        d["last_verified"] = LAST_VERIFIED
    print(f"  shapes assigned:    {shape_count}/{n_total}")
    print(f"  incidents assigned: {incident_count}/{n_total}")

    # Write index back
    INDEX_PATH.write_text(json.dumps(docs, ensure_ascii=False))
    print(f"  → {INDEX_PATH}")

    # Emit incidents.json — MERGE: preserve hand-curated entries (incl. `status`)
    # and overlay/insert script-derived metadata for any matched incident.
    existing: dict = {}
    if INCIDENTS_PATH.exists():
        try:
            prev = json.loads(INCIDENTS_PATH.read_text())
            existing = prev.get("incidents", prev) or {}
        except Exception:
            existing = {}
    matched_ids = {d["incident_id"] for d in docs if d.get("incident_id")}
    merged: dict = dict(existing)
    for inc in INCIDENTS:
        if inc["id"] not in matched_ids and inc["id"] not in merged:
            continue  # script-derived but never matched, and not curated → skip
        cur = dict(merged.get(inc["id"], {}))
        for k, v in inc.items():
            if k == "patterns":
                continue
            if k == "status":
                continue  # never let the script clobber curated status
            cur.setdefault(k, v)
            # but DO refresh summary/date/location if the script has values
            if k in ("name", "summary", "date", "location"):
                cur[k] = v
        # restore curated status if any
        if "status" in merged.get(inc["id"], {}):
            cur["status"] = merged[inc["id"]]["status"]
        merged[inc["id"]] = cur
    INCIDENTS_PATH.write_text(json.dumps({"incidents": merged}, ensure_ascii=False, indent=2))
    print(f"  → {INCIDENTS_PATH} ({len(merged)} total: {len(matched_ids)} matched, {len(merged) - len(matched_ids)} curated-only)")

    # ─── today_index.json — MM-DD bucket for "This Day in Disclosure" ───
    today_idx: dict[str, list[dict]] = {}
    def _bucket(mm_dd: str, entry: dict):
        today_idx.setdefault(mm_dd, []).append(entry)

    def _parse_iso(s: str):
        if not s: return None
        try:
            y, m, d = s[:10].split("-")
            return int(y), int(m), int(d)
        except Exception:
            return None

    def _parse_us(s: str):
        # "5/8/26", "10/28/2001-10/29/2001"
        if not s: return None
        s = s.split("-")[0].strip()
        parts = s.split("/")
        if len(parts) != 3: return None
        try:
            mo, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 100: y += 2000 if y < 50 else 1900
            return y, mo, d
        except Exception:
            return None

    # Bucket incidents
    for iid, inc in merged.items():
        ymd = _parse_iso(inc.get("date", ""))
        if not ymd: continue
        y, m, d = ymd
        notability = 3 + (2 if inc.get("status") == "identified" else 0)
        _bucket(f"{m:02d}-{d:02d}", {
            "kind": "incident",
            "id": iid,
            "year": y,
            "label": inc.get("name") or iid,
            "notability": notability,
        })

    # Bucket records
    for d in docs:
        for src in (d.get("incident_date"), d.get("release_date")):
            ymd = _parse_iso(src) or _parse_us(src or "")
            if not ymd:
                continue
            y, mo, day = ymd
            notability = (
                (3 if d.get("incident_id") else 0)
                + len(d.get("dossier_hits") or {})
                + (1 if len(d.get("text") or "") > 5000 else 0)
            )
            _bucket(f"{mo:02d}-{day:02d}", {
                "kind": "document",
                "id": d["id"],
                "year": y,
                "label": d.get("title") or d["id"],
                "agency": d.get("agency") or "",
                "thumb": d.get("thumb_small") or (d.get("thumbnail_local") or [""])[0] if isinstance(d.get("thumbnail_local"), list) else d.get("thumbnail_local"),
                "notability": notability,
            })
            break  # don't double-count if both dates exist

    # Sort each bucket by notability desc, year desc
    for k in list(today_idx.keys()):
        today_idx[k].sort(key=lambda e: (-e["notability"], -e["year"]))

    TODAY_INDEX_PATH.write_text(json.dumps({"by_mmdd": today_idx, "generated": LAST_VERIFIED}, ensure_ascii=False))
    print(f"  → {TODAY_INDEX_PATH} ({len(today_idx)} MM-DD buckets, {sum(len(v) for v in today_idx.values())} entries)")

    # Compute force-directed graph layout for the full corpus.
    # Skip synthetic IMG records (parent_id is set on those) — including them
    # creates a hairball since each PDF has dozens of children.
    graph_docs = [d for d in docs if d.get("type") in ("PDF", "VID")]
    id_to_idx = {d["id"]: i for i, d in enumerate(graph_docs)}
    edges: list[tuple[int, int, float]] = []
    for i, d in enumerate(graph_docs):
        for s in d.get("similar_text", []) or []:
            j = id_to_idx.get(s["id"])
            if j is not None and j != i:
                edges.append((i, j, float(s.get("score", 0))))

    print(f"  computing graph layout — {len(graph_docs)} nodes, {len(edges)} edges…")
    coords = force_layout([d["id"] for d in graph_docs], edges)

    layout_payload = {
        "tranche_id": TRANCHE_ID,
        "generated": LAST_VERIFIED,
        "nodes": [
            {"id": d["id"], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
            for i, d in enumerate(graph_docs)
        ],
        "edge_count": len(edges),
    }
    LAYOUT_PATH.write_text(json.dumps(layout_payload, ensure_ascii=False))
    print(f"  → {LAYOUT_PATH}")


if __name__ == "__main__":
    main()

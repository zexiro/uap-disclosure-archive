"""Build static case dossier pages (ui/case/<id>/index.html) + case index.

For every incident/program in ui/incidents.json (minus the tic-tac-2004
duplicate, folded into nimitz-tic-tac-2004), match archive records by
keyword patterns over title/blurb/OCR text, rank them, and render a
fully server-side page per case — crawlable without JS, with OG tags and
JSON-LD. Also emits:

  ui/cases.json        case id → meta + ranked record ids (for JS pages)
  ui/record_cases.json record id → case ids (record-page badge)

Re-run whenever the archive index changes.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
OUT_DIR = UI / "case"
SITE = "https://uapdisclosuremirror.com"

# tic-tac-2004 is the script-derived duplicate of the curated nimitz entry.
SKIP_IDS = {"tic-tac-2004"}
PROGRAM_IDS = {"aaro", "uaptf", "project-blue-book", "project-sign-grudge"}

PATTERNS: dict[str, list[str]] = {
    "foo-fighters-1944": [r"\bfoo[\s-]?fighters?\b"],
    "arnold-1947": [r"\bKenneth\s+Arnold\b", r"\bMt\.?\s+Rainier\b.{0,60}\b(?:saucer|nine|1947)\b"],
    "roswell-1947": [r"\bRoswell\b", r"\bJesse\s+Marcel\b"],
    "mantell-1948": [r"\bMantell\b"],
    "mariana-film-1950": [r"\bMariana\b", r"\bGreat\s+Falls\b.{0,50}\b(?:film|objects?)\b"],
    "washington-dc-1952": [r"\bWashington\s+(?:D\.?C\.?|flap|saucer)\b", r"\bNational\s+Airport\b.{0,50}\b(?:1952|radar)\b"],
    "soccoro-1964": [r"\bZamora\b", r"\bSocorro\b"],
    "malmstrom-1967": [r"\bMalmstrom\b", r"\bRobert\s+Salas\b", r"\bMinuteman\b.{0,50}\b(?:offline|shutdown|UAP|UFO)\b"],
    "iran-1976": [r"\bTehran\b.{0,60}\b(?:F-4|1976|UAP|UFO)\b", r"\bJafari\b", r"\bYousefi\b"],
    "rendlesham-1980": [r"\bRendlesham\b", r"\bBentwaters\b", r"\bWoodbridge\b", r"\bHalt\b.{0,40}\b(?:memorandum|tape)\b"],
    "jal-1628-1986": [r"\bJAL\s*1628\b", r"\bflight\s+1628\b", r"\bTerauchi\b"],
    "belgian-wave-1989": [r"\bBelgian\b.{0,30}\b(?:wave|triangle)\b", r"\bBelgium\b.{0,40}\b(?:triangle|UAP|UFO)\b"],
    "phoenix-lights-1997": [r"\bPhoenix\s+Lights\b"],
    "ohare-2006": [r"\bO'?Hare\b"],
    "stephenville-2008": [r"\bStephenville\b"],
    "nimitz-tic-tac-2004": [
        r"\btic[\s-]?tac\b", r"\bU?SS\s+Nimitz\b", r"\bFravor\b", r"\bPrincipi\b",
        r"\bFLIR\s*1\b", r"\bFLIR-1\b",
    ],
    "aguadilla-2013": [r"\bAguadilla\b", r"\bRafael\s+Hernandez\b"],
    "eastern-seaboard-2014-2019": [r"\beastern\s+seaboard\b", r"\bRyan\s+Graves\b"],
    "gimbal-2015": [r"\bgimbal\b", r"\bATFLIR\b"],
    "go-fast-2015": [r"\bgo[\s-]?fast\b"],
    "kabul-2021": [r"\bKabul\b.{0,30}\b(?:UAP|sphere)\b"],
    "project-blue-book": [r"\bProject\s+Blue\s*Book\b", r"\bBlue\s*Book\b"],
    "project-sign-grudge": [r"\bProject\s+Sign\b", r"\bProject\s+Grudge\b"],
    "uaptf": [r"\bUAPTF\b", r"\bUAP\s+Task\s+Force\b"],
    "aaro": [r"\bAARO\b", r"\bAll[\s-]?domain\s+Anomaly\s+Resolution\b"],
}

MAX_STORED = 60   # ids kept in cases.json
MAX_SHOWN = 48    # cards rendered on the page

BADGE_CLS = {"PDF": "pdf", "VID": "vid", "AUD": "aud", "IMG": "img"}


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def thumb_of(d: dict) -> str:
    t = d.get("thumb_small") or ""
    if not t:
        tl = d.get("thumbnail_local")
        if isinstance(tl, list) and tl:
            t = tl[0]
        elif isinstance(tl, str):
            t = tl
    return t


def doc_year_of(d: dict) -> int | None:
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", d.get("incident_date") or "")
    if not m:
        return None
    y = int(m.group(3))
    if y < 100:
        y += 2000 if y < 30 else 1900
    return y


def main() -> None:
    incidents = json.loads((UI / "incidents.json").read_text())["incidents"]
    full = json.loads((UI / "search-index.json").read_text())
    lite = {d["id"]: d for d in json.loads((UI / "records-lite.json").read_text())}

    case_ids = [i for i in incidents if i not in SKIP_IDS]
    compiled = {i: [re.compile(p, re.I) for p in PATTERNS.get(i, [])] for i in case_ids}

    # ── Match records to cases (full OCR text) ───────────────────────
    scores: dict[str, dict[str, int]] = {i: {} for i in case_ids}
    for d in full:
        title = d.get("title", "")
        blurb = d.get("blurb", "")
        text = d.get("text") or ""
        for cid, pats in compiled.items():
            if not pats:
                continue
            s = 0
            for p in pats:
                if p.search(title):
                    s += 5
                if p.search(blurb):
                    s += 3
                if p.search(text):
                    s += 1
            if s:
                scores[cid][d["id"]] = s

    ranked = {
        cid: sorted(scores[cid], key=lambda rid: -scores[cid][rid])[:MAX_STORED]
        for cid in case_ids
    }

    # ── Era fallback: for thin cases, nearby-year records for context ─
    lite_docs = list(lite.values())
    era: dict[str, list[str]] = {}
    for cid in case_ids:
        if len(ranked[cid]) >= 6:
            era[cid] = []
            continue
        m = re.match(r"(\d{4})", incidents[cid].get("date", ""))
        if not m:
            era[cid] = []
            continue
        cy = int(m.group(1))
        pool = [
            d for d in lite_docs
            if doc_year_of(d) is not None and abs(doc_year_of(d) - cy) <= 3
            and d["id"] not in scores[cid]
        ]
        pool.sort(key=lambda d: (not thumb_of(d), d.get("release_date") or ""), reverse=False)
        era[cid] = [d["id"] for d in pool[:12]]

    # ── JSON outputs ─────────────────────────────────────────────────
    cases_json = {}
    for cid in case_ids:
        inc = incidents[cid]
        cases_json[cid] = {
            **{k: inc.get(k, "") for k in ("id", "name", "date", "location", "status", "summary")},
            "kind": "program" if cid in PROGRAM_IDS else "incident",
            "records": ranked[cid],
            "era": era[cid],
            "count": len(scores[cid]),
        }
    (UI / "cases.json").write_text(json.dumps(cases_json, ensure_ascii=False, indent=1))

    record_cases: dict[str, list[str]] = {}
    for cid, ids in ranked.items():
        for rid in ids:
            record_cases.setdefault(rid, []).append(cid)
    for rid in record_cases:
        record_cases[rid] = record_cases[rid][:5]
    (UI / "record_cases.json").write_text(json.dumps(record_cases, ensure_ascii=False))
    print(f"cases.json: {len(cases_json)} cases · record_cases.json: {len(record_cases)} records")

    # ── Static pages ─────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    chrono = sorted(case_ids, key=lambda c: incidents[c].get("date", "9999"))
    for cid in case_ids:
        render_case_page(cid, incidents[cid], ranked[cid], lite, chrono, era[cid])
    render_index_page(cases_json, chrono)
    print(f"wrote {len(case_ids)} case pages + index → {OUT_DIR}")


def render_era_section(era_ids: list[str], lite: dict) -> str:
    cards = "\n".join(record_card(r, lite) for r in era_ids)
    return f"""<h2 class="case-h2 case-h2-era">From the same era <span class="case-h2-note">not direct references</span></h2>
  <div class="case-grid">
    {cards}
  </div>"""


NAV = """<nav class="topnav"><a class="topnav-brand" href="/ui/index.html">UFO/UAP <span>Disclosure Archive</span></a><div class="topnav-links"><a href="/ui/search.html">🔎 Search</a><a href="/ui/map.html">🗺 Map</a><a href="/ui/timeline.html">📅 Timeline</a><a href="/ui/graph.html">🕸 Graph</a><a href="/ui/ask.html">⌘ Ask</a><a href="/ui/case/" class="active">◉ Cases</a></div><a class="topnav-coffee" href="https://buymeacoffee.com/uapdisclosuremirror" target="_blank" rel="noopener" title="Help keep this archive online">☕</a></nav>"""


def head(title: str, description: str, canonical: str, jsonld: dict | None = None) -> str:
    ld = f'<script type="application/ld+json">{json.dumps(jsonld, ensure_ascii=False)}</script>' if jsonld else ""
    return f"""<meta charset="utf-8" />
<title>{esc(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<meta name="description" content="{esc(description[:155])}" />
<link rel="canonical" href="{canonical}" />
<meta property="og:type" content="article" />
<meta property="og:title" content="{esc(title)}" />
<meta property="og:description" content="{esc(description[:200])}" />
<meta property="og:url" content="{canonical}" />
<meta property="og:image" content="{SITE}/ui/og/og.png" />
<meta name="twitter:card" content="summary_large_image" />
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%230b0e14'/%3E%3Cellipse cx='32' cy='34' rx='22' ry='6' fill='%23ffd34d'/%3E%3Ccircle cx='32' cy='28' r='8' fill='%2357c7ff'/%3E%3C/svg%3E" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;450;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/ui/css/app.css" />
{ld}"""


def record_card(rid: str, lite: dict) -> str:
    d = lite.get(rid)
    if not d:
        return ""
    badge_cls = BADGE_CLS.get(d.get("type", ""), "img")
    thumb = thumb_of(d)
    img = (
        f'<div class="cc-thumb"><img loading="lazy" src="/{esc(thumb)}" alt="" onerror="this.parentNode.style.display=\'none\'"/></div>'
        if thumb else ""
    )
    meta = " · ".join(x for x in [d.get("agency"), d.get("incident_date"), d.get("incident_location")] if x)
    blurb = (d.get("blurb") or "")[:150]
    return f"""<a class="case-card" href="/ui/record.html?id={esc(rid)}">
  {img}
  <div class="cc-body">
    <div class="cc-top"><span class="badge {badge_cls}">{esc(d.get("type") or "?")}</span></div>
    <div class="cc-title">{esc(d.get("title") or "Untitled record")}</div>
    <div class="cc-meta">{esc(meta)}</div>
    {f'<div class="cc-blurb">{esc(blurb)}…</div>' if blurb else ""}
  </div>
</a>"""


def render_case_page(cid: str, inc: dict, rids: list[str], lite: dict, chrono: list[str], era_ids: list[str]) -> None:
    kind = "program" if cid in PROGRAM_IDS else "incident"
    name = inc.get("name", cid)
    summary = inc.get("summary", "")
    status = (inc.get("status") or "unresolved").lower()
    date = inc.get("date", "")
    loc = inc.get("location", "")

    docs = [lite[r] for r in rids if r in lite]
    agencies = sorted({d.get("agency") for d in docs if d.get("agency")})
    types: dict[str, int] = {}
    for d in docs:
        t = d.get("type") or "?"
        types[t] = types.get(t, 0) + 1
    years = set()
    for d in docs:
        y = doc_year_of(d)
        if y:
            years.add(y)
    years = sorted(years)
    q = re.sub(r"\s*\(.*?\)", "", name).strip()

    pos = chrono.index(cid)
    prev_c = chrono[pos - 1] if pos > 0 else None
    next_c = chrono[pos + 1] if pos < len(chrono) - 1 else None

    cards = "\n".join(record_card(r, lite) for r in rids[:MAX_SHOWN])
    type_chips = " ".join(f'<span class="badge {BADGE_CLS.get(t, "img")}">{esc(t)}</span> ×{n}' for t, n in sorted(types.items()))
    year_span = f"{years[0]}–{years[-1]}" if len(years) > 1 else (str(years[0]) if years else "—")

    jsonld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": f"{name} — declassified files",
        "description": summary,
        "about": {
            "@type": "Event",
            "name": name,
            "startDate": date,
            "location": loc,
        },
    }

    rel_links = []
    if prev_c:
        rel_links.append(f'<a class="case-rel" href="/ui/case/{prev_c}/">← {esc(incidents_name(prev_c))}</a>')
    if next_c:
        rel_links.append(f'<a class="case-rel" href="/ui/case/{next_c}/">{esc(incidents_name(next_c))} →</a>')

    body = f"""<article class="case-main">
  <div class="case-crumb"><a href="/ui/case/">CASES</a> / <span>{kind.upper()}</span></div>
  <h1 class="case-title">{esc(name)}</h1>
  <div class="case-chips">
    <span class="status-pill {esc(status)}">{esc(status.upper())}</span>
    <span class="case-chip">{esc(date)}</span>
    <span class="case-chip">{esc(loc)}</span>
  </div>
  <p class="case-summary">{esc(summary)}</p>

  <div class="case-stats">
    <div><strong>{len(rids)}</strong><span>archive records</span></div>
    <div><strong>{len(agencies)}</strong><span>agencies</span></div>
    <div><strong>{esc(year_span)}</strong><span>record years</span></div>
    <div>{type_chips}</div>
  </div>

  <h2 class="case-h2">Declassified records referencing this {kind}</h2>
  <div class="case-grid">
    {cards if cards else '<p class="case-empty">No direct references in the archive yet — matching improves as OCR coverage grows.</p>'}
  </div>

  {render_era_section(era_ids, lite) if era_ids else ""}

  <div class="case-rel-row">{"<span class='case-rel-sep'></span>".join(rel_links)}</div>

  <div class="case-explore">
    <a href="/ui/timeline.html">📅 See it on the timeline</a>
    <a href="/ui/search.html?q={esc(q)}">🔎 Search the archive</a>
    <a href="/ui/map.html">🌐 Explore the globe</a>
  </div>
</article>"""

    page = f"""<!doctype html>
<html lang="en">
<head>
{head(f"{name} — declassified files & records", summary or f"Declassified records referencing {name}.", f"{SITE}/ui/case/{cid}/", jsonld)}
</head>
<body class="case-page">
{NAV}
{body}
</body>
</html>
"""
    d = OUT_DIR / cid
    d.mkdir(exist_ok=True)
    (d / "index.html").write_text(page)


_incidents_cache: dict = {}


def incidents_name(cid: str) -> str:
    if not _incidents_cache:
        _incidents_cache.update(json.loads((UI / "incidents.json").read_text())["incidents"])
    return (_incidents_cache.get(cid) or {}).get("name", cid)


def render_index_page(cases: dict, chrono: list[str]) -> None:
    def tile(cid: str) -> str:
        c = cases[cid]
        status = (c.get("status") or "unresolved").lower()
        return f"""<a class="case-tile" href="/ui/case/{cid}/">
  <div class="ct-top"><span class="status-pill {esc(status)}">{esc(status.upper())}</span><span class="ct-date">{esc(c.get("date", "")[:4])}</span></div>
  <div class="ct-name">{esc(c.get("name", cid))}</div>
  <div class="ct-loc">{esc(c.get("location", ""))}</div>
  <div class="ct-count">{c["count"]} record{"s" if c["count"] != 1 else ""}</div>
</a>"""

    incidents_part = "\n".join(tile(c) for c in chrono if cases[c]["kind"] == "incident")
    programs_part = "\n".join(tile(c) for c in chrono if cases[c]["kind"] == "program")

    page = f"""<!doctype html>
<html lang="en">
<head>
{head("Case dossiers — famous UAP incidents & programs", "Deep-dive dossiers on the major UAP incidents and government programs, each pulling every declassified record that references it.", f"{SITE}/ui/case/")}
</head>
<body class="case-page">
{NAV}
<main class="case-main">
  <h1 class="case-title">Case dossiers</h1>
  <p class="case-summary">Every major incident and program in the archive, with the declassified records that reference it.</p>
  <h2 class="case-h2">Incidents</h2>
  <div class="case-tiles">{incidents_part}</div>
  <h2 class="case-h2">Programs &amp; offices</h2>
  <div class="case-tiles">{programs_part}</div>
</main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(page)


if __name__ == "__main__":
    main()

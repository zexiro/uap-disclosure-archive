#!/usr/bin/env python3
"""Generate an Obsidian-friendly vault from records.json + extracted PDF text.

Layout:
  vault/
    README.md                 — landing page with dashboard queries
    Index/
      By Agency.md
      By Location.md
      By Type.md
      Timeline.md
    Releases/
      <id>.md                 — one note per record (frontmatter + blurb + linked media + extracted text)
    media/                    — symlinks to raw assets so embeds render
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
VAULT = ROOT / "vault"
RECORDS = json.loads((RAW / "records.json").read_text())
TEXT_DIR = RAW / "text"
LINKS_PATH = RAW / "links.json"
LINKS = json.loads(LINKS_PATH.read_text()) if LINKS_PATH.exists() else {}


def yaml_escape(s):
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    if any(c in s for c in [":", "#", "'", '"', "[", "]", ",", "&", "*", "!", "|", ">", "%", "@", "`"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def yaml_list(items):
    items = [i for i in items if i]
    if not items:
        return "[]"
    return "[" + ", ".join(yaml_escape(i) for i in items) + "]"


def parse_date(s):
    if not s or s == "N/A":
        return None
    # Common forms: "5/8/26", "May 2022", "12/22/2017"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s.strip())
    if m:
        mo, d, y = m.groups()
        y = int(y)
        if y < 100:
            y += 2000
        return f"{y:04d}-{int(mo):02d}-{int(d):02d}"
    return s.strip()


def find_text(rec):
    """Find extracted .txt file for a record, if present."""
    for lp in rec.get("primary_local", []):
        if lp.endswith(".pdf"):
            txt = TEXT_DIR / (Path(lp).stem + ".txt")
            if txt.exists():
                return txt
    return None


def media_relpath(local_path: str) -> str:
    """Return path to raw asset relative to vault note (vault/Releases/X.md)."""
    return os.path.relpath(ROOT / local_path, VAULT / "Releases")


def build_release_note(rec):
    title = rec["title"] or rec["id"]
    safe_title = re.sub(r"[\\/:*?\"<>|]+", "-", title)[:120].strip()

    fm = {
        "id": rec["id"],
        "title": title,
        "agency": rec["agency"],
        "type": rec["type"].strip(),
        "release_date": parse_date(rec["release_date"]) or rec["release_date"],
        "incident_date": rec["incident_date"],
        "incident_location": rec["incident_location"],
        "redaction": rec["redaction"],
        "tags": [
            "uap",
            f"agency/{(rec['agency'] or 'Unknown').lower().replace(' ', '_')}",
            f"type/{rec['type'].strip().lower()}",
        ],
    }
    if rec.get("dvids_video_id"):
        fm["dvids_video_id"] = rec["dvids_video_id"]

    # Build frontmatter
    fm_lines = ["---"]
    for k, v in fm.items():
        if k == "tags":
            fm_lines.append(f"tags: {yaml_list(v)}")
        else:
            fm_lines.append(f"{k}: {yaml_escape(v)}")
    # Source URLs
    sources = [u for u in [rec.get("pdf_image_link", "")] if u.startswith("http")]
    fm_lines.append(f"source_urls: {yaml_list(sources)}")
    fm_lines.append("---")

    body = ["", f"# {title}", ""]

    # Blurb
    if rec.get("blurb"):
        body.append("## Description")
        body.append("")
        body.append(rec["blurb"].strip())
        body.append("")

    # Quick facts
    body.append("## Facts")
    body.append("")
    body.append(f"- **Agency:** {rec['agency']}")
    body.append(f"- **Type:** {rec['type'].strip()}")
    body.append(f"- **Released:** {fm['release_date']}")
    body.append(f"- **Incident Date:** {rec['incident_date']}")
    body.append(f"- **Location:** {rec['incident_location']}")
    if rec.get("redaction"):
        body.append(f"- **Redaction:** {rec['redaction']}")
    body.append("")

    # Thumbnail / preview
    for thumb in rec.get("thumbnail_local", []):
        body.append(f"![[{Path(thumb).name}]]")
    for primary in rec.get("primary_local", []):
        suf = Path(primary).suffix.lower()
        if suf in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            body.append(f"![[{Path(primary).name}]]")
    if rec.get("video_local"):
        body.append(f"![[{Path(rec['video_local']).name}]]")
    body.append("")

    # Document links
    body.append("## Files")
    body.append("")
    for lp in rec.get("primary_local", []):
        rel = media_relpath(lp)
        body.append(f"- [Local: {Path(lp).name}]({rel})")
    if rec.get("video_local"):
        body.append(f"- [Local video: {Path(rec['video_local']).name}]({media_relpath(rec['video_local'])})")
    if rec.get("pdf_image_link"):
        body.append(f"- [Source: war.gov]({rec['pdf_image_link']})")
    body.append("")

    # AI-confirmed enrichments (only `approved` claims, never raw candidates).
    enrich_path = ROOT / "ui" / "enrichments" / f"document_{re.sub(r'[^a-zA-Z0-9._-]+', '_', rec['id'])[:160]}.json"
    if enrich_path.exists():
        try:
            store = json.loads(enrich_path.read_text())
            approved = []
            for run in store.get("runs", []):
                for c in run.get("claims", []):
                    if c.get("status") == "approved":
                        approved.append(c)
            if approved:
                body.append("## AI-Confirmed Enrichments")
                body.append("")
                body.append("> Web-discovered facts that were verified and manually approved.")
                body.append("> Source: Disclosure Archive enrichment pipeline.")
                body.append("")
                for c in approved:
                    body.append(f"- **{c.get('claim','').strip()}**")
                    meta = []
                    if c.get('verdict'): meta.append(c['verdict'])
                    if c.get('confidence'): meta.append(f"conf: {c['confidence']}")
                    if c.get('date'): meta.append(c['date'])
                    if c.get('location'): meta.append(c['location'])
                    if meta:
                        body.append(f"  - _{' · '.join(meta)}_")
                    for u in (c.get('supporting_urls') or [])[:5]:
                        body.append(f"  - [Source]({u})")
                body.append("")
        except Exception:
            pass

    # Extracted PDF text
    txt = find_text(rec)
    if txt:
        body.append("## Extracted Text")
        body.append("")
        body.append("```text")
        body.append(txt.read_text(errors="replace")[:200_000])
        body.append("```")

    note_path = VAULT / "Releases" / f"{safe_title}.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(fm_lines + body))
    return safe_title


def build_index_pages(titles_by_record):
    by_agency = defaultdict(list)
    by_type = defaultdict(list)
    by_location = defaultdict(list)
    timeline = []
    for rec, safe_title in titles_by_record:
        by_agency[rec["agency"] or "Unknown"].append(safe_title)
        by_type[rec["type"].strip() or "Unknown"].append(safe_title)
        loc = (rec["incident_location"] or "Unknown").strip() or "Unknown"
        by_location[loc].append(safe_title)
        timeline.append((parse_date(rec["incident_date"]) or "Unknown", safe_title, rec["agency"]))

    def write_groups(filename, title, groups):
        lines = [f"# {title}", ""]
        for k in sorted(groups):
            lines.append(f"## {k}  *({len(groups[k])})*")
            lines.append("")
            for st in sorted(set(groups[k])):
                lines.append(f"- [[{st}]]")
            lines.append("")
        (VAULT / "Index" / filename).write_text("\n".join(lines))

    (VAULT / "Index").mkdir(parents=True, exist_ok=True)
    write_groups("By Agency.md", "Releases by Agency", by_agency)
    write_groups("By Type.md", "Releases by Type", by_type)
    write_groups("By Location.md", "Releases by Incident Location", by_location)

    # Timeline
    timeline.sort(key=lambda x: (x[0] == "Unknown", x[0]))
    lines = ["# Timeline", "", "| Incident Date | Agency | Title |", "|---|---|---|"]
    for date, st, agency in timeline:
        lines.append(f"| {date} | {agency} | [[{st}]] |")
    (VAULT / "Index" / "Timeline.md").write_text("\n".join(lines))


def build_readme(records):
    counts = defaultdict(int)
    for r in records:
        counts[r["agency"] or "Unknown"] += 1
    lines = [
        "# UFO/UAP Disclosure Vault",
        "",
        f"Mirror of [war.gov/UFO](https://www.war.gov/UFO/) Release 1, fetched on the day it dropped.",
        "",
        f"**{len(records)} records** across these agencies:",
        "",
    ]
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Browse",
        "",
        "- [[Index/By Agency]]",
        "- [[Index/By Type]]",
        "- [[Index/By Location]]",
        "- [[Index/Timeline]]",
        "",
        "## Search the corpus",
        "",
        "Open `ui/index.html` in a browser for full-text search across every PDF + metadata.",
        "Inside Obsidian, press `Ctrl/Cmd-Shift-F` for vault-wide grep.",
        "",
        "## Find patterns",
        "",
        "- Open Obsidian's **graph view** (`Ctrl/Cmd-G`): every note's Related section",
        "  creates wiki-links, so visually similar items and content-similar items form clusters.",
        "  Bigger clumps = stronger thematic groupings.",
        "- The Related sections come from `raw/links.json`, generated by `scripts/build_links.py`",
        "  using TF-IDF cosine on extracted PDF text + perceptual-hash on images.",
        "- When future tranches drop: rerun `make all` (or just `python3 scripts/build_links.py`",
        "  followed by `build_vault.py` and `build_search_index.py`) to recompute links across",
        "  the combined corpus.",
        "",
        "## Layout",
        "",
        "- `Releases/` — one note per disclosure item, with frontmatter, blurb, embedded images/video, and full extracted PDF text.",
        "- `Index/` — generated cross-cut indexes.",
        "- `../raw/` — original downloads (PDFs, images, videos, DVIDS metadata).",
        "- `../raw/text/` — extracted plaintext per PDF.",
        "",
    ]
    (VAULT / "README.md").write_text("\n".join(lines))


def append_related_section(note_path: Path, rec, id_to_safe):
    """Append a Related section with [[wikilinks]] derived from links.json."""
    info = LINKS.get(rec["id"], {})
    text = info.get("similar_text", [])
    image = info.get("similar_image", [])
    if not text and not image:
        return
    lines = ["", "## Related", ""]
    if text:
        lines.append("**By content (TF-IDF cosine):**")
        for s in text:
            target = id_to_safe.get(s["id"])
            if not target:
                continue
            lines.append(f"- [[{target}]] — score {s['score']:.2f} · {s['type']}")
        lines.append("")
    if image:
        lines.append("**Visually similar (pHash):**")
        for s in image:
            target = id_to_safe.get(s["id"])
            if not target:
                continue
            lines.append(f"- [[{target}]] — distance {s['distance']} · {s['type']}")
        lines.append("")
    with note_path.open("a") as f:
        f.write("\n".join(lines))


def main():
    VAULT.mkdir(exist_ok=True)
    titles_by_record = []
    used = {}
    for rec in RECORDS:
        # Predict the safe_title that build_release_note will produce, then
        # disambiguate before writing if it collides with a prior note.
        title = rec["title"] or rec["id"]
        base = re.sub(r"[\\/:*?\"<>|]+", "-", title)[:120].strip()
        n = used.get(base, 0) + 1
        used[base] = n
        if n > 1:
            # Tag this record's title so build_release_note picks up the suffix
            rec["title"] = f"{title} ({n})"
        st = build_release_note(rec)
        titles_by_record.append((rec, st))
    # Second pass: append Related sections (needed both ids resolved)
    id_to_safe = {rec["id"]: st for rec, st in titles_by_record}
    for rec, st in titles_by_record:
        np = VAULT / "Releases" / f"{st}.md"
        append_related_section(np, rec, id_to_safe)
    build_index_pages(titles_by_record)
    build_readme(RECORDS)
    print(f"Wrote {len(titles_by_record)} notes to {VAULT}/Releases")


if __name__ == "__main__":
    main()

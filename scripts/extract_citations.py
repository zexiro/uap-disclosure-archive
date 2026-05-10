#!/usr/bin/env python3
"""Extract in-PDF case-number citations and build an explicit citation graph.

Corpus-driven regex patterns (verified against sampled text files):
  1. Incident numbers (Project Blue Book era, 38_143685_box7 files):
       "Incident #106", "Incident # 112a", "See Incident 175"
  2. Blue Book case numbers (blue_book_NNNN records):
       "Case No. 1234", "Case #1234", "Blue Book case #1234"
  3. MDR (Mission Debrief Report) IDs (DOW mission reports):
       "MDR 25-0094", "MDR 25-0094 thru MDR 25-0099"
  4. Diplomatic cable message references (State Dept cables):
       "REF: (A) MOSCOW 13072", "Reference: (A) 23 MEXICO 2468"
  5. Serial / Section references within FBI file series (65_HS1* records):
       "Serial 449", "Section 9" (only when referencing a known sibling)
  6. FOIPA numbers:
       "FOIPA#222227"

Outputs:
  ui/citations.json  — edges + dangling + by_record index

Resolution strategy:
  - Incident #N  → 38_143685_box* (the incident summary PDFs cover ranges)
  - Case #N      → blue_book_N (exact match on case_number field)
  - MDR NN-NNNN  → DOW-UAP-PR records whose file references that MDR range
  - Cable origin → State_Department_UAP_Cable_* by station/date
  - Serial/Section → sibling records in same file series
"""
import json
import re
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
TEXT_DIR = RAW / "text"
SEARCH_INDEX = ROOT / "ui" / "search-index.json"
OUT_PATH = ROOT / "ui" / "citations.json"

# ---------------------------------------------------------------------------
# Regex patterns (corpus-verified)
# ---------------------------------------------------------------------------

PATTERNS = {
    # Incident #NN or Incident # NN (optional suffix a/b/c/d/e/f)
    "incident": re.compile(
        r"\bIncident\s*#?\s*([0-9]{1,3}[a-f]?(?:\s*,\s*[0-9]{1,3}[a-f]?)*)\b",
        re.IGNORECASE,
    ),
    # Blue Book case number: "Case No. 1234", "Case #1234", "case #12"
    "blue_book_case": re.compile(
        r"\b(?:Blue Book\s+)?[Cc]ase\s+(?:No\.?\s*|#\s*|Num\.?\s*)?([0-9]{1,6})\b",
        re.IGNORECASE,
    ),
    # MDR NN-NNNN (USCENTCOM Mission Debrief Reports)
    "mdr": re.compile(
        r"\bMDR\s+([0-9]{2}-[0-9]{4})\b",
        re.IGNORECASE,
    ),
    # Diplomatic cable references: "MOSCOW 13072", "TBILISI 3087"
    # Only all-caps station names with 4-6 digit number
    "cable_ref": re.compile(
        r"\b([A-Z]{4,12})\s+([0-9]{4,6})\b",
    ),
    # FOIPA numbers
    "foipa": re.compile(
        r"\bFOIPA#?([0-9]+)\b",
        re.IGNORECASE,
    ),
    # Serial references (within FBI file series): "Serial 449", "serial 130"
    "serial_ref": re.compile(
        r"\bSerial\s+([0-9]+)\b",
        re.IGNORECASE,
    ),
    # Section references (within multi-section FBI files): "Section 9"
    "section_ref": re.compile(
        r"\bSection\s+([0-9]+)\b",
        re.IGNORECASE,
    ),
}

# Cable station names that appear in this corpus (from sampling)
KNOWN_CABLE_STATIONS = {
    "MOSCOW", "TBILISI", "ASHGABAT", "DUSHANBE", "MEXICO",
    "AMEMBASSY", "SECSTATE", "WASHDC",
}

# MDR ranges per DOW-PR file (from corpus sampling):
# "MDR 25-0094 thru MDR 25-0099" appears in dow-uap-d12/pr20 etc.
# We build this index from the text files at runtime.

CONTEXT_WINDOW = 120  # chars around each match

# ---------------------------------------------------------------------------
# Build lookup tables from search index
# ---------------------------------------------------------------------------


def build_lookup_tables(records):
    """Return multiple index structures for resolution."""
    # 1. Case number -> record id (blue book)
    case_num_to_id = {}
    for r in records:
        cn = r.get("case_number")
        if cn is not None:
            try:
                case_num_to_id[int(cn)] = r["id"]
            except (ValueError, TypeError):
                pass

    # 2. Incident range bounds: 38_143685 files cover ranges 1-100, 101-172, 173-233
    # Map each incident number to which record covers it
    incident_ranges = []
    for r in records:
        rid = r["id"]
        # Matches like "38_143685_box7_Incident_Summaries_1-100"
        m = re.search(r"Incident_Summaries_(\d+)-(\d+)", rid, re.IGNORECASE)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            incident_ranges.append((lo, hi, rid))

    def incident_to_id(n):
        for lo, hi, rid in incident_ranges:
            if lo <= n <= hi:
                return rid
        return None

    # 3. MDR number -> DOW-PR record (by scanning PR record titles/files)
    mdr_to_pr_id = {}
    for r in records:
        rid = r["id"]
        if "DOW-UAP-PR" in rid or "DOW-UAP-D" in rid:
            # Check the text file associated with this record
            text = _load_text(r)
            for m in PATTERNS["mdr"].finditer(text):
                mdr_num = m.group(1)
                if mdr_num not in mdr_to_pr_id:
                    mdr_to_pr_id[mdr_num] = []
                if rid not in mdr_to_pr_id[mdr_num]:
                    mdr_to_pr_id[mdr_num].append(rid)

    # 4. Serial/Section -> sibling records in the same file series
    # Key = (series_prefix, type, number) -> record_id
    # e.g. ("65_HS1-834228961_62-HQ-83894", "Serial", 449) -> ...
    series_member_index = {}
    for r in records:
        rid = r["id"]
        # Match "65_HS1-834228961_62-HQ-83894_Serial_449"
        m = re.match(r"^(.+?)_(Serial|Section)_([0-9]+)$", rid, re.IGNORECASE)
        if m:
            prefix, kind, num = m.group(1), m.group(2).lower(), int(m.group(3))
            series_member_index[(prefix, kind, num)] = rid

    # 5. Cable station lookup: record_id -> (station, cable_num) parsed from title
    cable_station_index = {}
    for r in records:
        if "State_Department_UAP_Cable" in r["id"]:
            # Extract station from title like "MOSCOW 13169"
            title = r.get("title", "")
            for pat in [
                re.compile(r"([A-Z]{4,12})\s+([0-9]{4,6})"),
            ]:
                m = pat.search(title)
                if m and m.group(1) in KNOWN_CABLE_STATIONS:
                    key = (m.group(1), m.group(2))
                    cable_station_index[key] = r["id"]

    # Also index by the MRN field in the text
    for r in records:
        if "State_Department_UAP_Cable" in r["id"]:
            text = _load_text(r)
            m = re.search(r"MRN:\s+(\d{2})\s+([A-Z]+)\s+([0-9]+)", text)
            if m:
                station = m.group(2)
                num = m.group(3)
                cable_station_index[(station, num)] = r["id"]

    return {
        "case_num": case_num_to_id,
        "incident_fn": incident_to_id,
        "mdr": mdr_to_pr_id,
        "series_member": series_member_index,
        "cable_station": cable_station_index,
    }


# ---------------------------------------------------------------------------
# Text loading (mirrors build_links.py logic)
# ---------------------------------------------------------------------------


def _load_text(rec):
    """Load full text for a record: try raw/text/ file first, then 'text' field."""
    chunks = []
    for lp in rec.get("primary_local", []):
        if lp.endswith(".pdf"):
            txt_path = TEXT_DIR / (Path(lp).stem + ".txt")
            if txt_path.exists():
                chunks.append(txt_path.read_text(errors="replace"))
    # Also include embedded 'text' field
    inline = rec.get("text", "")
    if inline:
        chunks.append(inline)
    return "\n".join(chunks)


def _context(text, match, window=CONTEXT_WINDOW):
    """Return a snippet of text around a regex match."""
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end].replace("\n", " ").strip()
    # Highlight match position with ellipsis
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


# ---------------------------------------------------------------------------
# Extract citations from a single record's text
# ---------------------------------------------------------------------------


def extract_citations_from_text(rec, text, lookups):
    """
    Returns list of dicts:
      {"token": str, "resolved_id": str|None, "context": str, "pat_name": str}

    Note: we deduplicate on (token, resolved_id) within a single record's text,
    so that header repetitions (e.g. MDR numbers stamped on every page) do not
    inflate counts. The returned list has at most one entry per unique citation target.
    """
    results = []
    seen: set[tuple] = set()  # (token, resolved_id or token) already emitted
    rid = rec["id"]

    def _emit(token, resolved_id, context, pat_name):
        dedup_key = (token, resolved_id if resolved_id else token)
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        results.append({
            "token": token,
            "resolved_id": resolved_id,
            "context": context,
            "pat_name": pat_name,
        })

    # -- Incident numbers --
    for m in PATTERNS["incident"].finditer(text):
        raw_nums = m.group(1)
        ctx = _context(text, m)
        # May be a comma list "106, 107a"
        for part in re.split(r"[,\s]+", raw_nums):
            part = part.strip()
            if not part:
                continue
            token = f"Incident #{part}"
            # Try to resolve numeric part
            num_m = re.match(r"(\d+)", part)
            resolved = None
            if num_m:
                try:
                    resolved = lookups["incident_fn"](int(num_m.group(1)))
                except Exception:
                    pass
            _emit(token, resolved, ctx, "incident")

    # -- Blue Book case numbers --
    for m in PATTERNS["blue_book_case"].finditer(text):
        num_str = m.group(1)
        try:
            num = int(num_str)
        except ValueError:
            continue
        token = f"Case #{num}"
        resolved = lookups["case_num"].get(num)
        ctx = _context(text, m)
        _emit(token, resolved, ctx, "blue_book_case")

    # -- MDR numbers --
    for m in PATTERNS["mdr"].finditer(text):
        mdr_num = m.group(1)
        token = f"MDR {mdr_num}"
        ctx = _context(text, m)
        resolved_list = lookups["mdr"].get(mdr_num, [])
        # Remove self-reference
        resolved_list = [r for r in resolved_list if r != rid]
        if resolved_list:
            for res_id in resolved_list:
                _emit(token, res_id, ctx, "mdr")
        else:
            _emit(token, None, ctx, "mdr")

    # -- Diplomatic cable references --
    for m in PATTERNS["cable_ref"].finditer(text):
        station = m.group(1)
        num = m.group(2)
        if station not in KNOWN_CABLE_STATIONS:
            continue
        # Filter out obvious false positives (too-short stations that are common words)
        if station in {"INFO", "FROM", "INTO", "COPY", "PAGE", "TAGS", "SNIS"}:
            continue
        token = f"{station} {num}"
        ctx = _context(text, m)
        resolved = lookups["cable_station"].get((station, num))
        _emit(token, resolved, ctx, "cable_ref")

    # -- Serial references (only for records that belong to a series) --
    series_m = re.match(r"^(.+?)_(Serial|Section)_[0-9]+$", rid, re.IGNORECASE)
    if series_m:
        prefix = series_m.group(1)
        for m in PATTERNS["serial_ref"].finditer(text):
            num_str = m.group(1)
            try:
                num = int(num_str)
            except ValueError:
                continue
            token = f"Serial {num}"
            ctx = _context(text, m)
            resolved = lookups["series_member"].get((prefix, "serial", num))
            if resolved == rid:
                resolved = None
            _emit(token, resolved, ctx, "serial_ref")

        for m in PATTERNS["section_ref"].finditer(text):
            num_str = m.group(1)
            try:
                num = int(num_str)
            except ValueError:
                continue
            # Only emit if the target section actually exists in the corpus
            target = lookups["series_member"].get((prefix, "section", num))
            if target is None or target == rid:
                continue
            token = f"Section {num}"
            ctx = _context(text, m)
            _emit(token, target, ctx, "section_ref")

    # -- FOIPA numbers (dangling by nature — no FOIPA index in corpus) --
    for m in PATTERNS["foipa"].finditer(text):
        token = f"FOIPA#{m.group(1)}"
        ctx = _context(text, m)
        _emit(token, None, ctx, "foipa")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading search index…")
    records = json.loads(SEARCH_INDEX.read_text())
    print(f"  {len(records)} records")

    print("Building lookup tables…")
    lookups = build_lookup_tables(records)
    bb_cases = len(lookups["case_num"])
    incident_ranges = len([r for r in records
                           if re.search(r"Incident_Summaries", r["id"], re.IGNORECASE)])
    print(f"  {bb_cases} blue-book case numbers indexed")
    print(f"  {incident_ranges} incident-range records")
    print(f"  {len(lookups['mdr'])} MDR numbers indexed")
    print(f"  {len(lookups['series_member'])} series-member records")
    print(f"  {len(lookups['cable_station'])} cable station refs")

    # Edge accumulator: (from_id, to_id) -> {count, via, contexts}
    edge_acc: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "via": "", "contexts": []})
    # Dangling: (from_id, token) -> {count, contexts}
    dang_acc: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "contexts": []})

    print("Extracting citations…")
    records_with_text = 0
    for rec in records:
        rid = rec["id"]
        text = _load_text(rec)
        if not text.strip():
            continue
        records_with_text += 1

        citations = extract_citations_from_text(rec, text, lookups)
        for c in citations:
            if c["resolved_id"] and c["resolved_id"] != rid:
                key = (rid, c["resolved_id"])
                acc = edge_acc[key]
                acc["count"] += 1
                acc["via"] = c["token"]  # last token (may overwrite, but stable)
                if len(acc["contexts"]) < 3:
                    acc["contexts"].append(c["context"])
            else:
                key = (rid, c["token"])
                acc = dang_acc[key]
                acc["count"] += 1
                if len(acc["contexts"]) < 3:
                    acc["contexts"].append(c["context"])

    print(f"  Scanned {records_with_text} records with text")

    # Build edges list
    edges = []
    for (from_id, to_id), acc in edge_acc.items():
        edges.append({
            "from": from_id,
            "to": to_id,
            "via": acc["via"],
            "count": acc["count"],
            "contexts": acc["contexts"],
        })

    # Build dangling list
    dangling = []
    for (from_id, token), acc in dang_acc.items():
        dangling.append({
            "from": from_id,
            "token": token,
            "count": acc["count"],
            "contexts": acc["contexts"],
        })

    # Build by_record index (bidirectional)
    by_record: dict[str, dict] = {r["id"]: {"out": [], "in": []} for r in records}
    for e in edges:
        if e["to"] not in by_record[e["from"]]["out"]:
            by_record[e["from"]]["out"].append(e["to"])
        if e["from"] not in by_record[e["to"]]["in"]:
            by_record[e["to"]]["in"].append(e["from"])

    # Prune empty by_record entries to keep the file small
    by_record = {k: v for k, v in by_record.items() if v["out"] or v["in"]}

    out = {
        "edges": edges,
        "dangling": dangling,
        "by_record": by_record,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_PATH}")

    # ---------------------------------------------------------------------------
    # Smoke-test summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Total resolved edges : {len(edges)}")
    print(f"Total dangling refs  : {len(dangling)}")
    if edges:
        resolution_pct = 100 * len(edges) / max(len(edges) + len(dangling), 1)
        print(f"Resolution rate      : {resolution_pct:.1f}%")

    # Top-5 most cited records (most in-edges)
    cited_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        cited_counts[e["to"]] += e["count"]
    top_cited = sorted(cited_counts.items(), key=lambda x: -x[1])[:5]
    print()
    print("Top-5 most-cited records:")
    for rid, cnt in top_cited:
        print(f"  [{cnt:3d}x] {rid}")

    # Top-5 most-citing records (most out-edges)
    citing_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        citing_counts[e["from"]] += e["count"]
    top_citing = sorted(citing_counts.items(), key=lambda x: -x[1])[:5]
    print()
    print("Top-5 most-citing records:")
    for rid, cnt in top_citing:
        print(f"  [{cnt:3d}x] {rid}")

    # Sanity-check 3 random edges
    print()
    print("Sample edges (random 3):")
    sample = random.sample(edges, min(3, len(edges)))
    for e in sample:
        ctx = e["contexts"][0] if e["contexts"] else "(no context)"
        print(f"  FROM : {e['from']}")
        print(f"  TO   : {e['to']}")
        print(f"  VIA  : {e['via']}  (x{e['count']})")
        print(f"  CTX  : {ctx[:120]}")
        print()


if __name__ == "__main__":
    main()

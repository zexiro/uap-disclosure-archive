#!/usr/bin/env python3
"""Auto-extract Named Entities from OCR'd PDF text using spaCy en_core_web_sm.

Outputs:
  ui/entities.json — all entities, redacted fingerprints, by-record index

Run:
  python3 scripts/extract_entities.py

First-run auto-downloads the model if missing:
  python -m spacy download en_core_web_sm
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
TEXT_DIR = RAW / "text"
RECORDS_PATH = RAW / "records.json"
OUTPUT_PATH = ROOT / "ui" / "entities.json"

# ─── spaCy model loading with auto-download ───────────────────────────────────

SPACY_MODEL = "en_core_web_sm"

def load_nlp():
    try:
        import spacy
        try:
            return spacy.load(SPACY_MODEL)
        except OSError:
            print(f"[NER] Model '{SPACY_MODEL}' not found — downloading…", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--break-system-packages",
                 f"https://github.com/explosion/spacy-models/releases/download/"
                 f"{SPACY_MODEL}-3.8.0/{SPACY_MODEL}-3.8.0-py3-none-any.whl"],
                check=True,
            )
            return spacy.load(SPACY_MODEL)
    except ImportError:
        print("[NER] spaCy not installed — run: pip install spacy", file=sys.stderr)
        sys.exit(1)


# ─── Filter lists ─────────────────────────────────────────────────────────────

# Single-token false positives to drop (case-insensitive)
FALSE_POSITIVE_TOKENS = {
    "page", "figure", "figure.", "fig", "fig.", "table", "exhibit", "exhibit.",
    "attachment", "enclosure", "encl", "section", "paragraph", "para",
    "appendix", "annex", "item", "note", "ref", "reference", "tab",
    "volume", "vol", "part", "doc", "document", "see", "ibid",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "u.s.", "u.s", "us", "american", "american.", "united", "states",
}

# Minimum character length for an entity surface form
MIN_SURFACE_LEN = 3

# Max text bytes fed to spaCy per chunk (model limits; ~1MB is safe)
CHUNK_SIZE = 80_000

# Types we want from spaCy
WANTED_SPACY_LABELS = {"PERSON", "ORG", "GPE", "DATE", "LAW", "EVENT", "NORP"}

# ─── Redaction fingerprint patterns ──────────────────────────────────────────

# Match sequences of redaction glyphs or bracketed words
_REDACT_RE = re.compile(
    r"""
    (?:
        \[REDACTED\]        |   # bracketed word
        \[DELETED\]         |
        \[b\]\s*\(1\)       |   # FOIA exemption inline
        (?:█+)              |   # filled block characters
        (?:x{4,})           |   # xxxx strings
        (?:_{4,})           |   # ____ underscores
        (?:\*{4,})          |   # **** asterisks
        (?:\(\([^)]*\)\))       # ((blacked out)) style
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Also catch preceding honorifics attached to redaction blocks
_HONORIFIC_REDACT_RE = re.compile(
    r"""
    (?:Mr\.|Mrs\.|Ms\.|Dr\.|Col\.|Capt\.|Gen\.|Lt\.|Sgt\.|Maj\.)\s*
    (?:█+|\[REDACTED\]|\[DELETED\]|x{4,}|_{4,}|\*{4,})
    """,
    re.VERBOSE | re.IGNORECASE,
)

CONTEXT_WINDOW = 50  # chars of context on each side


def length_class(text: str) -> str:
    n = len(text)
    if n <= 6:
        return "short"
    if n <= 14:
        return "medium"
    return "long"


def extract_redacted_fingerprints(text: str, record_id: str) -> list[dict]:
    """Find redacted blocks in text, return fingerprint dicts."""
    results = []
    seen_sigs: set[str] = set()

    for m in _REDACT_RE.finditer(text):
        block = m.group()
        lc = length_class(block)
        # context
        s = max(0, m.start() - CONTEXT_WINDOW)
        e = min(len(text), m.end() + CONTEXT_WINDOW)
        ctx = text[s:e].replace("\n", " ").replace("  ", " ").strip()
        # short hash of block length + surrounding context for cross-ref
        sig_key = f"{lc}:{len(block)}:{ctx[:20]}"
        h = hashlib.sha256(sig_key.encode()).hexdigest()[:4]
        fp_id = f"redacted:{lc}:{h}"
        if fp_id not in seen_sigs:
            seen_sigs.add(fp_id)
            results.append({
                "id": fp_id,
                "length_class": lc,
                "block_len": len(block),
                "context_sample": ctx,
                "record_id": record_id,
            })

    # Also check honorific + redaction
    for m in _HONORIFIC_REDACT_RE.finditer(text):
        block = m.group()
        lc = length_class(block)
        s = max(0, m.start() - CONTEXT_WINDOW)
        e = min(len(text), m.end() + CONTEXT_WINDOW)
        ctx = text[s:e].replace("\n", " ").replace("  ", " ").strip()
        sig_key = f"honorific:{lc}:{len(block)}:{ctx[:20]}"
        h = hashlib.sha256(sig_key.encode()).hexdigest()[:4]
        fp_id = f"redacted:{lc}:{h}"
        if fp_id not in seen_sigs:
            seen_sigs.add(fp_id)
            results.append({
                "id": fp_id,
                "length_class": lc,
                "block_len": len(block),
                "context_sample": ctx,
                "record_id": record_id,
            })

    return results


# ─── Codename heuristic ───────────────────────────────────────────────────────

# ─── Codename heuristic ───────────────────────────────────────────────────────
# Strategy: only trust two signals for codenames —
#   1. Explicit PREFIX (PROJECT/OPERATION/PROGRAM/OP) + title-case or all-caps
#      word(s) that are not boilerplate.
#   2. A curated seed list of historically documented UAP-related codenames.
# Standalone all-caps phrase detection produces too many false positives from
# government document boilerplate (headers, routing slips, etc.) to be useful
# without a much larger training set, so we drop it.

# Prefixed codenames: PROJECT BLUE BOOK, OPERATION MOON DUST, etc.
# The prefix is group 1; the name phrase is group 2.
_PREFIXED_CODENAME_RE = re.compile(
    r"\b(PROJECT|OPERATION|PROGRAM|OP(?:ERATION)?)\s+"
    r"([A-Z][A-Z0-9]*(?:[ \t]+[A-Z][A-Z0-9]*){0,3})",
)

# Boilerplate words that are NOT valid codename words even after a prefix
_CODENAME_STOP_WORDS = {
    # classification
    "SECRET", "TOP", "CONFIDENTIAL", "UNCLASSIFIED", "DECLASSIFIED",
    "CLASSIFIED", "RESTRICTED", "FOUO",
    # structure
    "SUMMARY", "SUBJECT", "REPORT", "FILE", "FILES", "RECORD", "RECORDS",
    "ANNEX", "APPENDIX", "ATTACHMENT", "ENCLOSURE", "REFERENCE", "SECTION",
    # generic org words
    "OFFICE", "DEPARTMENT", "DIVISION", "BRANCH", "UNIT", "GROUP",
    "COMMAND", "STAFF", "BUREAU", "AGENCY", "ADMINISTRATION",
    # common function / auxiliary words that appear in all-caps
    "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "HAVE", "BEEN",
    "WAS", "WERE", "ARE", "HAS", "HAD", "WILL", "WOULD", "COULD", "SHOULD",
    "NOT", "BUT", "WHO", "WHY", "WHEN", "WHERE", "WHAT", "HOW",
    "ITS", "THEIR", "THEM", "THEY", "HIS", "HER", "HIM",
    "ALL", "ANY", "EACH", "EVERY", "SOME", "NONE", "BOTH",
    # location/direction words that read as generic context, not codename names
    "AREA", "ZONE", "REGION", "SITE", "BASE", "FACILITY",
    "NORTH", "SOUTH", "EAST", "WEST", "CENTRAL",
    # abbreviation-like words that aren't codename components
    "IVO", "IYO", "AOR",  # military jargon: "In Vicinity Of", "Area of Responsibility"
    # agencies
    "FBI", "CIA", "NSA", "DIA", "USAF", "USMC", "DOD", "DOE", "NATO",
}

# Hard exclusions — exact prefix+name combos that are not real codenames
_CODENAME_PHRASE_EXCLUSIONS = {
    "PROJECT SIGN",  # actually IS real — remove if desired
    "PROGRAM OFFICE",
    "OPERATION SECURITY",
    "OPERATION RISK",
    "PROJECT MANAGEMENT",
    "PROJECT OFFICER",
    "PROJECT NUMBER",
    "PROJECT FILE",
    "PROGRAM REVIEW",
    "PROGRAM STATUS",
    "OPERATION ORDER",
}

# Known / historically documented UAP codenames (seed list; always accepted)
_KNOWN_CODENAMES = {
    "BLUE FLY",
    "MOON DUST",
    "PROJECT GRUDGE",
    "PROJECT SIGN",
    "PROJECT MOGUL",
    "PROJECT BLUE BOOK",
    "OPERATION AQUARIUS",
    "PROJECT AQUARIUS",
    "OPERATION MAJESTIC",
    "MAJESTIC 12",
    "PROJECT REDLIGHT",
    "OPERATION SNOWBIRD",
    "PROJECT POUNCE",
    "PROJECT TWINKLE",
    "OPERATION OUTPOST",
    "OPERATION PRESS",
    "HORSE COLLAR",
    "PROJECT HORSE COLLAR",
}


def is_valid_prefixed_codename(full_phrase: str, name_part: str) -> bool:
    """Validate a prefix+name codename candidate."""
    # Check whole-phrase exclusion
    if full_phrase in _CODENAME_PHRASE_EXCLUSIONS:
        return False
    # No newlines or tabs — they indicate line-breaks in OCR output crossing word boundaries
    if "\n" in full_phrase or "\t" in full_phrase:
        return False
    # No multiple consecutive spaces — indicates OCR table columns, not a codename
    if re.search(r"  +", full_phrase):
        return False
    words = name_part.strip().split()
    if not words:
        return False
    # Every word must be ≥ 3 chars (avoids "PROJECT UN", "OPERATION AT" etc.)
    if any(len(w) < 3 for w in words):
        return False
    # All words must be purely alphabetic (no digits or punctuation in name)
    if any(not w.isalpha() for w in words):
        return False
    # All words in stop list → boilerplate phrase
    if all(w in _CODENAME_STOP_WORDS for w in words):
        return False
    return True


def extract_codenames(text: str) -> list[str]:
    """Return deduplicated list of detected codenames from text."""
    found: set[str] = set()

    # Normalise text: collapse all whitespace to single space for matching
    # (OCR often inserts newlines mid-word/phrase)
    text_norm = re.sub(r"\s+", " ", text)
    text_upper = text_norm.upper()

    # 1. Always include any occurrence of known codenames (case-insensitive)
    for kn in _KNOWN_CODENAMES:
        if kn in text_upper:
            found.add(kn)

    # 2. Prefixed codenames — run on normalised (single-space) text
    for m in _PREFIXED_CODENAME_RE.finditer(text_norm):
        name_part = m.group(2).strip()
        full_phrase = m.group(0).strip()
        if is_valid_prefixed_codename(full_phrase.upper(), name_part.upper()):
            found.add(full_phrase)

    return list(found)


# ─── Entity normalisation ────────────────────────────────────────────────────

def normalise_entity(surface: str) -> str:
    """Return a stable lowercase key for dedup."""
    # Collapse whitespace, strip punctuation from edges
    s = re.sub(r"\s+", " ", surface).strip(" .,;:-")
    return s.lower()


def make_entity_id(ent_type: str, surface: str) -> str:
    """Make a stable slug-style ID."""
    slug = re.sub(r"[^a-z0-9]+", "_", normalise_entity(surface)).strip("_")
    return f"{ent_type.lower()}:{slug}"


def is_noise(surface: str) -> bool:
    """True if the surface form is likely a false positive."""
    s = surface.strip()
    # Contains newlines or tabs — spaCy sometimes spans across line breaks in bad OCR
    if "\n" in s or "\t" in s:
        return True
    # Pure numbers / dates (digit-heavy)
    if re.fullmatch(r"[\d\s,.\-/]+", s):
        return True
    # Too short
    if len(s) < MIN_SURFACE_LEN:
        return True
    # Single-token in exclusion list
    if normalise_entity(s) in FALSE_POSITIVE_TOKENS:
        return True
    # Mostly punctuation/special chars
    alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
    if alpha_ratio < 0.4:
        return True
    # OCR garbage: contains mostly non-ASCII (a sign of bad OCR output)
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii > len(s) * 0.3:
        return True
    # OCR garbage: contains parentheses, brackets, backslash — not valid in entity names
    if re.search(r"[()[\]\\|@#$%^&*~`]", s):
        return True
    # Looks like an OCR artefact: contains 2+ consecutive non-word chars
    if re.search(r"[^A-Za-z0-9\s\-'.,]{2,}", s):
        return True
    # All-uppercase single word ≤ 4 chars that isn't a useful abbreviation
    # (spaCy often tags routing codes, initials clusters as PERSON/ORG)
    words = s.split()
    if len(words) == 1 and s.isupper() and len(s) <= 4 and s not in {
        "FBI", "CIA", "NSA", "DIA", "NRO", "DHS", "FAA", "DOD",
        "NASA", "USAF", "NAVY", "NATO",
    }:
        return True
    return False


# ─── Main extraction loop ────────────────────────────────────────────────────

def text_for_record(rec: dict) -> str:
    """Load extracted text for a record (same logic as build_search_index)."""
    chunks = []
    for lp in rec.get("primary_local", []):
        if lp.endswith(".pdf"):
            txt = TEXT_DIR / (Path(lp).stem + ".txt")
            if txt.exists():
                chunks.append(txt.read_text(errors="replace"))
    return "\n\n".join(chunks)


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks spaCy can handle."""
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])
    return chunks


def process_record(nlp, rec: dict, smoke_limit: int | None = None) -> dict:
    """
    Run NER on a single record.
    Returns {
        "ner_entities": [(surface, label)],
        "codenames": [str],
        "redacted_fps": [fingerprint_dict],
    }
    """
    record_id = rec["id"]
    text = text_for_record(rec)
    if not text:
        return {"ner_entities": [], "codenames": [], "redacted_fps": []}

    # Redaction fingerprints
    fps = extract_redacted_fingerprints(text, record_id)

    # Codenames (whole-text scan, fast)
    codenames = list(set(extract_codenames(text)))

    # spaCy NER — chunked to stay within model limits
    ner_results: list[tuple[str, str]] = []
    for chunk in chunk_text(text):
        doc = nlp(chunk)
        for ent in doc.ents:
            if ent.label_ not in WANTED_SPACY_LABELS:
                continue
            surface = ent.text.strip()
            if is_noise(surface):
                continue
            ner_results.append((surface, ent.label_))

    return {
        "ner_entities": ner_results,
        "codenames": codenames,
        "redacted_fps": fps,
    }


def build_entities_json(
    records: list[dict],
    nlp,
    smoke: bool = False,
) -> dict:
    """
    Process all records and build the entities.json output.
    If smoke=True, process only 10 records.
    """
    if smoke:
        records = records[:10]
        print(f"[NER] SMOKE TEST — processing {len(records)} records only")
    else:
        print(f"[NER] Processing {len(records)} records…")

    # Accumulate across all records:
    #   entity_surface_by_key: normalised_key -> {label, Counter(surface)}
    #   entity_records:        normalised_key -> set(record_id)
    #   codename_counter:      normalised -> {surface, Counter, set(records)}
    #   fp_data:               fp_id -> {id, context_samples[], record_ids[]}
    #   by_record:             record_id -> [entity_id, ...]

    entity_surface_by_key: dict[str, dict] = {}
    entity_records: dict[str, set] = defaultdict(set)
    codename_data: dict[str, dict] = {}
    fp_data: dict[str, dict] = {}
    by_record: dict[str, list] = defaultdict(list)

    for i, rec in enumerate(records, 1):
        rid = rec["id"]
        result = process_record(nlp, rec)

        # ── NER entities ──────────────────────────────────────────────────
        for surface, label in result["ner_entities"]:
            key = normalise_entity(surface)
            if key not in entity_surface_by_key:
                entity_surface_by_key[key] = {"label": label, "surfaces": Counter()}
            entity_surface_by_key[key]["surfaces"][surface] += 1
            entity_records[key].add(rid)

        # ── Codenames ─────────────────────────────────────────────────────
        for cn in result["codenames"]:
            nk = normalise_entity(cn)
            if nk not in codename_data:
                codename_data[nk] = {"surfaces": Counter(), "record_ids": set()}
            codename_data[nk]["surfaces"][cn] += 1
            codename_data[nk]["record_ids"].add(rid)

        # ── Redacted fingerprints ─────────────────────────────────────────
        for fp in result["redacted_fps"]:
            fid = fp["id"]
            if fid not in fp_data:
                fp_data[fid] = {
                    "id": fid,
                    "length_class": fp["length_class"],
                    "context_samples": [],
                    "record_ids": [],
                }
            if len(fp_data[fid]["context_samples"]) < 3:
                fp_data[fid]["context_samples"].append(fp["context_sample"])
            if rid not in fp_data[fid]["record_ids"]:
                fp_data[fid]["record_ids"].append(rid)

        if i % 10 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] processed", flush=True)

    # ── Build entity list ──────────────────────────────────────────────────
    entities = []
    for key, data in entity_surface_by_key.items():
        # Canonical surface = most frequent capitalisation
        canonical = data["surfaces"].most_common(1)[0][0]
        label = data["label"]
        count = sum(data["surfaces"].values())
        ent_id = make_entity_id(label, canonical)
        rec_ids = sorted(entity_records[key])
        entities.append({
            "id": ent_id,
            "surface": canonical,
            "type": label,
            "count": count,
            "record_ids": rec_ids,
        })
        # Populate by_record
        for rid in rec_ids:
            by_record[rid].append(ent_id)

    # ── Build codename list ────────────────────────────────────────────────
    codenames_out = []
    for nk, data in codename_data.items():
        canonical = data["surfaces"].most_common(1)[0][0]
        count = sum(data["surfaces"].values())
        cn_id = make_entity_id("CODENAME", canonical)
        rec_ids = sorted(data["record_ids"])
        codenames_out.append({
            "id": cn_id,
            "surface": canonical,
            "type": "CODENAME",
            "count": count,
            "record_ids": rec_ids,
        })
        for rid in rec_ids:
            by_record[rid].append(cn_id)

    # ── Merge codenames into entities list for unified access ──────────────
    entities.extend(codenames_out)

    # Sort by count desc
    entities.sort(key=lambda e: -e["count"])

    redacted_fps = sorted(fp_data.values(), key=lambda f: -len(f["record_ids"]))

    # Deduplicate by_record entries
    by_record_clean = {rid: sorted(set(ents)) for rid, ents in by_record.items()}

    return {
        "entities": entities,
        "redacted_fingerprints": redacted_fps,
        "by_record": by_record_clean,
    }


# ─── Smoke test helpers ───────────────────────────────────────────────────────

def run_smoke_test(result: dict) -> None:
    """Print entity counts per type, top 5, sample fps and codenames."""
    entities = result["entities"]
    fps = result["redacted_fingerprints"]

    # Counts per type
    type_counts: Counter = Counter(e["type"] for e in entities)
    print("\n=== SMOKE TEST RESULTS ===")
    print("\nEntity counts per type:")
    for etype, cnt in sorted(type_counts.items()):
        print(f"  {etype}: {cnt}")

    print("\nTop-5 most cited entities (by occurrence count):")
    for e in entities[:5]:
        print(f"  [{e['type']}] {e['surface']!r}  count={e['count']}  records={len(e['record_ids'])}")

    print("\nSample redacted fingerprints:")
    for fp in fps[:5]:
        print(f"  {fp['id']}  records={fp['record_ids']}")
        for ctx in fp["context_samples"][:1]:
            print(f"    ctx: {ctx[:80]!r}…")

    print("\nSample codenames:")
    cns = [e for e in entities if e["type"] == "CODENAME"]
    for cn in cns[:5]:
        print(f"  {cn['surface']!r}  count={cn['count']}  records={len(cn['record_ids'])}")

    # Sanity-check 5 random entities
    import random
    sample = random.sample(entities, min(5, len(entities)))
    print("\nRandom sanity-check entities:")
    for e in sample:
        assert e.get("id"), f"Missing id: {e}"
        assert e.get("surface"), f"Missing surface: {e}"
        assert e.get("type"), f"Missing type: {e}"
        assert isinstance(e.get("record_ids"), list), f"record_ids not a list: {e}"
        print(f"  OK  {e['id']!r}  surface={e['surface']!r}")

    # Validate JSON round-trip
    json_str = json.dumps(result, ensure_ascii=False)
    parsed = json.loads(json_str)
    assert len(parsed["entities"]) == len(entities), "round-trip entity count mismatch"
    print("\nJSON round-trip: OK")
    print("=== SMOKE TEST PASSED ===\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract NER entities from OCR'd PDF text")
    parser.add_argument("--smoke", action="store_true",
                        help="Run on first 10 records only (smoke test)")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Output path for entities.json")
    args = parser.parse_args()

    records = json.loads(RECORDS_PATH.read_text())

    print(f"[NER] Loading spaCy model '{SPACY_MODEL}'…")
    nlp = load_nlp()
    # Increase max_length for large documents
    nlp.max_length = 2_000_000

    result = build_entities_json(records, nlp, smoke=args.smoke)

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[NER] Wrote {out_path} — "
          f"{len(result['entities'])} entities, "
          f"{len(result['redacted_fingerprints'])} redacted fingerprints, "
          f"{len(result['by_record'])} records indexed")

    if args.smoke:
        run_smoke_test(result)


if __name__ == "__main__":
    main()

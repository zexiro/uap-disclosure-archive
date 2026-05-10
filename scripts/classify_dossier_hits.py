#!/usr/bin/env python3
"""Use a cheap LLM (Claude Haiku 4.5) to decide whether each dossier keyword
hit actually refers to the dossier's theme, or is a false positive (e.g.
"occupants of the car" matched by `\\boccupant\\w*\\b` in a UAP file but
referring to vehicle occupants, not entities).

Each hit becomes {kw, pat, ctx, relevant, summary} after classification.

Caches by SHA-1 of (kw, ctx) into raw/dossier_classifications.json so
re-runs across the 6-hourly Railway refresh are essentially free — only
genuinely new (kw, ctx) pairs cost an API call.

Run AFTER build_search_index.py. Patches ui/search-index.json in place.
Gracefully no-ops with a warning if ANTHROPIC_API_KEY is unset (keeps the
keyword-only filter working as a fallback).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
INDEX = ROOT / "ui" / "search-index.json"
DOSSIERS_PATH = ROOT / "ui" / "dossiers.json"
CACHE = RAW / "dossier_classifications.json"

MODEL = "claude-haiku-4-5-20251001"


def hit_key(kw: str, ctx: str) -> str:
    h = hashlib.sha1()
    h.update(kw.encode("utf-8"))
    h.update(b"\x00")
    h.update(ctx.encode("utf-8"))
    return h.hexdigest()


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            print(f"[classify] WARNING: cache file at {CACHE} is corrupt, starting fresh", file=sys.stderr)
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True))


def build_dossier_lookup() -> dict[str, dict]:
    if not DOSSIERS_PATH.exists():
        return {}
    return {d["id"]: d for d in json.loads(DOSSIERS_PATH.read_text())}


def classify_one(client, dossier: dict, kw: str, ctx: str) -> dict:
    """Single classification call. Returns {relevant: bool, summary: str}."""
    label = dossier.get("label", dossier.get("id", "this dossier"))
    description = dossier.get("description", "")
    sys_msg = (
        "You classify whether a passage from a UAP/UFO archive document "
        f"actually refers to the theme of the '{label}' dossier"
        + (f" — {description}" if description else "")
        + ". A simple keyword match is not enough — you must read the surrounding "
        "context. For example, 'occupants of the car' matched the keyword 'occupant' "
        "but is NOT about UAP/alien occupants. Reply with JSON only — no prose."
    )
    user = (
        f"Keyword that matched: {kw!r}\n"
        f"Passage (with the keyword in context):\n{ctx!r}\n\n"
        "Reply with this exact JSON shape:\n"
        '{"relevant": true|false, "summary": "..."}\n\n'
        "summary: when relevant=true, write ONE concise sentence (<=20 words) "
        "describing what the passage actually says about the theme — be specific "
        "(who/what/era if available). When relevant=false, summary should be a "
        "very short note explaining why it's a false positive (<=15 words)."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=sys_msg,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    # Be defensive: model may wrap JSON in code fences or trailing prose
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        # Last-ditch: extract first {...} block
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        out = json.loads(m.group())
    return {
        "relevant": bool(out.get("relevant")),
        "summary": str(out.get("summary", ""))[:240],
    }


def main() -> int:
    if not INDEX.exists():
        print(f"[classify] no {INDEX} — run build_search_index.py first", file=sys.stderr)
        return 1
    docs = json.loads(INDEX.read_text())
    dossiers = build_dossier_lookup()
    cache = load_cache()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        except ImportError:
            print("[classify] anthropic SDK not installed — skipping classification", file=sys.stderr)

    new_calls = 0
    cache_hits = 0
    failures = 0
    classified_total = 0

    for d in docs:
        hits_by_dossier = d.get("dossier_hits") or {}
        if not hits_by_dossier:
            continue
        for did, hits in hits_by_dossier.items():
            dossier = dossiers.get(did, {"id": did})
            for h in hits:
                key = hit_key(h["kw"], h["ctx"])
                if key in cache:
                    cached = cache[key]
                    h["relevant"] = bool(cached.get("relevant"))
                    h["summary"] = cached.get("summary", "")
                    cache_hits += 1
                    classified_total += 1
                    continue
                # Seed the cache from any pre-existing classifications already
                # baked into search-index.json (e.g. from a commit produced by
                # a local run). This avoids re-paying for the same call after
                # a fresh deploy where raw/ volume starts empty.
                if "relevant" in h:
                    cache[key] = {"relevant": bool(h["relevant"]), "summary": h.get("summary", ""), "kw": h["kw"], "ctx": h["ctx"]}
                    classified_total += 1
                    continue
                if client is None:
                    # No key/SDK — leave hit unclassified; UI treats undefined as relevant.
                    continue
                try:
                    verdict = classify_one(client, dossier, h["kw"], h["ctx"])
                    cache[key] = {**verdict, "kw": h["kw"], "ctx": h["ctx"]}
                    h["relevant"] = verdict["relevant"]
                    h["summary"] = verdict["summary"]
                    new_calls += 1
                    classified_total += 1
                    if new_calls % 10 == 0:
                        save_cache(cache)
                        print(f"[classify] checkpoint at {new_calls} new classifications", file=sys.stderr)
                except Exception as e:
                    failures += 1
                    print(f"[classify] FAIL kw={h['kw']!r}: {e}", file=sys.stderr)

    # Final cache save + index patch
    if new_calls or failures == 0:
        save_cache(cache)
    INDEX.write_text(json.dumps(docs, ensure_ascii=False))
    skipped = sum(
        1 for d in docs for hits in (d.get("dossier_hits") or {}).values()
        for h in hits if "relevant" not in h
    )
    print(f"[classify] cached={cache_hits} new_calls={new_calls} failed={failures} unclassified={skipped} classified_total={classified_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""ASGI server (uvicorn + starlette) for the Disclosure Archive.

Run via:
    uvicorn scripts.serve:app --host 0.0.0.0 --port $PORT --workers 1

Routes:
  GET  /                    → 302 /ui/
  GET  /healthz             → 200 ok
  GET  /api/ai/status       → JSON capability flags
  POST /api/ask             → SSE streaming RAG answer (OpenRouter)
  POST /api/enrich          → SSE streaming enrichment (Anthropic + web_search)
  POST /api/enrich/decide   → approve/reject a candidate enrichment claim
  GET  /api/enrich/all      → approved claims across all entities
  GET  /api/enrich/get/{kind}/{id} → raw enrichment store for one entity
  GET  /raw/*               → static files, long cache (immutable media)
  GET  /ui/*                → static files, short cache (changing UI assets)
"""
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
INCIDENTS_PATH = ROOT / "ui" / "incidents.json"
EMBED_PATH = ROOT / "ui" / "embeddings.npz"
ENRICH_DIR = ROOT / "ui" / "enrichments"
ENV_PATH = ROOT / ".env"
CORRECTIONS_DIR = ROOT / "vault" / "corrections"
CORRECTIONS_JSONL = CORRECTIONS_DIR / "corrections.jsonl"
ENRICH_DIR.mkdir(exist_ok=True)

# Minimal .env loader so ANTHROPIC_API_KEY can be persisted in the project root
# without depending on a shell startup file. Lines like KEY=value or KEY="value".
if ENV_PATH.exists():
    for _line in ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _v = _v.strip().strip('"').strip("'")
        os.environ.setdefault(_k.strip(), _v)

import threading

# Lazily loaded so a missing/old index doesn't break the static server.
_CORPUS = None
_INCIDENTS = None
_CORPUS_LOCK = threading.Lock()

# Embeddings (loaded once at startup if available).
_EMB_IDS = None        # np.ndarray[str], shape (N,)
_EMB_MATRIX = None     # np.ndarray[float32], shape (N, D), L2-normalized
_EMB_ID_TO_IDX = None  # dict[str, int]
_EMBEDDER = None       # fastembed.TextEmbedding, lazily initialized
_EMB_LOCK = threading.Lock()


def load_corpus():
    global _CORPUS, _INCIDENTS
    with _CORPUS_LOCK:
        if _CORPUS is None:
            try:
                _CORPUS = json.loads(INDEX_PATH.read_text())
            except Exception as e:
                print(f"[ask] failed to load index: {e}", file=sys.stderr)
                _CORPUS = []
            try:
                inc = json.loads(INCIDENTS_PATH.read_text())
                _INCIDENTS = inc.get("incidents", inc)
            except Exception:
                _INCIDENTS = {}
    return _CORPUS, _INCIDENTS


def load_embeddings():
    """Load the precomputed corpus embeddings into module globals (idempotent)."""
    global _EMB_IDS, _EMB_MATRIX, _EMB_ID_TO_IDX
    if _EMB_MATRIX is not None:
        return True
    if not EMBED_PATH.exists():
        return False
    try:
        import numpy as np
        npz = np.load(EMBED_PATH, allow_pickle=True)
        _EMB_IDS = npz["ids"]
        _EMB_MATRIX = npz["vectors"].astype("float32")
        _EMB_ID_TO_IDX = {str(_EMB_IDS[i]): i for i in range(len(_EMB_IDS))}
        print(f"[ask] embeddings loaded: {_EMB_MATRIX.shape}", flush=True)
        return True
    except Exception as e:
        print(f"[ask] embeddings load failed: {e}", file=sys.stderr)
        return False


def get_query_embedder():
    """Lazily construct the fastembed model (slow first call, fast after)."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMB_LOCK:
        if _EMBEDDER is None:
            try:
                from fastembed import TextEmbedding
                _EMBEDDER = TextEmbedding()
                print(f"[ask] query embedder loaded: {_EMBEDDER.model_name}", flush=True)
            except Exception as e:
                print(f"[ask] embedder unavailable: {e}", file=sys.stderr)
                _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


# ─── Retrieval ────────────────────────────────────────────────────────
_STOP = set("a an the of for and or but if to in on at by from with as is are was were be been being this that these those it its they them their he she his her i you we our".split())

def _tokens(s):
    return [t for t in re.findall(r"[a-zA-Z0-9_-]{2,}", (s or "").lower()) if t not in _STOP]

def _lexical_score(question, doc, qtoks, qset):
    """Cheap lexical signal used as a tiebreaker / boost over the dense score."""
    title = doc.get("title") or ""
    blurb = doc.get("blurb") or ""
    text = (doc.get("text") or "")[:30000]
    title_t = _tokens(title)
    blurb_t = _tokens(blurb)
    text_set = set(_tokens(text))
    title_hits = sum(1 for t in qtoks if t in title_t)
    blurb_hits = sum(1 for t in qtoks if t in blurb_t)
    text_hits = sum(1 for t in qset if t in text_set)
    phrase_bonus = 0
    if len(qtoks) >= 2:
        text_lower = text.lower()
        for a, b in zip(qtoks, qtoks[1:]):
            if f"{a} {b}" in text_lower:
                phrase_bonus += 2
    return title_hits * 6 + blurb_hits * 3 + text_hits * 1 + phrase_bonus


def retrieve(question, scope=None, k=8):
    corpus, _ = load_corpus()
    if not corpus:
        return []
    qtoks = _tokens(question)
    qset = set(qtoks)

    # Fast lookup of doc-by-id for the dense path.
    by_id = {d.get("id"): d for d in corpus if d.get("type") != "IMG"}

    # ─── Dense (semantic) retrieval if embeddings are available ───────
    dense_scores = {}     # id -> cosine sim
    if load_embeddings():
        embedder = get_query_embedder()
        if embedder is not None:
            try:
                import numpy as np
                qv = list(embedder.embed([question]))[0]
                qv = np.asarray(qv, dtype="float32")
                n = float(np.linalg.norm(qv)) or 1.0
                qv = qv / n
                sims = _EMB_MATRIX @ qv  # cosine since both normalized
                # Build candidate set: top 32 by sim, then re-rank with lexical bias.
                top_idx = np.argpartition(-sims, min(32, len(sims) - 1))[:32]
                for i in top_idx:
                    dense_scores[str(_EMB_IDS[i])] = float(sims[i])
            except Exception as e:
                print(f"[ask] dense retrieval failed, falling back to lexical: {e}", file=sys.stderr)

    # ─── Combine: dense + lexical ────────────────────────────────────
    candidates = set(dense_scores.keys())
    # Always consider docs that have any lexical match too (catches exact matches
    # for IDs/titles that the embedding may rank lower).
    if qtoks:
        for d in by_id.values():
            if scope and d.get("id") not in scope:
                continue
            if any(t in (d.get("title") or "").lower() for t in qtoks):
                candidates.add(d.get("id"))

    scored = []
    for did in candidates:
        d = by_id.get(did)
        if not d:
            continue
        if scope and did not in scope:
            continue
        dense = dense_scores.get(did, 0.0)
        lex = _lexical_score(question, d, qtoks, qset)
        # Hybrid score: dense in [0,1] dominates ranking; lex acts as a soft bias.
        # Scale lex modestly so a strong dense match still wins overall.
        score = dense * 100.0 + lex * 0.6
        if d.get("incident_id"):
            score += 0.3
        scored.append((score, d, dense, lex))
    scored.sort(key=lambda x: -x[0])
    out = []
    for s, d, dense, lex in scored[:k]:
        out.append({"score": round(s, 3), "dense": round(dense, 4), "lex": lex, "doc": d})
    return out

def make_snippet(text, qtoks, max_chars=320):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    low = text.lower()
    pos = -1
    for t in qtoks:
        i = low.find(t)
        if i >= 0:
            pos = i
            break
    if pos < 0:
        return text[:max_chars]
    start = max(0, pos - 80)
    end = min(len(text), start + max_chars)
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


# ─── Prompts ──────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are a research assistant for the Disclosure Archive, a corpus of declassified UAP "
    "documentation. Answer the user's question using ONLY the provided sources.\n\n"
    "CITATION RULES (strict — your output is post-processed):\n"
    "  1. Place a citation IMMEDIATELY after the clause or sentence that asserts the claim. "
    "Example: 'The disc was metallic [^2] and hovered for 12 minutes [^5].'\n"
    "  2. Use AT MOST 2 sources per citation cluster. NEVER write [^1][^2][^3][^4]. "
    "If a claim is supported by many sources, pick the 2 strongest.\n"
    "  3. Do NOT pile all citations at the end of a sentence or paragraph. Each "
    "independent claim gets its own citation in-line where it is made.\n"
    "  4. Do not invent source numbers. Only use [^N] where N is between 1 and the number "
    "of sources provided. Bare assertions without a citation will be flagged.\n\n"
    "OTHER RULES:\n"
    "  - If the sources don't contain enough to answer, say so directly. Do not speculate.\n"
    "  - Distinguish primary sources (government documents) from secondary (testimony, journalism).\n"
    "  - When sources contradict each other, surface the contradiction explicitly.\n"
    "  - Be terse: a tight paragraph is usually right.\n\n"
    "AFTER YOUR ANSWER, on a new line, output EXACTLY these two blocks in this order "
    "(no extra prose):\n\n"
    "<evidence>\n"
    "[^N]: \"verbatim quote from source N's excerpt that justifies your use of [^N]\"\n"
    "...one line per UNIQUE citation number you used in the answer...\n"
    "</evidence>\n\n"
    "<follow_ups>\n"
    "1. <first follow-up question>\n"
    "2. <second follow-up question>\n"
    "3. <third follow-up question>\n"
    "</follow_ups>\n\n"
    "Evidence rules: the quote must appear VERBATIM (exact substring) in the source's "
    "Summary or Excerpt — no paraphrasing. Pick the most representative span. Keep it "
    "under 200 characters. If you cannot find a verbatim span, write the line as "
    "[^N]: \"\" so the UI can show 'no quotable evidence'.\n\n"
    "Follow-up rules: answerable from this corpus, ends with '?', under 12 words. "
    "They render as clickable buttons."
)

MODE_SUFFIX = {
    "researcher": "",
    "skeptic": (
        " Apply skeptical priors. Flag prosaic explanations (balloons, satellites, sensor "
        "artifacts, misidentification) where plausible. Note when claims rest on single-witness "
        "testimony vs multi-source corroboration."
    ),
    "believer": (
        " Apply credulous priors. Take witness testimony at face value where uncontradicted. "
        "Note where official explanations have been disputed or revised."
    ),
}

def build_prompt(question, sources, mode):
    src_blocks = []
    for i, s in enumerate(sources, 1):
        d = s["doc"]
        title = d.get("title") or d.get("id")
        agency = d.get("agency") or ""
        rel = d.get("release_date") or ""
        snippet = make_snippet(d.get("text") or "", _tokens(question), max_chars=900)
        blurb = d.get("blurb") or ""
        src_blocks.append(
            f"[Source {i}] id={d.get('id')} | {agency} | released {rel}\n"
            f"Title: {title}\n"
            f"Summary: {blurb}\n"
            f"Excerpt: {snippet}\n"
        )
    sources_text = "\n".join(src_blocks)
    user = (
        f"QUESTION: {question}\n\n"
        f"SOURCES (cite as [^1], [^2], …):\n\n{sources_text}\n\n"
        f"Answer:"
    )
    return user


# ─── Rate limit ───────────────────────────────────────────────────────
_RATE_LOCK = threading.Lock()
_BUCKETS = defaultdict(deque)            # ip -> deque[ts] (hourly)
_GLOBAL_BUCKET = deque()                 # global deque[ts] (daily cap)
_RATE_PUBLIC   = int(os.environ.get("ASK_RATE_PUBLIC", "10"))
_RATE_OWNER    = int(os.environ.get("ASK_RATE_OWNER",  "60"))
_GLOBAL_DAILY  = int(os.environ.get("ASK_GLOBAL_DAILY_CAP", "500"))

def rate_check(ip, is_owner=False):
    """Returns (ok, retry_seconds, reason)."""
    now = time.time()
    cutoff = now - 3600
    cutoff_day = now - 86400
    limit = _RATE_OWNER if is_owner else _RATE_PUBLIC
    with _RATE_LOCK:
        # Global daily cap (defends the API budget against any single user / wave)
        while _GLOBAL_BUCKET and _GLOBAL_BUCKET[0] < cutoff_day:
            _GLOBAL_BUCKET.popleft()
        if not is_owner and len(_GLOBAL_BUCKET) >= _GLOBAL_DAILY:
            reset = int(_GLOBAL_BUCKET[0] + 86400 - now)
            return False, reset, "global_daily"
        # Per-IP hourly
        q = _BUCKETS[ip]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            reset = int(q[0] + 3600 - now)
            return False, reset, "hourly"
        q.append(now)
        _GLOBAL_BUCKET.append(now)
        return True, 0, ""


# ─── Corrections store ────────────────────────────────────────────────
CORRECTION_FIELDS = frozenset(
    ("title", "date", "location", "summary", "transcript", "tag", "link", "general")
)
CORRECTIONS_MAX_BODY = 8 * 1024
_VALID_IDS: set | None = None
_VALID_IDS_LOCK = threading.Lock()
_CORR_RATE_LOCK = threading.Lock()
_CORR_BUCKETS: defaultdict = defaultdict(deque)
_CORR_RATE_LIMIT = int(os.environ.get("CORRECTIONS_RATE_HOURLY", "5"))
_CORRECTIONS_WRITE_LOCK = threading.Lock()


def _load_valid_ids() -> set:
    global _VALID_IDS
    if _VALID_IDS is not None:
        return _VALID_IDS
    with _VALID_IDS_LOCK:
        if _VALID_IDS is None:
            try:
                index = json.loads(INDEX_PATH.read_text())
                _VALID_IDS = {d.get("id") for d in index if d.get("id")}
            except Exception as e:
                print(f"[corrections] failed to load search index: {e}", file=sys.stderr)
                _VALID_IDS = set()
    return _VALID_IDS


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def _corr_rate_check(ip: str):
    now = time.time()
    cutoff = now - 3600
    with _CORR_RATE_LOCK:
        q = _CORR_BUCKETS[ip]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= _CORR_RATE_LIMIT:
            reset = int(q[0] + 3600 - now)
            return False, max(reset, 1)
        q.append(now)
    return True, 0


def _append_correction(record: dict) -> None:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _CORRECTIONS_WRITE_LOCK:
        with CORRECTIONS_JSONL.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _read_corrections(record_id: str) -> list:
    if not CORRECTIONS_JSONL.exists():
        return []
    out = []
    try:
        with CORRECTIONS_JSONL.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("record_id") == record_id:
                    out.append({
                        "id": obj.get("id"),
                        "field": obj.get("field"),
                        "current_value": obj.get("current_value", ""),
                        "suggested_value": obj.get("suggested_value"),
                        "rationale": obj.get("rationale", ""),
                        "submitter_handle": obj.get("submitter_handle", ""),
                        "submitted_at": obj.get("submitted_at"),
                        "status": obj.get("status", "pending"),
                    })
    except Exception as e:
        print(f"[corrections] read error: {e}", file=sys.stderr)
    return out


# ─── AI gating ────────────────────────────────────────────────────────
def _is_localhost(ip):
    return ip in ("127.0.0.1", "::1", "localhost")

def _bool_env(name):
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")

def _get_client_ip(request):
    """Honour XFF when present (single hop)."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def _is_owner_request(request):
    """Localhost OR matching ASK_OWNER_TOKEN (header or ?owner=)."""
    ip = _get_client_ip(request)
    if _is_localhost(ip):
        return True
    token = (os.environ.get("ASK_OWNER_TOKEN") or "").strip()
    if not token:
        return False
    hdr = request.headers.get("x-ask-token", "").strip()
    if hdr and hdr == token:
        return True
    owner_qs = request.query_params.get("owner", "")
    if owner_qs and owner_qs == token:
        return True
    return False

def ai_status_for_request(request):
    """Client-facing capability flags."""
    owner = _is_owner_request(request)
    ask_enabled = _bool_env("ASK_ENABLED") or owner
    enrich_enabled = (_bool_env("ENRICH_ENABLED") and owner)
    return {
        "ask_enabled": bool(ask_enabled),
        "enrich_enabled": bool(enrich_enabled),
        "owner": bool(owner),
        "model": os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324"),
        "rate_public": _RATE_PUBLIC,
        "rate_owner": _RATE_OWNER,
    }


# ─── Citation guardrail ───────────────────────────────────────────────
_CITE_RE = re.compile(r"\[\^(\d+)\]")
def strip_unmatched_citations(text, valid_ns):
    """Remove [^N] markers whose N is not in valid_ns. Returns (cleaned, n_dropped)."""
    n_dropped = 0
    def sub(m):
        nonlocal n_dropped
        if int(m.group(1)) in valid_ns:
            return m.group(0)
        n_dropped += 1
        return ""
    out = _CITE_RE.sub(sub, text)
    return out, n_dropped


# ─── SSE helpers ──────────────────────────────────────────────────────
def sse(event, data):
    if event:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ─── Enrichment helpers (sync file I/O — fast, small JSON files) ──────
def _enrich_path(kind, eid):
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", eid)[:160]
    return ENRICH_DIR / f"{kind}_{safe}.json"

def _enrich_load(kind, eid):
    p = _enrich_path(kind, eid)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def _enrich_save(kind, eid, data):
    p = _enrich_path(kind, eid)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def _extract_json_block(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidate = m.group(1) if m else None
    if not candidate:
        m = re.search(r"\{[\s\S]*\}", text)
        candidate = m.group(0) if m else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def _enrich_existing_context(kind, eid, name):
    corpus, incidents = load_corpus()
    if kind == "incident":
        inc = (incidents or {}).get(eid)
        if not inc:
            return None
        related = [d for d in corpus if d.get("incident_id") == eid][:6]
        parts = [
            f"Name: {inc.get('name')}",
            f"Date: {inc.get('date','')}",
            f"Location: {inc.get('location','')}",
            f"Status: {inc.get('status','')}",
            f"Curated summary: {inc.get('summary','')}",
        ]
        if related:
            parts.append("Linked archive documents:")
            for d in related:
                parts.append(f"  - {d.get('title')} [{d.get('agency','')}]: {(d.get('blurb') or '')[:200]}")
        return "\n".join(parts)
    elif kind == "document":
        d = next((x for x in corpus if x.get("id") == eid), None)
        if not d:
            return None
        parts = [
            f"Title: {d.get('title')}",
            f"Agency: {d.get('agency','')}",
            f"Released: {d.get('release_date','')}",
            f"Incident date: {d.get('incident_date','')}",
            f"Location: {d.get('incident_location','')}",
            f"Summary: {d.get('blurb','')}",
            f"Excerpt: {(d.get('text') or '')[:1500]}",
        ]
        return "\n".join(parts)
    return None


# ─── ASGI app via Starlette ───────────────────────────────────────────
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    Response, JSONResponse, RedirectResponse, StreamingResponse, FileResponse
)
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect


LONG_CACHE_EXTS = {".pdf", ".mp4", ".jpg", ".jpeg", ".png", ".webm"}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Max-Age": "86400",
}


async def healthz(request: Request):
    return Response("ok", media_type="text/plain", headers={"Cache-Control": "no-store"})


async def root_redirect(request: Request):
    return RedirectResponse(url="/ui/", status_code=302)


async def ai_status(request: Request):
    s = ai_status_for_request(request)
    return JSONResponse(s, headers={"Cache-Control": "no-store", **CORS_HEADERS})


async def options_handler(request: Request):
    return Response(status_code=204, headers=CORS_HEADERS)


async def ask(request: Request):
    """POST /api/ask — streaming SSE RAG answer via OpenRouter."""
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=CORS_HEADERS)
    # Parse body
    try:
        body = await request.json()
    except Exception as e:
        return Response(f"bad request: {e}", status_code=400, media_type="text/plain")

    question = (body.get("question") or "").strip()
    mode = body.get("mode") or "researcher"
    if mode not in MODE_SUFFIX:
        mode = "researcher"
    scope = body.get("scope") or None
    # history reserved for future use
    # history = body.get("history") or []

    sse_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        **CORS_HEADERS,
    }

    async def stream_error(msg):
        yield sse("error", {"message": msg})

    if not question:
        return StreamingResponse(stream_error("empty question"), headers=sse_headers)

    status = ai_status_for_request(request)
    if not status["ask_enabled"]:
        return StreamingResponse(
            stream_error("AI is disabled on this server."), headers=sse_headers
        )

    ip = _get_client_ip(request)
    owner = status["owner"]
    ok, reset, reason = rate_check(ip, is_owner=owner)
    if not ok:
        tip = "rate limit reached" if reason == "hourly" else "the archive is busy — daily AI budget is exhausted"
        return StreamingResponse(
            stream_error(f"{tip} — try again in {reset // 60}m {reset % 60}s"),
            headers=sse_headers,
        )

    # Run CPU-bound retrieval in a thread so the event loop stays free.
    sources = await asyncio.get_event_loop().run_in_executor(
        None, lambda: retrieve(question, scope=scope, k=8)
    )

    async def generate():
        if not sources:
            yield sse("sources", {"sources": []})
            yield sse("token", {"text": (
                "I couldn't find any documents in this corpus that match your question. "
                "Try rephrasing with different keywords, or open the search view to "
                "browse the index directly."
            )})
            yield sse("done", {"dropped_citations": 0})
            return

        # Stream sources first so the user can read while answer streams.
        sources_payload = []
        for i, s in enumerate(sources, 1):
            d = s["doc"]
            sources_payload.append({
                "n": i,
                "id": d.get("id"),
                "title": d.get("title"),
                "agency": d.get("agency"),
                "release_date": d.get("release_date"),
                "incident_date": d.get("incident_date"),
                "thumb": d.get("thumb_small") or (
                    d["thumbnail_local"][0]
                    if isinstance(d.get("thumbnail_local"), list) and d["thumbnail_local"]
                    else None
                ),
                "blurb": (d.get("blurb") or "")[:280],
                "snippet": make_snippet(d.get("text") or "", _tokens(question), max_chars=240),
                "score": s["score"],
            })
        yield sse("sources", {"sources": sources_payload, "mode": mode})

        # Build prompt
        prompt = build_prompt(question, sources, mode)
        sys_prompt = SYSTEM_BASE + MODE_SUFFIX[mode]

        or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not or_key:
            yield sse("token", {"text": (
                "(server-side OPENROUTER_API_KEY is not set — showing retrieved sources only. "
                "Set the env var and restart the server to enable AI synthesis.)"
            )})
            yield sse("done", {"dropped_citations": 0})
            return

        try:
            import openai
        except Exception as e:
            yield sse("error", {"message": f"openai SDK missing: {e}"})
            return

        valid_ns = set(range(1, len(sources_payload) + 1))
        model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
        full_text = []
        carry = ""
        n_dropped = 0

        try:
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
                default_headers={
                    "HTTP-Referer": "https://uapdisclosuremirror.com/",
                    "X-Title": "Disclosure Archive - Ask",
                },
            )
            # Run the blocking OpenAI streaming call in a thread executor so we
            # don't block the event loop. We collect chunks via a queue.
            loop = asyncio.get_event_loop()
            chunk_queue: asyncio.Queue = asyncio.Queue()

            def _do_stream():
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        max_tokens=1024,
                        stream=True,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        text = getattr(delta, "content", None) or ""
                        if text:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, text)
                except Exception as exc:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, None)  # sentinel

            # Start the blocking stream in a thread
            fut = loop.run_in_executor(None, _do_stream)

            in_meta = False
            META_TAGS = ["<evidence>", "<follow_ups>"]

            while True:
                item = await chunk_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    yield sse("error", {"message": f"AI call failed: {item}"})
                    return

                text = item
                full_text.append(text)
                if in_meta:
                    continue

                buf = carry + text
                first_idx = -1
                for tag in META_TAGS:
                    idx = buf.find(tag)
                    if idx >= 0 and (first_idx < 0 or idx < first_idx):
                        first_idx = idx
                if first_idx >= 0:
                    emit, carry = buf[:first_idx], ""
                    in_meta = True
                else:
                    hold = 0
                    max_tag_len = max(len(t) for t in META_TAGS)
                    for k in range(min(len(buf), max_tag_len), 0, -1):
                        tail = buf[-k:]
                        if any(t.startswith(tail) for t in META_TAGS):
                            hold = k
                            break
                    if not hold:
                        m = re.search(r"\[\^?\d*$", buf)
                        if m:
                            hold = len(buf) - m.start()
                    if hold:
                        emit, carry = buf[:-hold], buf[-hold:]
                    else:
                        emit, carry = buf, ""

                if emit:
                    cleaned, dropped = strip_unmatched_citations(emit, valid_ns)
                    n_dropped += dropped
                    if cleaned:
                        yield sse("token", {"text": cleaned})

            await fut  # propagate any thread-side exception

            if carry and not in_meta:
                cleaned, dropped = strip_unmatched_citations(carry, valid_ns)
                n_dropped += dropped
                if cleaned:
                    yield sse("token", {"text": cleaned})

        except Exception as e:
            yield sse("error", {"message": f"AI call failed: {e}"})
            return

        joined = "".join(full_text)

        # Parse <evidence> block
        evidence = {}
        em = re.search(r"<evidence>([\s\S]*?)</evidence>", joined, re.I)
        if em:
            for line in em.group(1).strip().splitlines():
                lm = re.match(r"\s*\[\^?(\d+)\]\s*:\s*\"([^\"]*)\"", line)
                if lm:
                    n = int(lm.group(1))
                    if n in valid_ns:
                        evidence[str(n)] = lm.group(2).strip()

        # Parse follow-ups
        follow = []
        fm = re.search(r"<follow_ups>([\s\S]*?)</follow_ups>", joined, re.I)
        if fm:
            blob = fm.group(1)
        else:
            m2 = re.search(
                r"(?:follow[- ]?ups?|follow[- ]?up\s+questions?)\s*:?\s*\n+([\s\S]+?)$",
                joined, re.I
            )
            blob = m2.group(1) if m2 else ""
        if blob:
            for line in blob.strip().splitlines():
                q = line.strip().lstrip("-•*0123456789. )").strip().strip('"').strip("'")
                if q and len(q) >= 6 and (
                    "?" in q or q.lower().startswith(
                        ("what", "who", "when", "where", "why", "how", "which", "did", "does", "is", "are")
                    )
                ):
                    if not q.endswith("?"):
                        q = q + "?"
                    follow.append(q)
            follow = follow[:3]
        if not follow:
            tail = joined[-1500:]
            cands = re.findall(r"([A-Z][^.!?\n]{6,120}\?)", tail)
            follow = cands[-3:]

        yield sse("done", {"dropped_citations": n_dropped, "follow_ups": follow, "evidence": evidence})

    return StreamingResponse(generate(), headers=sse_headers)


async def enrich(request: Request):
    """POST /api/enrich — owner-only SSE streaming enrichment via Anthropic."""
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=CORS_HEADERS)
    sse_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        **CORS_HEADERS,
    }

    try:
        body = await request.json()
    except Exception as e:
        return Response(f"bad request: {e}", status_code=400, media_type="text/plain")

    async def generate():
        status = ai_status_for_request(request)
        if not status["enrich_enabled"]:
            yield sse("error", {"message": "Enrichment is owner-only and requires ENRICH_ENABLED=true."})
            return

        kind = body.get("kind") or ""
        eid = body.get("id") or ""
        name = body.get("name") or eid
        if kind not in ("incident", "document") or not eid:
            yield sse("error", {"message": "kind ('incident' or 'document') and id required"})
            return

        ctx = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _enrich_existing_context(kind, eid, name)
        )
        if not ctx:
            yield sse("error", {"message": "entity not found"})
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            yield sse("error", {"message": "ANTHROPIC_API_KEY not set on the server"})
            return
        try:
            import anthropic
        except Exception as e:
            yield sse("error", {"message": f"anthropic SDK missing: {e}"})
            return

        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ASK_MODEL", "claude-sonnet-4-5")

        run_id = f"r{int(time.time())}"
        yield sse("status", {"phase": "discover", "msg": f"Asking Claude to research {name}…"})

        # ─── Discovery pass ────────────────────────────────────────────
        discover_sys = (
            "You are a research assistant for the Disclosure Archive. Use web_search to find "
            "NEW factual information about the entity below that ISN'T already in the existing "
            "source material. Return AT MOST 5 candidate claims as a single JSON object. "
            "Every claim MUST cite at least one URL. Be skeptical: prefer reputable journalism, "
            "official government / NASA / DoD / AARO releases, and academic sources over forums."
        )
        discover_user = (
            f"ENTITY ({kind}): {name}\n\n"
            f"EXISTING SOURCE MATERIAL (do not re-discover):\n{ctx}\n\n"
            "Output strictly JSON, no prose:\n"
            "{\n  \"claims\": [\n    {\n"
            "      \"claim\": \"single-sentence factual claim\",\n"
            "      \"type\": \"corroborate|contradict|extend\",\n"
            "      \"supporting_urls\": [\"https://...\"],\n"
            "      \"confidence\": \"high|medium|low\",\n"
            "      \"notes\": \"one-line context\"\n"
            "    }\n  ]\n}"
        )

        loop = asyncio.get_event_loop()

        try:
            resp = await loop.run_in_executor(None, lambda: client.messages.create(
                model=model,
                max_tokens=2048,
                system=discover_sys,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
                messages=[{"role": "user", "content": discover_user}],
            ))
        except Exception as e:
            yield sse("error", {"message": f"discovery failed: {e}"})
            return

        text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        joined_disc = "\n".join(text_blocks)
        claims_raw = _extract_json_block(joined_disc) or {}
        claims = (claims_raw.get("claims") or []) if isinstance(claims_raw, dict) else []
        yield sse("discovery", {"n_candidates": len(claims), "raw_text": joined_disc[:1500]})

        if not claims:
            yield sse("error", {"message": "Discovery returned no parseable claims. Raw text included above."})
            return

        # ─── Verification pass (per claim) ─────────────────────────────
        verified = []
        for i, c in enumerate(claims, 1):
            if not isinstance(c, dict):
                continue
            claim_text = (c.get("claim") or "").strip()
            if not claim_text:
                continue
            yield sse("status", {"phase": "verify", "msg": f"Verifying claim {i}/{len(claims)}…", "claim": claim_text})
            v_sys = (
                "Independently verify the user's claim using fresh web_search queries. "
                "Find AT LEAST 2 reputable INDEPENDENT sources (different domains) that confirm "
                "or contradict it. Output strictly one JSON object."
            )
            v_user = (
                f"CLAIM TO VERIFY: {claim_text}\n"
                f"INITIAL CITED URLS: {json.dumps(c.get('supporting_urls') or [])}\n\n"
                "Also extract any date, named place, and lat/lng if the claim asserts them.\n"
                "Output strictly JSON, no prose:\n"
                "{\n"
                "  \"verdict\": \"verified|unverified|contradicted|partial\",\n"
                "  \"supporting_urls\": [\"https://...\"],\n"
                "  \"dissenting_urls\": [\"https://...\"],\n"
                "  \"date\": \"YYYY-MM-DD\" or null,\n"
                "  \"location\": \"place name\" or null,\n"
                "  \"geo\": [lat, lng] or null,\n"
                "  \"notes\": \"1-2 line explanation\"\n"
                "}"
            )
            try:
                vresp = await loop.run_in_executor(None, lambda: client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=v_sys,
                    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
                    messages=[{"role": "user", "content": v_user}],
                ))
                vtext = "\n".join(b.text for b in vresp.content if getattr(b, "type", None) == "text")
                vjson = _extract_json_block(vtext) or {}
            except Exception as e:
                vjson = {"verdict": "error", "notes": str(e), "supporting_urls": [], "dissenting_urls": []}

            geo = vjson.get("geo")
            if isinstance(geo, list) and len(geo) == 2:
                try:
                    geo = [float(geo[0]), float(geo[1])]
                except Exception:
                    geo = None
            else:
                geo = None

            merged = {
                "id": f"c{i}",
                "claim": claim_text,
                "type": c.get("type") or "extend",
                "confidence": c.get("confidence") or "medium",
                "discovery_urls": c.get("supporting_urls") or [],
                "discovery_notes": c.get("notes") or "",
                "verdict": (vjson.get("verdict") or "unverified").lower(),
                "supporting_urls": list(dict.fromkeys(
                    (c.get("supporting_urls") or []) + (vjson.get("supporting_urls") or [])
                )),
                "dissenting_urls": vjson.get("dissenting_urls") or [],
                "date": (vjson.get("date") or None),
                "location": (vjson.get("location") or None),
                "geo": geo,
                "verify_notes": vjson.get("notes") or "",
                "status": "candidate",
                "ts": int(time.time()),
            }
            verified.append(merged)
            yield sse("claim", {"claim": merged})

        # Persist (sync write is fine — small JSON, rare operation)
        store = _enrich_load(kind, eid)
        store.setdefault("kind", kind)
        store["id"] = eid
        store["name"] = name
        store.setdefault("runs", []).append({
            "run_id": run_id,
            "ts": int(time.time()),
            "model": model,
            "claims": verified,
        })
        _enrich_save(kind, eid, store)
        yield sse("done", {"run_id": run_id, "n_saved": len(verified)})

    return StreamingResponse(generate(), headers=sse_headers)


async def enrich_all(request: Request):
    """GET /api/enrich/all — approved claims across all entities."""
    out = []
    try:
        for p in sorted(ENRICH_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            kind = data.get("kind")
            eid = data.get("id")
            name = data.get("name")
            for run in data.get("runs", []):
                for c in run.get("claims", []):
                    if c.get("status") != "approved":
                        continue
                    out.append({
                        "kind": kind, "entity_id": eid, "entity_name": name,
                        "run_id": run.get("run_id"),
                        "claim_id": c.get("id"),
                        "claim": c.get("claim"),
                        "verdict": c.get("verdict"),
                        "date": c.get("date"),
                        "location": c.get("location"),
                        "geo": c.get("geo"),
                        "supporting_urls": c.get("supporting_urls") or [],
                    })
    except Exception as e:
        print(f"[enrich/all] {e}", file=sys.stderr)
    return JSONResponse(
        {"approved": out},
        headers={"Cache-Control": "no-store", **CORS_HEADERS},
    )


async def enrich_get(request: Request):
    """GET /api/enrich/get/{kind}/{id}"""
    kind = request.path_params.get("kind", "")
    eid = request.path_params.get("id", "")
    if kind not in ("incident", "document"):
        return Response("bad request", status_code=400)
    data = _enrich_load(kind, eid)
    return JSONResponse(
        data,
        headers={"Cache-Control": "no-store", **CORS_HEADERS},
    )


async def enrich_decide(request: Request):
    """POST /api/enrich/decide — approve or reject a candidate claim."""
    if not _is_owner_request(request):
        return Response("enrichment owner-gated", status_code=403)
    try:
        body = await request.json()
    except Exception:
        return Response("bad request", status_code=400)

    kind = body.get("kind")
    eid = body.get("id")
    run_id = body.get("run_id")
    claim_id = body.get("claim_id")
    decision = body.get("decision")
    if decision not in ("approved", "rejected"):
        return Response("bad request", status_code=400)

    store = _enrich_load(kind, eid)
    for run in store.get("runs", []):
        if run.get("run_id") != run_id:
            continue
        for c in run.get("claims", []):
            if c.get("id") == claim_id:
                c["status"] = decision
                c["decided_at"] = int(time.time())
                break
    _enrich_save(kind, eid, store)
    return JSONResponse({"ok": True}, headers=CORS_HEADERS)


# ─── Corrections endpoints ────────────────────────────────────────────
async def corrections_post(request: Request):
    """POST /api/corrections — submit a user correction/annotation for a record."""
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=CORS_HEADERS)

    ip = _get_client_ip(request)

    body_bytes = await request.body()
    if len(body_bytes) > CORRECTIONS_MAX_BODY:
        return JSONResponse({"error": "payload too large (max 8 KB)"}, status_code=413, headers=CORS_HEADERS)
    try:
        body = json.loads(body_bytes or b"{}")
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400, headers=CORS_HEADERS)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400, headers=CORS_HEADERS)

    record_id = (body.get("record_id") or "").strip()
    field = (body.get("field") or "").strip()
    current_value = str(body.get("current_value") or "").strip()
    suggested_value = str(body.get("suggested_value") or "").strip()
    rationale = str(body.get("rationale") or "").strip()
    submitter_handle = str(body.get("submitter_handle") or "").strip()

    if not record_id:
        return JSONResponse({"error": "record_id is required"}, status_code=400, headers=CORS_HEADERS)
    if not field:
        return JSONResponse({"error": "field is required"}, status_code=400, headers=CORS_HEADERS)
    if field not in CORRECTION_FIELDS:
        return JSONResponse(
            {"error": f"field must be one of: {', '.join(sorted(CORRECTION_FIELDS))}"},
            status_code=400, headers=CORS_HEADERS,
        )
    if not suggested_value:
        return JSONResponse({"error": "suggested_value is required and must not be empty"}, status_code=400, headers=CORS_HEADERS)
    if len(rationale) > 2000:
        return JSONResponse({"error": "rationale exceeds 2000 character limit"}, status_code=400, headers=CORS_HEADERS)
    if len(submitter_handle) > 64:
        return JSONResponse({"error": "submitter_handle exceeds 64 character limit"}, status_code=400, headers=CORS_HEADERS)

    valid_ids = _load_valid_ids()
    if record_id not in valid_ids:
        return JSONResponse({"error": f"record_id '{record_id}' not found in archive index"}, status_code=404, headers=CORS_HEADERS)

    ok, retry_after = _corr_rate_check(ip)
    if not ok:
        return JSONResponse(
            {"error": f"rate limit exceeded — max {_CORR_RATE_LIMIT} corrections per IP per hour"},
            status_code=429,
            headers={"Retry-After": str(retry_after), **CORS_HEADERS},
        )

    corr_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "id": corr_id,
        "record_id": record_id,
        "field": field,
        "current_value": current_value,
        "suggested_value": suggested_value,
        "rationale": rationale,
        "submitter_handle": submitter_handle,
        "submitted_at": now_iso,
        "status": "pending",
        "client_ip_hash": _hash_ip(ip),
    }
    try:
        await asyncio.get_event_loop().run_in_executor(None, _append_correction, record)
    except Exception as e:
        print(f"[corrections] write error: {e}", file=sys.stderr)
        return JSONResponse({"error": "failed to persist correction"}, status_code=500, headers=CORS_HEADERS)

    print(f"[corrections] new {corr_id} record={record_id!r} field={field!r} ip_hash={_hash_ip(ip)}", flush=True)
    return JSONResponse({"id": corr_id, "status": "received"}, status_code=201, headers=CORS_HEADERS)


async def corrections_get(request: Request):
    """GET /api/corrections/{record_id} — list corrections for a record."""
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=CORS_HEADERS)
    record_id = (request.path_params.get("record_id") or "").strip()
    if not record_id:
        return JSONResponse({"error": "record_id path segment is required"}, status_code=400, headers=CORS_HEADERS)
    valid_ids = _load_valid_ids()
    if record_id not in valid_ids:
        return JSONResponse({"error": f"record_id '{record_id}' not found in archive index"}, status_code=404, headers=CORS_HEADERS)
    corrections = await asyncio.get_event_loop().run_in_executor(None, _read_corrections, record_id)
    return JSONResponse(
        {"record_id": record_id, "corrections": corrections},
        headers={"Cache-Control": "no-store", **CORS_HEADERS},
    )


# ─── Static file handlers with correct cache headers ─────────────────
async def serve_raw(request: Request):
    """Serve /raw/* with long immutable cache for media files."""
    path_suffix = request.path_params.get("path", "")
    file_path = ROOT / "raw" / path_suffix
    if not file_path.exists() or not file_path.is_file():
        return Response("not found", status_code=404)
    # Prevent directory traversal
    try:
        file_path.resolve().relative_to((ROOT / "raw").resolve())
    except ValueError:
        return Response("forbidden", status_code=403)

    suffix = file_path.suffix.lower()
    if suffix in LONG_CACHE_EXTS:
        cache = "public, max-age=31536000, immutable"
    else:
        cache = "public, max-age=300"

    return FileResponse(
        file_path,
        headers={"Cache-Control": cache, "Access-Control-Allow-Origin": "*"},
    )


async def serve_ui(request: Request):
    """Serve /ui/* with short cache."""
    path_suffix = request.path_params.get("path", "")
    file_path = ROOT / "ui" / path_suffix
    if not file_path.exists() or not file_path.is_file():
        return Response("not found", status_code=404)
    try:
        file_path.resolve().relative_to((ROOT / "ui").resolve())
    except ValueError:
        return Response("forbidden", status_code=403)

    return FileResponse(
        file_path,
        headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"},
    )


# ─── Collab WebSocket hub ─────────────────────────────────────────────
# Ephemeral process-memory state. Drops on restart. Single-worker only.
class CollabHub:
    def __init__(self):
        self._sessions: dict[str, WebSocket] = {}
        self._ip_sessions: dict[str, set] = {}
        self._msg_ts: dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, session_id: str, ip: str) -> bool:
        async with self._lock:
            sessions_for_ip = self._ip_sessions.setdefault(ip, set())
            if len(sessions_for_ip) >= 3:
                await ws.close(code=1008, reason="too many connections from this IP")
                return False
            self._sessions[session_id] = ws
            sessions_for_ip.add(session_id)
            self._msg_ts[session_id] = deque()
        await self._broadcast({
            "type": "presence",
            "online": list(self._sessions.keys()),
            "joined": session_id,
        })
        return True

    async def disconnect(self, session_id: str, ip: str):
        async with self._lock:
            self._sessions.pop(session_id, None)
            self._msg_ts.pop(session_id, None)
            sessions_for_ip = self._ip_sessions.get(ip, set())
            sessions_for_ip.discard(session_id)
            if not sessions_for_ip:
                self._ip_sessions.pop(ip, None)
        await self._broadcast({
            "type": "presence",
            "online": list(self._sessions.keys()),
            "left": session_id,
        })

    async def handle_message(self, session_id: str, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return None
        now = time.time()
        cutoff = now - 60
        q = self._msg_ts.get(session_id)
        if q is None:
            return None
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= 30:
            retry = int(q[0] + 60 - now) + 1
            return {"type": "error", "code": "rate", "retry_after_s": retry}
        q.append(now)

        kind = msg.get("type")
        ts = datetime.now(timezone.utc).isoformat()
        if kind == "chat":
            text = (msg.get("text") or "")[:500]
            await self._broadcast({"type": "chat", "from": session_id, "text": text, "ts": ts})
        elif kind == "share":
            share_kind = msg.get("kind", "")
            if share_kind in ("search-query", "ask-query"):
                print(f"[collab] dropped share kind={share_kind!r} from {session_id}", file=sys.stderr)
                return None
            payload = msg.get("payload") or {}
            await self._broadcast({"type": "share", "from": session_id, "kind": share_kind, "payload": payload, "ts": ts})
        elif kind == "follow":
            await self._broadcast({"type": "follow", "follower": session_id, "target": msg.get("target", ""), "ts": ts})
        elif kind == "unfollow":
            await self._broadcast({"type": "follow", "follower": session_id, "target": None, "ts": ts})
        return None

    async def _broadcast(self, payload: dict):
        data = json.dumps(payload, ensure_ascii=False)
        dead = []
        for sid, ws in list(self._sessions.items()):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self._sessions.pop(sid, None)


_HUB = CollabHub()


async def ws_collab(ws: WebSocket):
    """WebSocket endpoint for /ws/collab — ephemeral collab surface."""
    session_id = ws.query_params.get("session", "") or "anon"
    session_id = re.sub(r"[^a-zA-Z0-9\-]", "", session_id)[:36] or "anon"
    ip = (
        (ws.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        or (ws.client.host if ws.client else "unknown")
    )
    await ws.accept()
    if not await _HUB.connect(ws, session_id, ip):
        return
    try:
        while True:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
            err = await _HUB.handle_message(session_id, data)
            if err:
                try:
                    await ws.send_text(json.dumps(err))
                except Exception:
                    pass
    finally:
        await _HUB.disconnect(session_id, ip)


# ─── Route table ─────────────────────────────────────────────────────
routes = [
    Route("/", endpoint=root_redirect, methods=["GET", "HEAD"]),
    Route("/healthz", endpoint=healthz, methods=["GET", "HEAD"]),
    Route("/healthz/", endpoint=healthz, methods=["GET", "HEAD"]),
    Route("/api/ai/status", endpoint=ai_status, methods=["GET", "OPTIONS"]),
    Route("/api/ask", endpoint=ask, methods=["POST", "OPTIONS"]),
    Route("/api/ask/", endpoint=ask, methods=["POST", "OPTIONS"]),
    Route("/api/enrich", endpoint=enrich, methods=["POST", "OPTIONS"]),
    Route("/api/enrich/", endpoint=enrich, methods=["POST", "OPTIONS"]),
    Route("/api/enrich/all", endpoint=enrich_all, methods=["GET", "OPTIONS"]),
    Route("/api/enrich/decide", endpoint=enrich_decide, methods=["POST", "OPTIONS"]),
    Route("/api/enrich/get/{kind}/{id:path}", endpoint=enrich_get, methods=["GET", "OPTIONS"]),
    Route("/api/corrections", endpoint=corrections_post, methods=["POST", "OPTIONS"]),
    Route("/api/corrections/", endpoint=corrections_post, methods=["POST", "OPTIONS"]),
    Route("/api/corrections/{record_id}", endpoint=corrections_get, methods=["GET", "OPTIONS"]),
    Route("/raw/{path:path}", endpoint=serve_raw, methods=["GET", "HEAD", "OPTIONS"]),
    Route("/ui/{path:path}", endpoint=serve_ui, methods=["GET", "HEAD", "OPTIONS"]),
    WebSocketRoute("/ws/collab", endpoint=ws_collab),
]

app = Starlette(routes=routes)


# ─── Legacy __main__ entrypoint (for local dev without uvicorn CLI) ───
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    addr = os.environ.get("BIND", "0.0.0.0")
    print(f"[http] serving on http://{addr}:{port}/", flush=True)
    uvicorn.run(app, host=addr, port=port, workers=1)

#!/usr/bin/env python3
"""Tiny static file server tuned for this project, plus a /api/ask
streaming endpoint that powers the ⌘K interrogator.

- Serves the entire repo root so /ui/, /raw/, /vault/ all resolve.
- Redirects "/" → "/ui/" so the search UI is the home page.
- Sets long cache headers on raw/* (immutable assets) and short on ui/* (changes often).
- Honours the PORT env var (Railway).
- POST /api/ask streams a Claude response over SSE, grounded in the
  search-index.json corpus with retrieval, citation enforcement, and a
  per-IP rate limit.
"""
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
INCIDENTS_PATH = ROOT / "ui" / "incidents.json"
EMBED_PATH = ROOT / "ui" / "embeddings.npz"
ENRICH_DIR = ROOT / "ui" / "enrichments"
ENV_PATH = ROOT / ".env"
ENRICH_DIR.mkdir(exist_ok=True)

# Minimal .env loader so ANTHROPIC_API_KEY can be persisted in the project root
# without depending on a shell startup file. Lines like KEY=value or KEY="value".
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)

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


# ─── AI gating: ask is public-grade, enrich is owner-only ────────────
def _is_localhost(ip):
    return ip in ("127.0.0.1", "::1", "localhost")

def _bool_env(name):
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")

def is_owner(handler):
    """Localhost OR matching ASK_OWNER_TOKEN (header or ?owner=)."""
    if _is_localhost(handler._client_ip()):
        return True
    token = (os.environ.get("ASK_OWNER_TOKEN") or "").strip()
    if not token:
        return False
    hdr = handler.headers.get("x-ask-token", "").strip()
    if hdr and hdr == token:
        return True
    qs = urlparse(handler.path).query
    if qs:
        from urllib.parse import parse_qs
        qp = parse_qs(qs)
        if qp.get("owner", [""])[0] == token:
            return True
    return False

def ai_status_for(handler):
    """Client-facing: {ask_enabled, enrich_enabled, owner, model, rate_public, rate_owner}."""
    owner = is_owner(handler)
    ask_enabled = _bool_env("ASK_ENABLED") or owner   # owner always enabled
    # Enrich requires explicit ENRICH_ENABLED, AND owner.
    enrich_enabled = (_bool_env("ENRICH_ENABLED") and owner)
    return {
        "ask_enabled": bool(ask_enabled),
        "enrich_enabled": bool(enrich_enabled),
        "owner": bool(owner),
        "model": os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324"),
        "rate_public": _RATE_PUBLIC,
        "rate_owner": _RATE_OWNER,
    }

def is_ask_authorized(handler):
    s = ai_status_for(handler)
    return s["ask_enabled"]

def is_enrich_authorized(handler):
    s = ai_status_for(handler)
    return s["enrich_enabled"]


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


# ─── Handler ──────────────────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):
    def _client_ip(self):
        # Honour XFF when present (single hop)
        xff = self.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
        return self.client_address[0]

    def _handle_special(self, write_body):
        # Liveness probe for Railway. Must stay cheap (no disk reads) so it
        # still answers when /raw/* image traffic is saturating worker threads.
        if self.path in ("/healthz", "/healthz/"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if write_body:
                self.wfile.write(body)
            return True
        if self.path in ("", "/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return True
        return False

    def do_GET(self):
        if self._handle_special(write_body=True):
            return
        path = urlparse(self.path).path
        if path == "/api/ai/status":
            return self._handle_ai_status()
        if path == "/api/enrich/all":
            return self._handle_enrich_all()
        if path.startswith("/api/enrich/get/"):
            return self._handle_enrich_get(path[len("/api/enrich/get/"):])
        return super().do_GET()

    def _handle_ai_status(self):
        s = ai_status_for(self)
        body = json.dumps(s).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        if self._handle_special(write_body=False):
            return
        return super().do_HEAD()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path in ("/api/ask", "/api/ask/"):
            return self._handle_ask()
        if path in ("/api/enrich", "/api/enrich/"):
            return self._handle_enrich()
        if path == "/api/enrich/decide":
            return self._handle_enrich_decide()
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _ssend(self, chunk):
        try:
            self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    def _handle_ask(self):
        # Parse body
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"bad request: " + str(e).encode())
            return

        question = (body.get("question") or "").strip()
        mode = body.get("mode") or "researcher"
        if mode not in MODE_SUFFIX:
            mode = "researcher"
        scope = body.get("scope") or None
        history = body.get("history") or []  # [{q, a}], unused for now but reserved

        if not question:
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "empty question"}))
            return

        if not is_ask_authorized(self):
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "AI is disabled on this server."}))
            return

        ip = self._client_ip()
        owner = is_owner(self)
        ok, reset, reason = rate_check(ip, is_owner=owner)
        if not ok:
            self._send_sse_headers()
            tip = "rate limit reached" if reason == "hourly" else "the archive is busy — daily AI budget is exhausted"
            self._ssend(sse("error", {"message": f"{tip} — try again in {reset // 60}m {reset % 60}s"}))
            return

        # Retrieve
        sources = retrieve(question, scope=scope, k=8)
        if not sources:
            self._send_sse_headers()
            self._ssend(sse("sources", {"sources": []}))
            self._ssend(sse("token", {"text": "I couldn't find any documents in this corpus that match your question. Try rephrasing with different keywords, or open the search view to browse the index directly."}))
            self._ssend(sse("done", {"dropped_citations": 0}))
            return

        # Stream sources first so the user can read them while the answer streams
        self._send_sse_headers()
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
                    d["thumbnail_local"][0] if isinstance(d.get("thumbnail_local"), list) and d["thumbnail_local"] else None
                ),
                "blurb": (d.get("blurb") or "")[:280],
                "snippet": make_snippet(d.get("text") or "", _tokens(question), max_chars=240),
                "score": s["score"],
            })
        if not self._ssend(sse("sources", {"sources": sources_payload, "mode": mode})):
            return

        # Build prompt
        prompt = build_prompt(question, sources, mode)
        sys_prompt = SYSTEM_BASE + MODE_SUFFIX[mode]

        # /api/ask uses OpenRouter (cheap public). /api/enrich uses Anthropic
        # directly because it needs the web_search tool.
        or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not or_key:
            self._ssend(sse("token", {"text": (
                "(server-side OPENROUTER_API_KEY is not set — showing retrieved sources only. "
                "Set the env var and restart serve.py to enable AI synthesis.)"
            )}))
            self._ssend(sse("done", {"dropped_citations": 0}))
            return
        try:
            import openai
        except Exception as e:
            self._ssend(sse("error", {"message": f"openai SDK missing: {e}"})); return

        valid_ns = set(range(1, len(sources_payload) + 1))
        model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
        full_text = []
        carry = ""
        n_dropped = 0
        try:
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=or_key,
                # Optional but polite — OpenRouter shows these in your dashboard.
                default_headers={
                    "HTTP-Referer": "https://uapdisclosuremirror.com/",
                    "X-Title": "Disclosure Archive - Ask",
                },
            )
            stream = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                stream=True,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            # Once any of the trailing meta-blocks starts, swallow everything until
            # end-of-stream (we still parse them from full_text afterwards).
            in_meta = False
            META_TAGS = ["<evidence>", "<follow_ups>"]
            for chunk in stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if not text: continue
                full_text.append(text)
                if in_meta:
                    continue
                buf = carry + text
                # If any meta tag appears, emit everything before it and lock down.
                first_idx = -1
                for tag in META_TAGS:
                    idx = buf.find(tag)
                    if idx >= 0 and (first_idx < 0 or idx < first_idx):
                        first_idx = idx
                if first_idx >= 0:
                    emit, carry = buf[:first_idx], ""
                    in_meta = True
                else:
                    # Hold back any tail that could be a partial meta tag OR a
                    # partial [^N] citation.
                    hold = 0
                    max_tag_len = max(len(t) for t in META_TAGS)
                    for k in range(min(len(buf), max_tag_len), 0, -1):
                        tail = buf[-k:]
                        if any(t.startswith(tail) for t in META_TAGS):
                            hold = k
                            break
                    if not hold:
                        m = re.search(r"\[\^?\d*$", buf)
                        if m: hold = len(buf) - m.start()
                    if hold:
                        emit, carry = buf[:-hold], buf[-hold:]
                    else:
                        emit, carry = buf, ""
                if emit:
                    cleaned, dropped = strip_unmatched_citations(emit, valid_ns)
                    n_dropped += dropped
                    if cleaned and not self._ssend(sse("token", {"text": cleaned})):
                        return
            if carry and not in_meta:
                cleaned, dropped = strip_unmatched_citations(carry, valid_ns)
                n_dropped += dropped
                if cleaned:
                    self._ssend(sse("token", {"text": cleaned}))
        except Exception as e:
            self._ssend(sse("error", {"message": f"AI call failed: {e}"})); return

        joined = "".join(full_text)

        # Parse <evidence> block: per-citation verbatim quote.
        evidence = {}
        em = re.search(r"<evidence>([\s\S]*?)</evidence>", joined, re.I)
        if em:
            for line in em.group(1).strip().splitlines():
                lm = re.match(r"\s*\[\^?(\d+)\]\s*:\s*\"([^\"]*)\"", line)
                if lm:
                    n = int(lm.group(1))
                    if n in valid_ns:
                        evidence[str(n)] = lm.group(2).strip()

        # Parse follow-ups from the assistant's emitted text. Try the structured
        # block first; fall back to grabbing any trailing question lines.
        follow = []
        m = re.search(r"<follow_ups>([\s\S]*?)</follow_ups>", joined, re.I)
        if m:
            blob = m.group(1)
        else:
            # Fallback 1: a fenced "Follow-up questions" header style
            m2 = re.search(r"(?:follow[- ]?ups?|follow[- ]?up\s+questions?)\s*:?\s*\n+([\s\S]+?)$",
                           joined, re.I)
            blob = m2.group(1) if m2 else ""
        if blob:
            for line in blob.strip().splitlines():
                q = line.strip().lstrip("-•*0123456789. )").strip().strip('"').strip("'")
                # accept anything that looks like a question
                if q and len(q) >= 6 and ("?" in q or q.lower().startswith(("what", "who", "when", "where", "why", "how", "which", "did", "does", "is", "are"))):
                    if not q.endswith("?"):
                        q = q + "?"
                    follow.append(q)
            follow = follow[:3]
        # Last-ditch fallback: take the last 3 question-shaped sentences in the answer body
        if not follow:
            tail = joined[-1500:]
            cands = re.findall(r"([A-Z][^.!?\n]{6,120}\?)", tail)
            follow = cands[-3:]
        self._ssend(sse("done", {"dropped_citations": n_dropped, "follow_ups": follow, "evidence": evidence}))

    # ───────────────────────────────────────────────────────────────────
    # /api/enrich — discover + verify new facts about an entity via web_search
    # ───────────────────────────────────────────────────────────────────
    def _handle_enrich(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            self.send_response(400); self.send_header("Content-Type","text/plain"); self.end_headers()
            self.wfile.write(b"bad request: " + str(e).encode()); return
        if not is_enrich_authorized(self):
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "Enrichment is owner-only and requires ENRICH_ENABLED=true."})); return

        kind = body.get("kind") or ""
        eid = body.get("id") or ""
        name = body.get("name") or eid
        if kind not in ("incident", "document") or not eid:
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "kind ('incident' or 'document') and id required"})); return

        # Build existing-context blurb so the model knows what NOT to re-discover.
        ctx = self._enrich_existing_context(kind, eid, name)
        if not ctx:
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "entity not found"})); return

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            self._send_sse_headers()
            self._ssend(sse("error", {"message": "ANTHROPIC_API_KEY not set on the server"})); return
        try:
            import anthropic
        except Exception as e:
            self._send_sse_headers()
            self._ssend(sse("error", {"message": f"anthropic SDK missing: {e}"})); return

        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ASK_MODEL", "claude-sonnet-4-5")

        self._send_sse_headers()
        run_id = f"r{int(time.time())}"
        self._ssend(sse("status", {"phase": "discover", "msg": f"Asking Claude to research {name}…"}))

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

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=discover_sys,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
                messages=[{"role": "user", "content": discover_user}],
            )
        except Exception as e:
            self._ssend(sse("error", {"message": f"discovery failed: {e}"})); return

        # Extract any text response and parse JSON
        text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        joined = "\n".join(text_blocks)
        claims_raw = self._extract_json_block(joined) or {}
        claims = (claims_raw.get("claims") or []) if isinstance(claims_raw, dict) else []
        self._ssend(sse("discovery", {"n_candidates": len(claims), "raw_text": joined[:1500]}))

        if not claims:
            self._ssend(sse("error", {"message": "Discovery returned no parseable claims. Raw text included above."}))
            return

        # ─── Verification pass (per claim) ─────────────────────────────
        verified = []
        for i, c in enumerate(claims, 1):
            if not isinstance(c, dict): continue
            claim_text = (c.get("claim") or "").strip()
            if not claim_text: continue
            self._ssend(sse("status", {"phase": "verify", "msg": f"Verifying claim {i}/{len(claims)}…", "claim": claim_text}))
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
                vresp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=v_sys,
                    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
                    messages=[{"role": "user", "content": v_user}],
                )
                vtext = "\n".join(b.text for b in vresp.content if getattr(b, "type", None) == "text")
                vjson = self._extract_json_block(vtext) or {}
            except Exception as e:
                vjson = {"verdict": "error", "notes": str(e), "supporting_urls": [], "dissenting_urls": []}
            geo = vjson.get("geo")
            if isinstance(geo, list) and len(geo) == 2:
                try: geo = [float(geo[0]), float(geo[1])]
                except Exception: geo = None
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
                "supporting_urls": list(dict.fromkeys((c.get("supporting_urls") or []) + (vjson.get("supporting_urls") or []))),
                "dissenting_urls": vjson.get("dissenting_urls") or [],
                "date": (vjson.get("date") or None),
                "location": (vjson.get("location") or None),
                "geo": geo,
                "verify_notes": vjson.get("notes") or "",
                "status": "candidate",
                "ts": int(time.time()),
            }
            verified.append(merged)
            self._ssend(sse("claim", {"claim": merged}))

        # Persist
        store = self._enrich_load(kind, eid)
        store.setdefault("kind", kind); store["id"] = eid; store["name"] = name
        store.setdefault("runs", []).append({
            "run_id": run_id,
            "ts": int(time.time()),
            "model": model,
            "claims": verified,
        })
        self._enrich_save(kind, eid, store)
        self._ssend(sse("done", {"run_id": run_id, "n_saved": len(verified)}))

    def _enrich_existing_context(self, kind, eid, name):
        corpus, incidents = load_corpus()
        if kind == "incident":
            inc = (incidents or {}).get(eid)
            if not inc: return None
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
            if not d: return None
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

    def _extract_json_block(self, text):
        if not text: return None
        # Try fenced ```json ... ``` first
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        candidate = m.group(1) if m else None
        if not candidate:
            # Otherwise try the first balanced { ... } chunk
            m = re.search(r"\{[\s\S]*\}", text)
            candidate = m.group(0) if m else None
        if not candidate: return None
        try:
            return json.loads(candidate)
        except Exception:
            # try to repair common issue: trailing commas
            cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try: return json.loads(cleaned)
            except Exception: return None

    def _enrich_path(self, kind, eid):
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", eid)[:160]
        return ENRICH_DIR / f"{kind}_{safe}.json"

    def _enrich_load(self, kind, eid):
        p = self._enrich_path(kind, eid)
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: return {}
        return {}

    def _enrich_save(self, kind, eid, data):
        p = self._enrich_path(kind, eid)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _handle_enrich_all(self):
        """Return ONLY approved claims across all entities — for graph/timeline/globe overlays."""
        out = []
        try:
            for p in sorted(ENRICH_DIR.glob("*.json")):
                try:
                    data = json.loads(p.read_text())
                except Exception:
                    continue
                kind = data.get("kind"); eid = data.get("id"); name = data.get("name")
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
        body = json.dumps({"approved": out}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_enrich_get(self, rest):
        # rest: "{kind}/{id}"
        try:
            kind, eid = rest.split("/", 1)
        except ValueError:
            self.send_response(400); self.end_headers(); return
        if kind not in ("incident", "document"):
            self.send_response(400); self.end_headers(); return
        data = self._enrich_load(kind, eid)
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_enrich_decide(self):
        if not is_enrich_authorized(self):
            self.send_response(403); self.end_headers()
            self.wfile.write(b"enrichment owner-gated"); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self.send_response(400); self.end_headers(); return
        kind = body.get("kind"); eid = body.get("id")
        run_id = body.get("run_id"); claim_id = body.get("claim_id")
        decision = body.get("decision")  # "approved" | "rejected"
        if decision not in ("approved", "rejected"):
            self.send_response(400); self.end_headers(); return
        store = self._enrich_load(kind, eid)
        for run in store.get("runs", []):
            if run.get("run_id") != run_id: continue
            for c in run.get("claims", []):
                if c.get("id") == claim_id:
                    c["status"] = decision
                    c["decided_at"] = int(time.time())
                    break
        self._enrich_save(kind, eid, store)
        body_out = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    def end_headers(self):
        # Range support is automatic in SimpleHTTPRequestHandler since 3.7
        if self.path in ("/healthz", "/healthz/"):
            pass  # /healthz already wrote its own headers
        elif self.path.startswith("/raw/") and any(
            self.path.endswith(ext) for ext in (".pdf", ".mp4", ".jpg", ".jpeg", ".png", ".webm")
        ):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        elif self.path.startswith("/ui/"):
            self.send_header("Cache-Control", "public, max-age=300")
        # CORS so iframes / clients work cleanly
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("[http] %s %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()


def main():
    port = int(os.environ.get("PORT", "8000"))
    addr = os.environ.get("BIND", "0.0.0.0")
    httpd = ThreadingHTTPServer((addr, port), Handler)
    httpd.daemon_threads = True
    print(f"[http] serving on http://{addr}:{port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

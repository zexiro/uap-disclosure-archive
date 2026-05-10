#!/usr/bin/env python3
"""Hourly war.gov change-watcher.

Catches drift in the public UFO release that the 6-hourly full refresh
would miss for up to 6 hours: a file silently re-uploaded, a PDF
re-released with new pages, an image swapped out, a record removed.

Algorithm
---------
1. Fetch the source CSV. Hash it. Compare to manifest.csv_sha256.
   If different → log "csv_changed" and mark a refresh as needed.
2. For every asset URL in raw/records.json:
   a. HEAD request → grab Content-Length, ETag, Last-Modified.
   b. If those agree with the manifest entry → discard (no work).
   c. Otherwise GET the body, sha256 it, compare to manifest.sha256.
      Same hash? Update manifest's headers, no change emitted.
      Different? Log "modified" with old/new hash.
3. For PDFs that byte-changed AND already have OCR sidecars in
   raw/text/, hash the existing text against manifest.ocr_sha256 — if
   the OCR is now stale, log "ocr_pending". (We don't re-OCR here; the
   triggered refresh.sh handles it. This watcher must stay fast.)
4. URLs missing from the new records.json but present in the manifest
   → log "removed". URLs in records.json but not in manifest → log
   "added". (CSV-level diff, but resolved per-URL so the log is
   actionable.)
5. Append every event to raw/wargov_changes.log (one JSON per line).
6. Write the new manifest. Drop a sentinel file
   raw/wargov_changes_pending if anything actually changed — the
   entrypoint loop reads this to decide whether to fire refresh.sh
   ahead of schedule.

We deliberately do NOT hash the war.gov/UFO/ landing page itself —
it's full of rotating __VIEWSTATE/cache-bust noise and would fire on
every run. Asset-level drift is the right signal.

The manifest is committed to git so a fresh container starts with full
state, instead of seeing every file as "new".
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RECORDS = RAW / "records.json"
MANIFEST = RAW / "wargov_manifest.json"
CHANGE_LOG = RAW / "wargov_changes.log"
PENDING = RAW / "wargov_changes_pending"

CSV_URL = "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv"

HTTP_TIMEOUT = 30

# war.gov sits behind Akamai, which 403s anything that doesn't look like a
# real browser (default urllib UA included). scripts/fetch.sh uses the same
# header set elsewhere — keep them in sync if Akamai's policy tightens.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Referer": "https://www.war.gov/UFO/",
}


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def normalize_url(url: str) -> str:
    """war.gov ships URLs with raw spaces and unicode (em-dash) in paths.
    urllib refuses to GET those — encode the path, leave host alone."""
    parts = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parts.path, safe="/%")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment))


def http_get(url: str) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(normalize_url(url), headers=BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, b"", {}
    except Exception:
        return 0, b"", {}


def http_head(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(normalize_url(url), headers=BROWSER_HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}




def headers_indicate_change(prev: dict, new_headers: dict) -> bool:
    """Cheap pre-check: any of ETag, Last-Modified, or Content-Length
    differs from what we last recorded? If yes, do a full GET."""
    if not prev:
        return True
    for k in ("etag", "last-modified", "content-length"):
        if prev.get(k, "") != new_headers.get(k, ""):
            return True
    return False


def asset_urls_from_records(records: list[dict]) -> set[str]:
    urls: set[str] = set()
    for r in records:
        for k in ("pdf_image_link", "modal_image"):
            v = (r.get(k) or "").strip()
            if v.startswith("http"):
                urls.add(v)
    return urls


def load_manifest() -> dict:
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        m.pop("page_sha256", None)  # tombstone — earlier versions tracked this
        return m
    return {"csv_sha256": "", "files": {}}


def write_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True, ensure_ascii=False))


def append_event(event: dict) -> None:
    CHANGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHANGE_LOG.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def ocr_text_hash_for(url: str) -> str:
    """For a PDF URL, look up its on-disk OCR sidecar and hash it.
    Returns "" if no sidecar yet (refresh.sh will populate it)."""
    if not url.lower().endswith(".pdf"):
        return ""
    stem = url.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    text_path = RAW / "text" / f"{stem}.txt"
    if not text_path.exists():
        return ""
    return sha256_bytes(text_path.read_bytes())


def check() -> int:
    now = dt.datetime.now(dt.UTC).isoformat()
    manifest = load_manifest()
    files = manifest.setdefault("files", {})
    changed_count = 0

    # 1. CSV
    print("[watch] fetching CSV…")
    status, body, _ = http_get(CSV_URL)
    if status == 200 and body:
        new_csv_hash = sha256_bytes(body)
        if manifest.get("csv_sha256") and new_csv_hash != manifest["csv_sha256"]:
            append_event({
                "ts": now, "kind": "csv_changed",
                "url": CSV_URL,
                "old_sha256": manifest["csv_sha256"], "new_sha256": new_csv_hash,
                "size": len(body),
            })
            changed_count += 1
        manifest["csv_sha256"] = new_csv_hash
    else:
        append_event({"ts": now, "kind": "fetch_error", "url": CSV_URL, "status": status})

    # 2. Per-asset HEAD/GET drift check, sourced from current records.json.
    if RECORDS.exists():
        records = json.loads(RECORDS.read_text())
        current_urls = asset_urls_from_records(records)
    else:
        records, current_urls = [], set()

    # adds: present now but not in manifest
    for url in current_urls - set(files.keys()):
        append_event({"ts": now, "kind": "added", "url": url})
        files[url] = {"first_seen": now, "last_seen": now}
        changed_count += 1

    # removed: in manifest but no longer in records.json
    for url in set(files.keys()) - current_urls:
        append_event({"ts": now, "kind": "removed", "url": url})
        del files[url]
        changed_count += 1

    print(f"[watch] checking {len(current_urls)} asset URLs for drift…")
    for url in sorted(current_urls):
        entry = files.setdefault(url, {"first_seen": now, "last_seen": now})

        # Cheap pre-check via HEAD.
        head_status, head_headers = http_head(url)
        prev_headers = entry.get("headers", {})

        if head_status == 200 and not headers_indicate_change(prev_headers, head_headers):
            entry["last_seen"] = now
            continue  # discard if match

        # Pre-check failed (or first time) — pull body and hash.
        get_status, body, headers = http_get(url)
        if get_status != 200 or not body:
            # Only log the error when the status changes (or first failure).
            # Some URLs are permanently broken (404 thumbnails, malformed paths
            # the war.gov template generated wrong); we don't want them
            # spamming the change log every hour.
            if entry.get("error_status") != get_status:
                append_event({"ts": now, "kind": "fetch_error", "url": url, "status": get_status})
                entry["error_status"] = get_status
            entry["last_seen"] = now
            continue
        # Clear any prior error state on first successful fetch.
        if "error_status" in entry:
            entry.pop("error_status", None)
        new_hash = sha256_bytes(body)
        old_hash = entry.get("sha256", "")

        if old_hash and new_hash != old_hash:
            ev = {
                "ts": now, "kind": "modified", "url": url,
                "old_sha256": old_hash, "new_sha256": new_hash,
                "size": len(body),
            }
            # OCR-text change detection — only meaningful for PDFs we already OCR'd.
            if url.lower().endswith(".pdf"):
                stale_ocr_hash = entry.get("ocr_sha256", "")
                fresh_ocr_hash = ocr_text_hash_for(url)
                if stale_ocr_hash and fresh_ocr_hash and stale_ocr_hash == fresh_ocr_hash:
                    # PDF bytes changed but the OCR sidecar still reflects the
                    # *old* file (refresh.sh hasn't re-run yet) — flag it so
                    # refresh.sh knows to re-OCR.
                    ev["ocr_pending"] = True
            append_event(ev)
            changed_count += 1

        entry["sha256"] = new_hash
        entry["size"] = len(body)
        entry["headers"] = {
            k: headers.get(k, "")
            for k in ("etag", "last-modified", "content-length", "content-type")
        }
        entry["last_seen"] = now
        if url.lower().endswith(".pdf"):
            entry["ocr_sha256"] = ocr_text_hash_for(url)

    write_manifest(manifest)

    if changed_count > 0:
        PENDING.write_text(now)
        print(f"[watch] {changed_count} change(s) detected — pending sentinel written")
    else:
        if PENDING.exists():
            PENDING.unlink()
        print("[watch] no changes (discarded all matches)")

    return 0


if __name__ == "__main__":
    sys.exit(check())

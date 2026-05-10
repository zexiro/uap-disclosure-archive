#!/usr/bin/env python3
"""Pull recent posts from UAP/UFO subreddits into the unified schema.

Uses Reddit's public .json endpoint — no auth needed for read access at
modest volumes. Each subreddit gives us back the most recent ~1000 posts
(Reddit caps deep paging there, even with paginating tokens).

Reddit r/UFOs encourages a templated post body:

    Time: 9 pm, 2016 maybe
    Location: Florida

…so when present we extract those into structured fields. When absent we
just keep the title + selftext. Either way provenance is "media_unverified"
and the caller's correlation pass treats them as low-trust hearsay.

Output:
  raw/sightings/reddit/raw/<sub>.json     (raw API responses, paginated)
  raw/sightings/reddit/sightings.json     (normalized)
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sightings.schema import PROVENANCE_FOR_SOURCE, verification_for  # noqa: E402

OUT_DIR = ROOT / "raw" / "sightings" / "reddit"
RAW_DIR = OUT_DIR / "raw"
NORMALIZED = OUT_DIR / "sightings.json"

SUBREDDITS = ["UFOs", "UFOB", "HighStrangeness"]
PER_PAGE = 100
MAX_PAGES = 10  # 10 * 100 = 1000 (Reddit's hard cap on deep paging)
USER_AGENT = "disclosure-archive/1.0 (multi-source UAP aggregator)"

# Loose templated-field extractors — many r/UFOs posts begin with these.
LOC_RE = re.compile(r"^\s*Location\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
DATE_RE = re.compile(r"^\s*(?:Date|Time|When)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def fetch_subreddit(sub: str) -> list[dict]:
    """Returns list of `data` dicts (one per post) across paginated pages.
    Caches the raw JSON pages on disk so reruns are free until you delete."""
    raw_path = RAW_DIR / f"{sub}.json"
    if raw_path.exists():
        print(f"[reddit] cached {sub}: {raw_path.relative_to(ROOT)}")
        return json.loads(raw_path.read_text())

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    posts: list[dict] = []
    after = None
    for page in range(MAX_PAGES):
        params = {"limit": PER_PAGE}
        if after:
            params["after"] = after
        url = f"https://www.reddit.com/r/{sub}/new.json?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"[reddit] {sub} page {page}: {e}")
            break
        children = data.get("data", {}).get("children", [])
        if not children:
            break
        for c in children:
            posts.append(c.get("data", {}))
        after = data.get("data", {}).get("after")
        if not after:
            break
        time.sleep(1.2)  # Reddit's loose rate limit; 1 req/sec is safe.
        print(f"[reddit] {sub} fetched page {page+1} ({len(posts)} cumulative)")
    raw_path.write_text(json.dumps(posts, ensure_ascii=False))
    print(f"[reddit] wrote {len(posts)} posts → {raw_path.relative_to(ROOT)}")
    return posts


def extract_templated(selftext: str) -> tuple[str, str]:
    if not selftext:
        return "", ""
    head = selftext[:1500]  # template fields, when present, are at the top
    loc = ""
    date = ""
    m = LOC_RE.search(head)
    if m:
        loc = m.group(1).strip()
    m = DATE_RE.search(head)
    if m:
        date = m.group(1).strip()
    return loc, date


def normalize_post(p: dict, fetched_at: str) -> dict | None:
    pid = p.get("id")
    if not pid:
        return None
    title = (p.get("title") or "").strip()
    selftext = (p.get("selftext") or "").strip()
    if not title:
        return None

    created = p.get("created_utc")
    occurred_at = ""
    precision = "unknown"
    if created:
        occurred_at = dt.datetime.utcfromtimestamp(int(created)).isoformat()
        precision = "minute"  # the post timestamp; the *event* timestamp is usually fuzzier

    loc_text, date_text = extract_templated(selftext)
    location: dict = {}
    if loc_text:
        location["name"] = loc_text

    sub = p.get("subreddit") or "UFOs"
    permalink = p.get("permalink") or ""
    source_url = f"https://www.reddit.com{permalink}" if permalink else (p.get("url") or "")

    prov = PROVENANCE_FOR_SOURCE["reddit"]
    rec = {
        "id": f"reddit:{sub}:{pid}",
        "source": "reddit",
        "source_id": f"{sub}:{pid}",
        "source_url": source_url,
        "fetched_at": fetched_at,
        "provenance": prov,
        "verification_status": verification_for(prov),
        "title": title[:240],
        "summary": selftext[:280],
        "text": selftext[:4000],
        "occurred_at": occurred_at,           # post creation time
        "occurred_at_precision": precision,
        "location": location,
        "media": [],
        "raw": {
            "subreddit": sub,
            "score": p.get("score", 0),
            "num_comments": p.get("num_comments", 0),
            "author": p.get("author", ""),
            "templated_date": date_text,      # what the OP wrote — a freeform string
            "is_video": bool(p.get("is_video")),
        },
    }
    return {k: v for k, v in rec.items() if v not in ("", [], {}, 0)}


def main() -> int:
    fetched_at = dt.date.today().isoformat()
    sightings: list[dict] = []
    for sub in SUBREDDITS:
        for post in fetch_subreddit(sub):
            rec = normalize_post(post, fetched_at)
            if rec:
                sightings.append(rec)

    NORMALIZED.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED.open("w") as f:
        json.dump(sightings, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[reddit] normalized {len(sightings)} posts → {NORMALIZED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

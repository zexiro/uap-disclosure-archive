#!/usr/bin/env python3
"""Download every PDF, image, and DVIDS video referenced by records.json.

Idempotent: skips any file that already exists on disk with non-zero size.
Concurrent (6 workers) but polite. Logs failures to raw/download_errors.log.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def encode_url(url: str) -> str:
    """Percent-encode the path segments of a URL (war.gov leaves spaces unencoded)."""
    parts = urllib.parse.urlsplit(url)
    quoted_path = urllib.parse.quote(parts.path, safe="/+")
    quoted_query = urllib.parse.quote(parts.query, safe="=&%")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, quoted_path, quoted_query, parts.fragment))

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
RECORDS = json.loads((RAW / "records.json").read_text())
ERR_LOG = RAW / "download_errors.log"
META_DIR = RAW / "dvids_meta"
META_DIR.mkdir(exist_ok=True)
(RAW / "videos").mkdir(exist_ok=True)
(RAW / "docs").mkdir(exist_ok=True)
(RAW / "images").mkdir(exist_ok=True)

DVIDS_API_KEY = "key-68bb60d16b35e"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def fetch(url: str, dest: Path, headers=None, timeout=120) -> tuple[str, int]:
    """Download url -> dest. Returns (status, bytes)."""
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", dest.stat().st_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Referer": "https://www.war.gov/UFO/",
    }
    if headers:
        h.update(headers)
    req = urllib.request.Request(encode_url(url), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        tmp.rename(dest)
        return ("ok", dest.stat().st_size)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        with ERR_LOG.open("a") as f:
            f.write(f"{url}\t{dest}\t{e}\n")
        return ("err", 0)


def fetch_dvids_meta(video_id: str) -> dict | None:
    meta_path = META_DIR / f"{video_id}.json"
    if meta_path.exists() and meta_path.stat().st_size > 0:
        return json.loads(meta_path.read_text())
    url = f"https://api.dvidshub.net/asset?api_key={DVIDS_API_KEY}&id=video:{video_id}&thumb_width=720"
    headers = {"Origin": "https://www.war.gov", "Referer": "https://www.war.gov/UFO/"}
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        meta_path.write_text(json.dumps(data, indent=2))
        return data
    except Exception as e:
        with ERR_LOG.open("a") as f:
            f.write(f"DVIDS_META\t{video_id}\t{e}\n")
        return None


def best_video_url(meta: dict) -> tuple[str, str] | None:
    """Pick highest-res mp4 from DVIDS asset metadata."""
    results = meta.get("results") or meta.get("data") or meta
    if not results:
        return None
    files = results.get("files") or []
    mp4s = [f for f in files if f.get("type") == "video/mp4"]
    if not mp4s:
        return None
    mp4s.sort(key=lambda f: (f.get("width", 0) * f.get("height", 0), f.get("size", 0)), reverse=True)
    chosen = mp4s[0]
    src = chosen["src"]
    name = src.rsplit("/", 1)[-1]
    return src, name


# Build the work list from downloads.tsv (PDFs + images)
direct_jobs: list[tuple[str, Path]] = []
seen: set[str] = set()
for line in (RAW / "downloads.tsv").read_text().splitlines():
    if not line.strip():
        continue
    url, local = line.split("\t", 1)
    if url in seen:
        continue
    seen.add(url)
    direct_jobs.append((url, ROOT / local))

# Add DVIDS video downloads
print(f"Resolving DVIDS metadata for {sum(1 for r in RECORDS if r['dvids_video_id'])} videos...")
video_jobs: list[tuple[str, Path]] = []
for r in RECORDS:
    vid = r["dvids_video_id"]
    if not vid:
        continue
    meta = fetch_dvids_meta(vid)
    if not meta:
        continue
    pair = best_video_url(meta)
    if not pair:
        continue
    src, name = pair
    dest = RAW / "videos" / f"{vid}_{name}"
    r["video_local"] = str(dest.relative_to(ROOT))
    if src not in seen:
        seen.add(src)
        video_jobs.append((src, dest))

# Persist video_local back into records.json
(RAW / "records.json").write_text(json.dumps(RECORDS, indent=2, ensure_ascii=False))

all_jobs = direct_jobs + video_jobs
print(f"Direct downloads: {len(direct_jobs)}  |  Videos: {len(video_jobs)}  |  Total: {len(all_jobs)}")


def worker(job):
    url, dest = job
    extra = None
    if "cloudfront" in url or "dvidshub" in url:
        extra = {"Origin": "https://www.war.gov"}
    status, size = fetch(url, dest, headers=extra)
    return url, dest, status, size


done = 0
ok = skip = err = 0
total_bytes = 0
start = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(worker, j) for j in all_jobs]
    for fut in as_completed(futs):
        url, dest, status, size = fut.result()
        done += 1
        if status == "ok":
            ok += 1
            total_bytes += size
        elif status == "skip":
            skip += 1
            total_bytes += size
        else:
            err += 1
        if done % 10 == 0 or done == len(all_jobs):
            elapsed = time.time() - start
            print(
                f"  [{done}/{len(all_jobs)}]  ok={ok} skip={skip} err={err}  {total_bytes/1e6:.1f}MB  "
                f"{elapsed:.0f}s",
                flush=True,
            )

print(f"\nDone. ok={ok}, skip={skip}, err={err}, bytes={total_bytes:,}")
if err:
    print(f"See {ERR_LOG}")

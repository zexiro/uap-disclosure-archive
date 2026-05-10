#!/usr/bin/env python3
"""Generate small JPEG thumbnails for every image surfaced in the row/grid UI.

Without this, the search UI references full-size source images (often
>1 MB each) for ~300+ row thumbnails, and lazy-loading just spreads the
pain across scroll. We pre-shrink to ~THUMB_WIDTH px wide JPEGs into
raw/thumbs/, mirroring the source path under raw/.

Run as part of the pipeline after extract_pdf_images.py and before
build_search_index.py — the latter populates each doc's `thumb_small`
field pointing at the generated thumb.
"""
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
THUMBS = RAW / "thumbs"

THUMB_WIDTH = 280
JPEG_QUALITY = 78


SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".jp2", ".tif", ".tiff")


def thumb_path_for(src_rel: str) -> Path | None:
    """Map raw/<rest> → raw/thumbs/<rest>.jpg. Returns None for non-images."""
    src = Path(src_rel)
    if not src.suffix.lower() in SUPPORTED_EXT:
        return None
    parts = src.parts
    if not parts or parts[0] != "raw":
        return None
    rel = Path(*parts[1:])
    return THUMBS / rel.with_suffix(".jpg")


def needs_rebuild(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    try:
        return src.stat().st_mtime > dst.stat().st_mtime
    except FileNotFoundError:
        return True


def make_thumb(src: Path, dst: Path) -> bool:
    """Render src into dst as a JPEG ~THUMB_WIDTH px wide. Returns True on success."""
    try:
        with Image.open(src) as im:
            im.load()
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            w, h = im.size
            if w > THUMB_WIDTH:
                ratio = THUMB_WIDTH / w
                im = im.resize((THUMB_WIDTH, max(1, int(h * ratio))), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        return True
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"[thumbs] skip {src}: {e}", file=sys.stderr)
        return False


def collect_sources() -> list[str]:
    """Every image we want a small thumb for: parent thumbnail_local[0] +
    every extracted image (since each becomes a synthetic IMG doc)."""
    seen: set[str] = set()
    records = json.loads((RAW / "records.json").read_text())
    for r in records:
        for t in r.get("thumbnail_local", []) or []:
            seen.add(t)
        # Also catch primary_local images (some records have only image primaries, no modal_image)
        for p in r.get("primary_local", []) or []:
            if Path(p).suffix.lower() in SUPPORTED_EXT:
                seen.add(p)
    extracted_path = RAW / "extracted_images.json"
    if extracted_path.exists():
        for e in json.loads(extracted_path.read_text()):
            seen.add(e["file"])
    return sorted(seen)


def main():
    sources = collect_sources()
    THUMBS.mkdir(parents=True, exist_ok=True)
    built = 0
    skipped = 0
    failed = 0
    missing_src = 0
    for src_rel in sources:
        dst = thumb_path_for(src_rel)
        if dst is None:
            continue
        src = ROOT / src_rel
        if not src.exists():
            missing_src += 1
            continue
        if not needs_rebuild(src, dst):
            skipped += 1
            continue
        if make_thumb(src, dst):
            built += 1
        else:
            failed += 1
    print(f"[thumbs] built={built} skipped={skipped} failed={failed} missing_src={missing_src} total_sources={len(sources)}")


if __name__ == "__main__":
    main()

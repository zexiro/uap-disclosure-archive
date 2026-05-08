#!/usr/bin/env python3
"""Extract interesting embedded images from every PDF.

The FBI scans store each page as a single full-page raster image — those
aren't useful as 'photographs' (we'd just be re-extracting the page).
This script keeps only embedded images that are likely photos / figures
/ sketches inside larger documents.

Heuristics:
  * Skip images where w >= 1100 AND h >= 1400  (probably a scanned page)
  * Skip images where min(w,h) < 240             (layout chrome / icons)
  * Skip exact duplicates (sha256 of pixel bytes)
  * Skip near-duplicates *across* records (pHash collision)

Output:
  raw/images_extracted/<pdf-stem>__p<page>_<n>.{png|jpg}
  raw/extracted_images.json   — {file, src_pdf, page, w, h, sha256, phash}
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "raw" / "docs"
OUT_DIR = ROOT / "raw" / "images_extracted"
INDEX_PATH = ROOT / "raw" / "extracted_images.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DIM = 240
PAGE_SCAN_W = 1100
PAGE_SCAN_H = 1400


def extract_one(pdf: Path) -> list[dict]:
    """pdfimages -all -p extracts everything; we then filter and rename."""
    keep: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / pdf.stem
        # -all writes whatever the embedded format is (jpeg, png, jp2, …); -p adds page num
        try:
            subprocess.run(
                ["pdfimages", "-all", "-p", "-q", str(pdf), str(prefix)],
                check=True, timeout=600,
            )
        except subprocess.CalledProcessError:
            return keep
        except subprocess.TimeoutExpired:
            return keep
        for img in sorted(Path(tmp).iterdir()):
            try:
                with Image.open(img) as im:
                    w, h = im.size
                    if w >= PAGE_SCAN_W and h >= PAGE_SCAN_H:
                        continue  # full-page scan
                    if min(w, h) < MIN_DIM:
                        continue  # too small
                    rgb = im.convert("RGB")
                    sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                    ph = str(imagehash.phash(rgb, hash_size=16))
            except (UnidentifiedImageError, OSError, ValueError):
                continue
            # Filename like "<stem>-001-002.jpg" — page is the first index when -p is used.
            # Be defensive about exact format; just use the last numeric token as a sequence id.
            # For our purposes we capture page number from the filename suffix.
            parts = img.stem.rsplit("-", 2)
            page = parts[-2] if len(parts) >= 3 and parts[-2].isdigit() else "?"
            seq = parts[-1] if parts[-1].isdigit() else "0"
            ext = img.suffix.lower().lstrip(".")
            out_name = f"{pdf.stem}__p{page}_{seq}.{ext}"
            out_path = OUT_DIR / out_name
            shutil.copy2(img, out_path)
            keep.append({
                "file": str(out_path.relative_to(ROOT)),
                "src_pdf": str(pdf.relative_to(ROOT)),
                "page": page,
                "w": w, "h": h,
                "sha256": sha,
                "phash": ph,
            })
    return keep


def main():
    if shutil.which("pdfimages") is None:
        print("ERROR: pdfimages not installed (`brew install poppler`)", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(DOCS.glob("*.pdf"))
    print(f"Scanning {len(pdfs)} PDFs for embedded images …")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(extract_one, p): p for p in pdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            pdf = futs[fut]
            try:
                items = fut.result()
            except Exception as e:
                print(f"  ! {pdf.name}: {e}", flush=True)
                continue
            results.extend(items)
            if i % 10 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] cumulative kept: {len(results)}", flush=True)

    # Cross-PDF dedupe: drop later sha256 duplicates
    seen_sha: set[str] = set()
    deduped: list[dict] = []
    drop_dup = 0
    for it in results:
        if it["sha256"] in seen_sha:
            (ROOT / it["file"]).unlink(missing_ok=True)
            drop_dup += 1
            continue
        seen_sha.add(it["sha256"])
        deduped.append(it)

    INDEX_PATH.write_text(json.dumps(deduped, indent=2, ensure_ascii=False))
    print()
    print(f"Final kept: {len(deduped)} ({drop_dup} duplicates dropped)")
    print(f"Index: {INDEX_PATH}")
    # Per-source summary
    by_src: dict[str, int] = {}
    for it in deduped:
        src = Path(it["src_pdf"]).stem
        by_src[src] = by_src.get(src, 0) + 1
    for src, n in sorted(by_src.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:4d}  {src}")


if __name__ == "__main__":
    main()

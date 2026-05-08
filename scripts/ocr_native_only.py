#!/usr/bin/env python3
"""Fast pass: extract text from PDFs that already have a native text layer.

Skips any PDF that:
  - already has a sidecar .txt
  - is currently being processed by ocrmypdf (lock file under /tmp/ocrmypdf.io.*)

This complements scripts/ocr.py (which is the slow OCR fallback). Intended
to run while ocr.py is still grinding through scanned FBI files in the
background — gives us searchable native text immediately for the ~80% of
the corpus that's not image-only scans.
"""
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "raw" / "docs"
TEXT = ROOT / "raw" / "text"
TEXT.mkdir(parents=True, exist_ok=True)

MIN_NATIVE_CHARS = 500


def in_use(pdf: Path) -> bool:
    """Crude check: is this PDF currently being read by an ocrmypdf worker?"""
    try:
        out = subprocess.check_output(["lsof", "-t", str(pdf)], stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def extract(pdf: Path):
    txt = TEXT / (pdf.stem + ".txt")
    if txt.exists() and txt.stat().st_size > 0:
        return pdf, "skip"
    if in_use(pdf):
        return pdf, "locked"
    try:
        # Write to a temp file first so partial results don't appear as "done"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", dir=str(TEXT)) as tmp:
            tmp_path = Path(tmp.name)
        out = subprocess.run(
            ["pdftotext", "-layout", "-q", str(pdf), str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size >= MIN_NATIVE_CHARS:
            tmp_path.rename(txt)
            return pdf, "native"
        tmp_path.unlink(missing_ok=True)
        return pdf, "thin_or_scan"
    except Exception as e:
        return pdf, f"err:{e}"


def main():
    pdfs = sorted(DOCS.glob("*.pdf"))
    print(f"Scanning {len(pdfs)} PDFs for native-text extraction…")
    stats = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(extract, p) for p in pdfs]
        for i, fut in enumerate(as_completed(futs), 1):
            pdf, status = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if i % 20 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] {stats}", flush=True)
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract searchable text from every PDF in raw/docs/.

Strategy:
  1. Try `pdftotext` (fast, native text). If output is non-trivial (>500 chars), use it.
  2. Otherwise the PDF is probably a scan — run `ocrmypdf --sidecar` to OCR it.

Output: raw/text/<stem>.txt next to one cached OCR'd PDF in raw/docs_ocr/<stem>.pdf
"""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DOCS = RAW / "docs"
TEXT = RAW / "text"
OCR_DIR = RAW / "docs_ocr"
TEXT.mkdir(exist_ok=True)
OCR_DIR.mkdir(exist_ok=True)

MIN_NATIVE_CHARS = 500


def have(cmd):
    return shutil.which(cmd) is not None


def extract_one(pdf: Path) -> tuple[Path, str, int]:
    txt_path = TEXT / (pdf.stem + ".txt")
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return pdf, "skip", txt_path.stat().st_size

    # 1) pdftotext native
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-q", str(pdf), str(txt_path)],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        return pdf, f"pdftotext_err:{e}", 0

    if out.returncode == 0 and txt_path.exists():
        size = txt_path.stat().st_size
        if size >= MIN_NATIVE_CHARS:
            return pdf, "native", size
        # Not enough text — assume scan
        txt_path.unlink(missing_ok=True)

    # 2) OCR fallback
    if not have("ocrmypdf"):
        # Even without ocrmypdf, write whatever pdftotext got
        try:
            out = subprocess.run(
                ["pdftotext", "-q", str(pdf), str(txt_path)],
                capture_output=True, text=True, timeout=300,
            )
            return pdf, "native_thin", txt_path.stat().st_size if txt_path.exists() else 0
        except Exception:
            return pdf, "no_text", 0

    ocr_pdf = OCR_DIR / pdf.name
    sidecar = TEXT / (pdf.stem + ".txt")
    try:
        subprocess.run(
            [
                "ocrmypdf",
                "--quiet",
                "--skip-text",       # skip pages that already have text
                "--rotate-pages",
                "--deskew",
                "--optimize", "0",
                "--sidecar", str(sidecar),
                str(pdf),
                str(ocr_pdf),
            ],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        return pdf, "ocr", sidecar.stat().st_size if sidecar.exists() else 0
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "")[:300].replace("\n", " ")
        return pdf, f"ocr_err:{msg}", 0
    except Exception as e:
        return pdf, f"ocr_err:{e}", 0


def main():
    if not have("pdftotext"):
        print("ERROR: pdftotext not installed (`brew install poppler`)", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(DOCS.glob("*.pdf"))
    print(f"Extracting text from {len(pdfs)} PDFs…")
    stats = {"native": 0, "ocr": 0, "skip": 0, "native_thin": 0, "no_text": 0, "err": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(extract_one, p) for p in pdfs]
        for i, fut in enumerate(as_completed(futs), 1):
            pdf, status, size = fut.result()
            if status.startswith("ocr_err") or status.startswith("pdftotext_err"):
                stats["err"] += 1
                failures.append((pdf.name, status))
            else:
                stats[status] = stats.get(status, 0) + 1
            if i % 5 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] {stats}", flush=True)
    print("\nFinal:", stats)
    if failures:
        (RAW / "ocr_errors.log").write_text("\n".join(f"{n}\t{s}" for n, s in failures))
        print(f"  {len(failures)} failures → raw/ocr_errors.log")


if __name__ == "__main__":
    main()

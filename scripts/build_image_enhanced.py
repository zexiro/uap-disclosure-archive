#!/usr/bin/env python3
"""Real-ESRGAN 4× upscaler — Wave 3 of IMAGE_FORENSICS_PLAN.md.

For every image under raw/images/, produce a 4×-upscaled JPEG sibling
at raw/images_enhanced/<name>.jpg using the realesr-general-x4v3 ONNX
model (~5 MB, fully convolutional, accepts any input size).

Tile-based with overlap blending so a 4 K source doesn't OOM the CPU
runner. Idempotent: existing output files are skipped, so a partial
run resumes cleanly on the next pipeline pass.

Output framing rule (per IMAGE_FORENSICS_PLAN.md): the lightbox surfaces
these as a "plausible reconstruction, not new evidence" toggle — never
as a replacement for the original image.

Skipped intentionally:
- raw/images_extracted/  These are PDF-embedded thumbs that get re-scaled
                         by pdfimages already; upscaling them produces
                         the artifacts of two scalers stacked. The
                         lightbox will only offer the enhance toggle
                         for images that have a direct download.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "raw" / "images"
OUT_DIR = ROOT / "raw" / "images_enhanced"
MODEL_REPO = "Samo629/real-esrgan-onnx"
MODEL_FILE = "realesr-general-x4v3.onnx"
JPEG_QUALITY = 92

# Tile size in INPUT pixels. The output is 4× this. 192² input → 768²
# output → 2.4 MB float32 activation, fits comfortably on a 512 MB
# Railway runner. Overlap blends the seams.
TILE_IN = 192
OVERLAP_IN = 16
SCALE = 4

# Skip images bigger than this on either edge in INPUT pixels. 4× a
# 4 000-pixel side is 16 000 px → 256 MB JPEG decode tree. Pointless
# blast radius for a forensic toggle.
MAX_INPUT_EDGE = 3500


def load_model():
    """Resolve and load the ONNX model. Downloads on first call, cached
    by huggingface_hub thereafter."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"[enhance] loaded model {Path(path).name} ({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _weight_mask(h: int, w: int, taper: int) -> np.ndarray:
    """Triangular falloff at each edge so overlapping tiles blend smoothly
    instead of leaving a visible seam. taper is in OUTPUT pixels."""
    if taper <= 0:
        return np.ones((h, w), dtype=np.float32)
    y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) / taper
    x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) / taper
    y = np.clip(y, 0.0, 1.0)
    x = np.clip(x, 0.0, 1.0)
    return np.outer(y, x).clip(0.001, None)  # never fully zero


def upscale_tile(sess, tile: np.ndarray) -> np.ndarray:
    """Run one ONNX pass on an HWC uint8 tile, return HWC float32 [0,1]
    of the 4×-larger output."""
    x = (tile.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    y = sess.run(None, {"input": x})[0]
    return np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0)


def upscale_image(sess, img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))  # HWC uint8
    h, w, _ = arr.shape

    # Short circuit: small enough to upscale in one shot.
    if h <= TILE_IN and w <= TILE_IN:
        out = upscale_tile(sess, arr)
        return Image.fromarray(np.round(out * 255).astype(np.uint8))

    step = TILE_IN - OVERLAP_IN
    out_h, out_w = h * SCALE, w * SCALE
    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wgt = np.zeros((out_h, out_w, 1), dtype=np.float32)

    ys = list(range(0, max(1, h - TILE_IN), step)) + [max(0, h - TILE_IN)]
    xs = list(range(0, max(1, w - TILE_IN), step)) + [max(0, w - TILE_IN)]
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for y0 in ys:
        for x0 in xs:
            tile = arr[y0:y0 + TILE_IN, x0:x0 + TILE_IN]
            th, tw = tile.shape[:2]
            if th < TILE_IN or tw < TILE_IN:
                # Pad short tiles to TILE_IN with edge replication so the
                # ONNX shape is consistent; we crop again after.
                pad_y = TILE_IN - th
                pad_x = TILE_IN - tw
                tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
            out_tile = upscale_tile(sess, tile)
            # Drop the padded region from the output.
            out_tile = out_tile[: th * SCALE, : tw * SCALE]
            mask = _weight_mask(th * SCALE, tw * SCALE, OVERLAP_IN * SCALE)[..., None]
            oy = y0 * SCALE
            ox = x0 * SCALE
            acc[oy:oy + th * SCALE, ox:ox + tw * SCALE] += out_tile * mask
            wgt[oy:oy + th * SCALE, ox:ox + tw * SCALE] += mask
    out = acc / np.maximum(wgt, 1e-6)
    return Image.fromarray(np.round(out * 255).astype(np.uint8))


def main() -> int:
    if not IN_DIR.exists():
        print(f"[enhance] {IN_DIR} doesn't exist; skipping")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(IN_DIR.glob("*.jpg")) + sorted(IN_DIR.glob("*.png"))
    todo: list[Path] = []
    for p in paths:
        out_path = OUT_DIR / (p.stem + ".jpg")
        if out_path.exists():
            continue
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception as e:
            print(f"[enhance] {p.name}: cannot open ({e})", flush=True)
            continue
        if max(w, h) > MAX_INPUT_EDGE:
            print(f"[enhance] {p.name}: skip (max edge {max(w, h)} > {MAX_INPUT_EDGE})", flush=True)
            continue
        todo.append(p)

    if not todo:
        print(f"[enhance] all up to date ({len(paths)} source images, 0 to enhance)")
        return 0

    print(f"[enhance] {len(todo)} images to enhance (of {len(paths)} total)", flush=True)
    try:
        sess = load_model()
    except Exception as e:
        print(f"[enhance] model load failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)
        return 0

    t_total = time.time()
    for i, src in enumerate(todo, 1):
        out_path = OUT_DIR / (src.stem + ".jpg")
        try:
            t0 = time.time()
            with Image.open(src) as img:
                enhanced = upscale_image(sess, img)
            # Save with a tag in the JPEG comment so casual exiftool
            # inspection can't confuse the upscaled output with an original.
            enhanced.save(
                out_path,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
                comment="Real-ESRGAN realesr-general-x4v3 4x reconstruction - not an original".encode("utf-8"),
            )
            print(f"[enhance] {i}/{len(todo)} {src.name} → {enhanced.size}  [{time.time()-t0:.1f}s]", flush=True)
        except Exception as e:
            print(f"[enhance] {src.name}: ERR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"[enhance] done in {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

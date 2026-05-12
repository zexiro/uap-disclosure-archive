#!/usr/bin/env python3
"""Depth-Anything-v2 monocular depth maps — Wave 3 of IMAGE_FORENSICS_PLAN.md.

For every image under raw/images/ (NOT images_extracted/), produce a
colorised depth-map PNG sibling at raw/images_depth/<stem>.png.

Model: onnx-community/depth-anything-v2-small (quantized ONNX, ~27 MB).
Output is a single-channel relative depth tensor — larger values mean
"closer to camera". We normalise per-image (min/max), upscale back to
the source resolution with bilinear filtering, and colourise with a
turbo-style LUT for the visual dramatic effect.

Framing rule (per IMAGE_FORENSICS_PLAN.md): the lightbox labels this
as a "monocular depth estimate, not a measurement" — it's a single-
view neural-net guess, useful for "this looks close, that looks far"
intuition but absolutely not an authoritative spatial reading.

Skipped intentionally:
- raw/images_extracted/  These are PDF-rescaled thumbs; depth on them
                         tracks the PDF render artifact more than the
                         original scene.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "raw" / "images"
OUT_DIR = ROOT / "raw" / "images_depth"
MODEL_REPO = "onnx-community/depth-anything-v2-small"
MODEL_FILE = "onnx/model_quantized.onnx"
INPUT_SIZE = 518  # Depth-Anything-v2 native input
PNG_OPT_LEVEL = 6

# ImageNet preprocessing — same mean/std the model was trained on.
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"[depth] loaded {Path(path).name} ({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _build_turbo_lut() -> np.ndarray:
    """Approximation of the Google Turbo colormap. 256 entries (R, G, B)
    of uint8. Pure-numpy so we don't need matplotlib in production.

    Derived from the polynomial fit published in
    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    """
    t = np.linspace(0, 1, 256, dtype=np.float32)
    # Polynomial fits (degree 6) for each channel
    r = 0.13572138 + 4.61539260 * t - 42.66032258 * t**2 + 132.13108234 * t**3 \
        - 152.94239396 * t**4 + 59.28637943 * t**5
    g = 0.09140261 + 2.19418839 * t + 4.84296658 * t**2 - 14.18503333 * t**3 \
        + 4.27729857 * t**4 + 2.82956604 * t**5
    b = 0.10667330 + 12.64194608 * t - 60.58204836 * t**2 + 110.36276771 * t**3 \
        - 89.90310912 * t**4 + 27.34824973 * t**5
    rgb = np.stack([r, g, b], axis=1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


TURBO_LUT = _build_turbo_lut()


def estimate_depth(sess, img: Image.Image) -> np.ndarray:
    """Run one depth pass. Returns a float32 [H, W] map at the input's
    original resolution, normalised to [0, 1] (1 = closest)."""
    src_w, src_h = img.size
    # Resize to model native input.
    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    arr = arr.transpose(2, 0, 1)[None]
    raw = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
    # Output shape is (1, H, W) — squeeze.
    depth = raw[0] if raw.ndim == 3 else raw[0, 0]
    # Resize back to source resolution. PIL handles the scale.
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros((src_h, src_w), dtype=np.float32)
    norm = ((depth - dmin) / (dmax - dmin)).astype(np.float32)
    dimg = Image.fromarray(np.round(norm * 255).astype(np.uint8))
    full = dimg.resize((src_w, src_h), Image.BILINEAR)
    return np.array(full, dtype=np.float32) / 255.0


def colorize(depth: np.ndarray) -> Image.Image:
    """Map [0, 1] depth → RGB via Turbo LUT."""
    idx = np.clip(np.round(depth * 255), 0, 255).astype(np.int32)
    rgb = TURBO_LUT[idx]  # shape: (H, W, 3) uint8
    return Image.fromarray(rgb, mode="RGB")


def main() -> int:
    if not IN_DIR.exists():
        print(f"[depth] {IN_DIR} doesn't exist; skipping")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(IN_DIR.glob("*.jpg")) + sorted(IN_DIR.glob("*.png"))
    todo: list[Path] = []
    for p in paths:
        out_path = OUT_DIR / (p.stem + ".png")
        if out_path.exists():
            continue
        todo.append(p)
    if not todo:
        print(f"[depth] all up to date ({len(paths)} source images)")
        return 0

    print(f"[depth] {len(todo)}/{len(paths)} to process", flush=True)
    try:
        sess = load_model()
    except Exception as e:
        print(f"[depth] model load failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)
        return 0

    t_total = time.time()
    for i, src in enumerate(todo, 1):
        out_path = OUT_DIR / (src.stem + ".png")
        try:
            t0 = time.time()
            with Image.open(src) as img:
                depth = estimate_depth(sess, img.convert("RGB"))
            color = colorize(depth)
            color.save(out_path, format="PNG", optimize=False, compress_level=PNG_OPT_LEVEL)
            print(f"[depth] {i}/{len(todo)} {src.name} → {color.size}  [{time.time()-t0:.1f}s]", flush=True)
        except Exception as e:
            print(f"[depth] {src.name}: ERR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"[depth] done in {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

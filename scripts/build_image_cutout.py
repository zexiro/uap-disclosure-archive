#!/usr/bin/env python3
"""BiRefNet foreground cutout — Wave 3 of IMAGE_FORENSICS_PLAN.md.

For every image under raw/images/, segment the salient foreground with
BiRefNet_lite (fp16 ONNX, ~114 MB) and write a transparent PNG cutout
to raw/images_cutout/<stem>.png.

The lightbox surfaces this as a "✂ cutout" toggle alongside enhance /
depth — useful for isolating the object of interest from a noisy or
distracting background (e.g. seeing the actual disc shape in the
composite-sketch without the sky and trees competing for attention).

Skipped intentionally:
- raw/images_extracted/  PDF-rescaled thumbs; salient segmentation
                         on these is essentially noise.

Tile-free: BiRefNet wants a fixed 1024×1024 input. We letterbox the
source (preserves aspect ratio, pads grey), run inference once, then
crop the output back to the un-padded region and resize to the
source's native dimensions. One inference per image (~3–6 s on CPU).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "raw" / "images"
OUT_DIR = ROOT / "raw" / "images_cutout"
MODEL_REPO = "onnx-community/BiRefNet_lite-ONNX"
MODEL_FILE = "onnx/model_fp16.onnx"
INPUT_SIZE = 1024
PNG_OPT_LEVEL = 6

# Standard ImageNet normalisation — what BiRefNet_lite was trained on.
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"[cutout] loaded {Path(path).name} ({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _letterbox(img: Image.Image) -> tuple[np.ndarray, int, int, int, int]:
    """Resize preserving aspect ratio; pad to INPUT_SIZE × INPUT_SIZE
    with mid-grey. Returns (NCHW float32 tensor, new_w, new_h, pad_x, pad_y)
    so we can crop the inferred mask back to the un-padded region."""
    w, h = img.size
    scale = INPUT_SIZE / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    arr = arr.transpose(2, 0, 1)[None]
    return arr, new_w, new_h, pad_x, pad_y


def segment(sess, img: Image.Image) -> Image.Image:
    """Return a transparent-background RGBA cutout at the source's native size."""
    src_w, src_h = img.size
    arr, new_w, new_h, pad_x, pad_y = _letterbox(img)
    # I/O is float32 even on the fp16-internal model variant.
    raw = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
    # Output shape: (1, 1, 1024, 1024) — squeeze to (1024, 1024) float
    mask = raw[0, 0].astype(np.float32)
    # The model emits raw logits; sigmoid them and clamp into [0, 1].
    mask = 1.0 / (1.0 + np.exp(-mask))
    # Crop out the un-padded region and resize back to the source size.
    cropped = mask[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    alpha = cropped_img.resize((src_w, src_h), Image.BILINEAR)
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def main() -> int:
    if not IN_DIR.exists():
        print(f"[cutout] {IN_DIR} doesn't exist; skipping")
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
        print(f"[cutout] all up to date ({len(paths)} source images)")
        return 0

    print(f"[cutout] {len(todo)}/{len(paths)} to process", flush=True)
    try:
        sess = load_model()
    except Exception as e:
        print(f"[cutout] model load failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)
        return 0

    t_total = time.time()
    for i, src in enumerate(todo, 1):
        out_path = OUT_DIR / (src.stem + ".png")
        try:
            t0 = time.time()
            with Image.open(src) as img:
                rgba = segment(sess, img.convert("RGB"))
            rgba.save(out_path, format="PNG", optimize=False, compress_level=PNG_OPT_LEVEL)
            print(f"[cutout] {i}/{len(todo)} {src.name} → {rgba.size}  [{time.time()-t0:.1f}s]", flush=True)
        except Exception as e:
            print(f"[cutout] {src.name}: ERR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"[cutout] done in {time.time()-t_total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

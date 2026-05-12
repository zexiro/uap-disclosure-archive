#!/usr/bin/env python3
"""Trained AI-generated / deepfake classifier — Wave 3 of IMAGE_FORENSICS_PLAN.md.

Runs the int8-quantized Deep-Fake-Detector-v2 ONNX (~87 MB) — a
ViT binary classifier with id2label = {0: "Realism", 1: "Deepfake"}
— over every image and writes the softmax probability to
ui/ai_detector.json, keyed by filename.

This complements the CLIP zero-shot synthetic-similarity score
(scripts/build_clip_labels.py). CLIP gives a *similarity*; this gives
a *classifier probability*. Two different signals — visitors see both
in the metadata panel so they can compare.

Hard limits (these matter for framing):
- The classifier was trained on face-deepfake datasets and modern
  generative-image leaks. It generalises okay to "image looks
  synthetic," but it will hallucinate on stylised sketches, scanned
  documents, and aerial photos. The model name + training context are
  ALWAYS shown in the UI so visitors can judge the framing.
- Output is a per-image probability, NOT a verdict. The metadata panel
  uses the same colour-coded bar idiom as the CLIP score with a clear
  "trained classifier — not infallible" disclaimer.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIRS = [ROOT / "raw" / "images", ROOT / "raw" / "images_extracted"]
OUT_PATH = ROOT / "ui" / "ai_detector.json"

MODEL_REPO = "onnx-community/Deep-Fake-Detector-v2-Model-ONNX"
MODEL_FILE = "onnx/model_int8.onnx"
INPUT_SIZE = 224  # ViT base
# Preprocessor config from the repo: [0.5, 0.5, 0.5] mean & std → normalise
# pixel values from [0, 1] to [-1, 1].
NORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
NORM_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def load_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"[ai-det] loaded {Path(path).name} ({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize to 224², normalise to [-1, 1], NCHW."""
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    a = np.array(img, dtype=np.float32) / 255.0
    a = (a - NORM_MEAN) / NORM_STD
    return a.transpose(2, 0, 1)[None]


def classify(sess, img: Image.Image) -> dict:
    x = preprocess(img)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    probs = _softmax(logits)
    return {
        "realism": round(float(probs[0]), 3),
        "deepfake": round(float(probs[1]), 3),
    }


def main() -> int:
    paths: list[Path] = []
    for d in IMG_DIRS:
        if not d.exists():
            print(f"[ai-det] {d} doesn't exist; skipping")
            continue
        paths.extend(sorted(d.glob("*.jpg")))
        paths.extend(sorted(d.glob("*.png")))
    if not paths:
        print("[ai-det] no images to process")
        return 0

    print(f"[ai-det] {len(paths)} images to process", flush=True)
    try:
        sess = load_model()
    except Exception as e:
        print(f"[ai-det] model load failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    t_total = time.time()
    for i, p in enumerate(paths, 1):
        try:
            with Image.open(p) as img:
                rec = classify(sess, img)
            rec["model"] = "Deep-Fake-Detector-v2 (int8)"
            out[p.name] = rec
        except Exception as e:
            print(f"[ai-det] {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
        if i % 100 == 0:
            print(f"[ai-det]   {i}/{len(paths)} processed [{time.time()-t_total:.0f}s]", flush=True)

    OUT_PATH.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")))
    elapsed = time.time() - t_total
    print(f"[ai-det] wrote {len(out)} entries → {OUT_PATH.relative_to(ROOT)}  [{elapsed:.0f}s]")

    high = sum(1 for r in out.values() if r["deepfake"] >= 0.7)
    mid = sum(1 for r in out.values() if 0.5 <= r["deepfake"] < 0.7)
    print(f"[ai-det]   {high} ≥0.70 deepfake-leaning, {mid} in [0.50, 0.70)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

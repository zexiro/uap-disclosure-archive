#!/usr/bin/env python3
"""YOLOv8n object detection — Wave 3 of IMAGE_FORENSICS_PLAN.md.

Runs the YOLOv8n COCO-80 detection model (~13 MB ONNX) over every image
under raw/images/ and raw/images_extracted/ and writes the union of
detected objects to ui/object_detections.json, keyed by filename.

Pipeline:
- Letterbox-resize each image to 640×640 (preserves aspect ratio,
  pads black so the model never has to interpret stretched geometry).
- ONNX inference (CPU): output is [1, 84, 8400] — 4 box coords + 80
  class scores per anchor point.
- Decode + class-aware NMS at IoU 0.45.
- Map boxes back to the original image's pixel coords (undo letterbox).

Per-image output:
  {
    "model": "yolov8n",
    "image_size": [w, h],
    "detections": [
      {"label": "airplane", "score": 0.84, "box": [x1, y1, x2, y2]},
      ...
    ],
    "label_counts": {"airplane": 1, "person": 3}
  }

Detections under SCORE_THRESHOLD are dropped before NMS — the model
hallucinates plenty of low-confidence garbage on stylised sketches and
heavily-compressed scans, and surfacing every one would drown the
useful hits. The metadata panel only renders detections at score ≥ 0.35.
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
OUT_PATH = ROOT / "ui" / "object_detections.json"

MODEL_REPO = "salim4n/yolov8n-detect-onnx"
MODEL_FILE = "yolov8n-onnx-web/yolov8n.onnx"
INPUT_SIZE = 640
SCORE_THRESHOLD = 0.30
NMS_IOU = 0.45
MAX_DETECTIONS = 25  # per image

# Standard COCO-80 label order — the model was trained against this.
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"[detect] loaded {Path(path).name} ({Path(path).stat().st_size/1e6:.1f} MB)", flush=True)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def letterbox(img: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio; pad with grey. Returns the input
    tensor (CHW float32, 0..1) plus scale + padding offsets so we can
    undo the transform on the output boxes."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = (np.array(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    return arr, scale, pad_x, pad_y


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box [4] against N boxes [N,4] — xyxy format."""
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / np.maximum(1e-9, area_a + area_b - inter)


def _nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thresh: float) -> list[int]:
    """Class-aware NMS — only suppress boxes of the same class."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same_class = classes[rest] == classes[i]
        ious = _iou(boxes[i], boxes[rest])
        drop = same_class & (ious > iou_thresh)
        order = rest[~drop]
    return keep


def decode_output(raw: np.ndarray, scale: float, pad_x: int, pad_y: int,
                  orig_w: int, orig_h: int) -> list[dict]:
    """YOLOv8 output is [1, 84, 8400]. First 4 channels are xc, yc, w, h
    in 640-pixel input space; the remaining 80 are per-class scores."""
    out = raw[0].T  # → [8400, 84]
    boxes_xywh = out[:, :4]
    class_scores = out[:, 4:]
    cls_id = class_scores.argmax(axis=1)
    cls_score = class_scores.max(axis=1)
    mask = cls_score >= SCORE_THRESHOLD
    if not mask.any():
        return []
    boxes_xywh = boxes_xywh[mask]
    cls_id = cls_id[mask]
    cls_score = cls_score[mask]
    # xywh → xyxy in input space
    xy1 = boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2
    xy2 = boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2
    boxes_xyxy = np.concatenate([xy1, xy2], axis=1)
    # Undo letterbox: subtract padding, divide by scale, clip to original.
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h - 1)
    keep = _nms(boxes_xyxy, cls_score, cls_id, NMS_IOU)
    keep = keep[:MAX_DETECTIONS]
    out_list = []
    for i in keep:
        out_list.append({
            "label": COCO_LABELS[int(cls_id[i])],
            "score": round(float(cls_score[i]), 3),
            "box": [round(float(x), 1) for x in boxes_xyxy[i].tolist()],
        })
    return out_list


def process_image(sess, path: Path) -> dict | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            arr, scale, pad_x, pad_y = letterbox(img)
    except Exception as e:
        print(f"[detect] {path.name}: cannot open ({e})", file=sys.stderr)
        return None
    out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
    detections = decode_output(out, scale, pad_x, pad_y, orig_w, orig_h)
    counts: dict[str, int] = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return {
        "model": "yolov8n",
        "image_size": [orig_w, orig_h],
        "detections": detections,
        "label_counts": counts,
    }


def main() -> int:
    paths: list[Path] = []
    for d in IMG_DIRS:
        if not d.exists():
            print(f"[detect] {d} doesn't exist; skipping")
            continue
        paths.extend(sorted(d.glob("*.jpg")))
        paths.extend(sorted(d.glob("*.png")))
    if not paths:
        print("[detect] no images to process")
        return 0

    print(f"[detect] {len(paths)} images to process", flush=True)
    try:
        sess = load_model()
    except Exception as e:
        print(f"[detect] model load failed ({type(e).__name__}: {e}); skipping", file=sys.stderr)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    t_total = time.time()
    for i, p in enumerate(paths, 1):
        rec = process_image(sess, p)
        if rec is not None:
            out[p.name] = rec
        if i % 50 == 0:
            print(f"[detect]   {i}/{len(paths)} processed [{time.time()-t_total:.0f}s]", flush=True)

    OUT_PATH.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")))
    elapsed = time.time() - t_total
    print(f"[detect] wrote {len(out)} entries → {OUT_PATH.relative_to(ROOT)}  [{elapsed:.0f}s]")

    # Tally — how many images had something detected.
    any_hit = sum(1 for r in out.values() if r["detections"])
    label_total: dict[str, int] = {}
    for r in out.values():
        for k, v in r["label_counts"].items():
            label_total[k] = label_total.get(k, 0) + v
    top = sorted(label_total.items(), key=lambda kv: -kv[1])[:8]
    print(f"[detect]   {any_hit}/{len(out)} images had ≥1 detection")
    print(f"[detect]   top labels: {', '.join(f'{k}={v}' for k,v in top)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract scene-boundary keyframes from VID records and append CLIP embeddings.

For each VID record in ui/search-index.json whose local video file exists:
  1. Detect scene boundaries with PySceneDetect ContentDetector (threshold=27).
  2. Extract one representative frame (middle frame) per scene.
  3. Save as raw/videos/keyframes/<record_id>/<scene_idx>.jpg (384px long-edge).
  4. CLIP-embed keyframes with fastembed ImageEmbedding (Qdrant/clip-ViT-B-32-vision).
  5. Append vectors to ui/image_embeddings.npz with ids like vid:<record_id>:scene000.

Output:
  raw/videos/keyframes/<record_id>/  — one JPEG per detected scene
  ui/image_embeddings.npz            — updated with new keyframe vectors appended

Idempotent: skips a video's keyframe dir if it already exists.
Atomic write: saves .npz to a temp file then renames.
"""

import json
import os
import sys
import time
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastembed import ImageEmbedding
from scenedetect import detect, ContentDetector

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
EMBEDDINGS_PATH = ROOT / "ui" / "image_embeddings.npz"
KEYFRAMES_BASE = ROOT / "raw" / "videos" / "keyframes"

MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
SCENE_THRESHOLD = 27
LONG_EDGE = 384
BATCH_SIZE = 16


def resize_to_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize image so its longest dimension = long_edge, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= long_edge:
        return img
    if w >= h:
        new_w = long_edge
        new_h = int(h * long_edge / w)
    else:
        new_h = long_edge
        new_w = int(w * long_edge / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_existing_embeddings() -> tuple[list[str], np.ndarray]:
    """Load existing ids and vectors from the .npz file."""
    if not EMBEDDINGS_PATH.exists():
        return [], np.empty((0, 512), dtype=np.float32)
    try:
        npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        ids = [str(x) for x in npz["ids"]]
        vecs = npz["vectors"].astype(np.float32)
        print(f"[vid-kf] loaded {len(ids)} existing embeddings from {EMBEDDINGS_PATH.name}")
        return ids, vecs
    except Exception as e:
        print(f"[vid-kf] warning: could not read existing .npz ({e}), starting fresh")
        return [], np.empty((0, 512), dtype=np.float32)


def extract_keyframes(video_path: Path, record_id: str) -> list[Path]:
    """Detect scenes, extract mid-frame per scene, save as JPEG. Returns list of saved paths."""
    out_dir = KEYFRAMES_BASE / record_id
    if out_dir.exists():
        # Idempotent: skip if already extracted
        existing = sorted(out_dir.glob("*.jpg"))
        if existing:
            print(f"[vid-kf] {record_id}: keyframe dir exists ({len(existing)} frames) — skipping")
            return existing

    try:
        scenes = detect(
            str(video_path),
            ContentDetector(threshold=SCENE_THRESHOLD),
            start_in_scene=True,
        )
    except Exception as e:
        print(f"[vid-kf] ERROR detecting scenes for {record_id}: {e}", file=sys.stderr)
        return []

    if not scenes:
        print(f"[vid-kf] {record_id}: no scenes detected — skipping")
        return []

    print(f"[vid-kf] {record_id}: {len(scenes)} scenes detected")

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved_paths: list[Path] = []

    try:
        for scene_idx, (start, end) in enumerate(scenes):
            mid_frame = (start.frame_num + end.frame_num) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"[vid-kf] {record_id}: scene {scene_idx}: could not read frame {mid_frame}", file=sys.stderr)
                continue

            # BGR → RGB → PIL
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = resize_to_long_edge(pil_img, LONG_EDGE)

            out_path = out_dir / f"{scene_idx:03d}.jpg"
            pil_img.save(out_path, "JPEG", quality=85)
            saved_paths.append(out_path)
    finally:
        cap.release()

    print(f"[vid-kf] {record_id}: saved {len(saved_paths)} keyframes → {out_dir}")
    return saved_paths


def embed_images(model: ImageEmbedding, paths: list[Path]) -> list[np.ndarray]:
    """Embed a list of image paths in batches; returns list of float32 arrays."""
    all_vecs: list[np.ndarray] = []
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start: batch_start + BATCH_SIZE]
        pil_images: list[Image.Image] = []
        valid_indices: list[int] = []

        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                img.load()
                pil_images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"[vid-kf] skip {p.name}: {e}", file=sys.stderr)

        if not pil_images:
            continue

        try:
            embeddings = list(model.embed(pil_images))
        except Exception as e:
            print(f"[vid-kf] embed batch failed: {e}", file=sys.stderr)
            continue

        for vec in embeddings:
            all_vecs.append(np.asarray(vec, dtype=np.float32))

    return all_vecs


def write_embeddings_atomic(ids: list[str], vectors: np.ndarray) -> None:
    """Write ids+vectors to a temp file then rename to EMBEDDINGS_PATH."""
    # L2-normalise so cosine similarity == dot product (matches existing pipeline)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=EMBEDDINGS_PATH.parent)
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, ids=np.asarray(ids), vectors=vectors)
        os.replace(tmp_path, EMBEDDINGS_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build video keyframes and CLIP embeddings")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N VID records (for smoke testing)")
    args = parser.parse_args()

    t0 = time.time()

    docs = json.loads(INDEX_PATH.read_text())
    vid_docs = [d for d in docs if d.get("type") == "VID"]
    print(f"[vid-kf] {len(vid_docs)} VID records in index")

    # Filter to those with a local video file that exists
    processable: list[tuple[str, Path]] = []
    for d in vid_docs:
        rid = d["id"]
        vl = d.get("video_local", "")
        if not vl:
            continue
        full = ROOT / vl
        if full.exists():
            processable.append((rid, full))
        else:
            print(f"[vid-kf] {rid}: video file not found ({vl}) — skipping")

    print(f"[vid-kf] {len(processable)} VID records with local video files")

    if args.limit is not None:
        processable = processable[: args.limit]
        print(f"[vid-kf] --limit {args.limit}: processing {len(processable)} records")

    if not processable:
        print("[vid-kf] no videos to process — done")
        return

    # Load existing embeddings
    existing_ids, existing_vecs = load_existing_embeddings()
    existing_set = set(existing_ids)

    # Extract keyframes for each video
    new_ids: list[str] = []
    new_kf_paths: list[Path] = []

    for rid, video_path in processable:
        kf_paths = extract_keyframes(video_path, rid)
        for scene_idx, kf_path in enumerate(kf_paths):
            embed_id = f"vid:{rid}:scene{scene_idx:03d}"
            if embed_id in existing_set:
                continue  # already embedded
            new_ids.append(embed_id)
            new_kf_paths.append(kf_path)

    print(f"[vid-kf] {len(new_kf_paths)} new keyframes to embed")

    if not new_kf_paths:
        print("[vid-kf] nothing new to embed")
        return

    print(f"[vid-kf] loading CLIP model: {MODEL_NAME}")
    model = ImageEmbedding(model_name=MODEL_NAME)

    new_vecs = embed_images(model, new_kf_paths)

    if len(new_vecs) != len(new_ids):
        # Trim ids to match successful embeddings
        new_ids = new_ids[: len(new_vecs)]

    print(f"[vid-kf] embedded {len(new_vecs)} keyframes in {time.time()-t0:.1f}s")

    # Merge and write
    if len(existing_vecs) > 0:
        all_vecs = np.vstack([existing_vecs, np.asarray(new_vecs, dtype=np.float32)])
    else:
        all_vecs = np.asarray(new_vecs, dtype=np.float32)

    all_ids = existing_ids + new_ids

    write_embeddings_atomic(all_ids, all_vecs)

    new_total = len(all_ids)
    delta = len(new_ids)
    print(f"[vid-kf] wrote {EMBEDDINGS_PATH}  shape={all_vecs.shape}")
    print(f"[vid-kf] vectors appended: +{delta}  new total: {new_total}")

    # Smoke check
    check = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    assert check["ids"].shape[0] == new_total, "id/vector count mismatch"
    assert check["vectors"].shape[1] == 512, f"unexpected dim {check['vectors'].shape[1]}"
    print(f"[vid-kf] smoke check OK: ids={check['ids'].shape} vectors={check['vectors'].shape}")
    print(f"[vid-kf] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

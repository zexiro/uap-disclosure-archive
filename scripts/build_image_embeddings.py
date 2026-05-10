#!/usr/bin/env python3
"""Embed every IMG record in ui/search-index.json with Qdrant/clip-ViT-B-32-vision
(via fastembed / ONNX) and write ui/image_embeddings.npz.

Output:
  ui/image_embeddings.npz  — keys: ids (object array), vectors (float32 [N,512])

Image path resolution order (stops at first file that exists on disk):
  1. thumb_small        (pre-generated ~280px JPEG — fast to process, already cached)
  2. thumbnail_local[0] (medium-quality JPEG)
  3. primary_local[0]   (full-res PNG/JPEG)

All paths in the index are relative to the project root. The script resolves
them against ROOT (same ancestor directory as this script).

Idempotency: existing ui/image_embeddings.npz is read at start-up; only IDs
not already embedded are processed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastembed import ImageEmbedding
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
OUT_PATH = ROOT / "ui" / "image_embeddings.npz"

MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
BATCH_SIZE = 16


def resolve_image_path(doc: dict) -> Path | None:
    """Return the best local image path for a record, or None if none exist."""
    candidates = []
    if doc.get("thumb_small"):
        candidates.append(doc["thumb_small"])
    for p in doc.get("thumbnail_local") or []:
        candidates.append(p)
    for p in doc.get("primary_local") or []:
        candidates.append(p)
    for rel in candidates:
        full = ROOT / rel
        if full.exists():
            return full
    return None


def load_existing() -> tuple[list[str], list[list[float]]]:
    """Load already-embedded IDs and vectors from disk (idempotency check)."""
    if not OUT_PATH.exists():
        return [], []
    try:
        npz = np.load(OUT_PATH, allow_pickle=True)
        ids = list(npz["ids"].astype(str))
        vecs = npz["vectors"].tolist()
        print(f"[img-embed] loaded {len(ids)} existing embeddings from {OUT_PATH.name}")
        return ids, vecs
    except Exception as e:
        print(f"[img-embed] warning: could not read existing .npz ({e}), starting fresh")
        return [], []


def main():
    t0 = time.time()

    docs = json.loads(INDEX_PATH.read_text())
    img_docs = [d for d in docs if d.get("type") == "IMG"]
    print(f"[img-embed] {len(img_docs)} IMG records in index")

    existing_ids, existing_vecs = load_existing()
    done_set = set(existing_ids)

    # Build work list: records not yet embedded whose image exists on disk.
    todo: list[tuple[str, Path]] = []
    skipped_missing = 0
    for d in img_docs:
        rid = d["id"]
        if rid in done_set:
            continue
        img_path = resolve_image_path(d)
        if img_path is None:
            skipped_missing += 1
            continue
        todo.append((rid, img_path))

    print(f"[img-embed] {len(todo)} new records to embed  "
          f"({len(done_set)} already done, {skipped_missing} skipped — no local image)")

    if not todo:
        print(f"[img-embed] nothing to do — {OUT_PATH} is up to date")
        if existing_ids:
            npz = np.load(OUT_PATH, allow_pickle=True)
            print(f"[img-embed] smoke check: ids={npz['ids'].shape} vectors={npz['vectors'].shape}")
        return

    model = ImageEmbedding(model_name=MODEL_NAME)
    print(f"[img-embed] model: {MODEL_NAME}")

    new_ids: list[str] = []
    new_vecs: list[np.ndarray] = []

    # Process in batches; fastembed.ImageEmbedding.embed() accepts PIL Images.
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start: batch_start + BATCH_SIZE]
        pil_images: list[Image.Image] = []
        batch_ids: list[str] = []

        for rid, path in batch:
            try:
                img = Image.open(path)
                img.load()
                if img.mode not in ("RGB",):
                    img = img.convert("RGB")
                pil_images.append(img)
                batch_ids.append(rid)
            except Exception as e:
                print(f"[img-embed] skip {rid} ({path.name}): {e}", file=sys.stderr)

        if not pil_images:
            continue

        try:
            embeddings = list(model.embed(pil_images))
        except Exception as e:
            print(f"[img-embed] batch embed failed: {e}", file=sys.stderr)
            continue

        for rid, vec in zip(batch_ids, embeddings):
            new_ids.append(rid)
            new_vecs.append(np.asarray(vec, dtype=np.float32))

        done = len(existing_ids) + len(new_ids)
        total = len(existing_ids) + len(todo)
        print(f"[img-embed] {done}/{total}  ({time.time()-t0:.1f}s)", flush=True)

    # Merge with existing
    all_ids = existing_ids + new_ids
    if existing_vecs:
        all_matrix = np.vstack(
            [np.asarray(existing_vecs, dtype=np.float32),
             np.asarray(new_vecs, dtype=np.float32) if new_vecs else np.empty((0, 512), dtype=np.float32)]
        )
    else:
        all_matrix = np.asarray(new_vecs, dtype=np.float32) if new_vecs else np.empty((0, 512), dtype=np.float32)

    # L2-normalise so cosine similarity = dot product (mirrors text embedding pipeline).
    norms = np.linalg.norm(all_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_matrix = all_matrix / norms

    np.savez_compressed(OUT_PATH, ids=np.asarray(all_ids), vectors=all_matrix)
    print(f"[img-embed] wrote {OUT_PATH}  shape={all_matrix.shape}  in {time.time()-t0:.1f}s")

    # Smoke check: reload and verify
    check = np.load(OUT_PATH, allow_pickle=True)
    assert check["ids"].shape[0] == all_matrix.shape[0], "id/vector count mismatch"
    assert check["vectors"].shape[1] == 512, f"unexpected dim {check['vectors'].shape[1]}"
    print(f"[img-embed] smoke check OK: ids={check['ids'].shape} vectors={check['vectors'].shape}")


if __name__ == "__main__":
    main()

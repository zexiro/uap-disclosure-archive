#!/usr/bin/env python3
"""Build a FAISS IndexFlatIP over the CLIP image embeddings.

Reads:   ui/image_embeddings.npz   (ids + L2-normalised 512-dim vectors)
Writes:  ui/image_index.faiss      (FAISS inner-product index)
         ui/image_index_ids.json   (list of record IDs in vector-row order)

Idempotent: skips rebuild if image_index.faiss is newer than image_embeddings.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "ui" / "image_embeddings.npz"
FAISS_PATH = ROOT / "ui" / "image_index.faiss"
IDS_PATH = ROOT / "ui" / "image_index_ids.json"


def main():
    if not EMBEDDINGS_PATH.exists():
        print(f"[img-index] {EMBEDDINGS_PATH} not found — run build_image_embeddings.py first", file=sys.stderr)
        sys.exit(1)

    # Idempotency check: skip if index is already newer than embeddings.
    if FAISS_PATH.exists() and IDS_PATH.exists():
        emb_mtime = EMBEDDINGS_PATH.stat().st_mtime
        idx_mtime = FAISS_PATH.stat().st_mtime
        if idx_mtime >= emb_mtime:
            print(f"[img-index] index is up to date (mtime {idx_mtime:.0f} >= emb mtime {emb_mtime:.0f}) — skipping rebuild")
            # Print stats
            ids = json.loads(IDS_PATH.read_text())
            print(f"[img-index] existing index: {len(ids)} records")
            return

    import faiss  # imported here so the script fails clearly if faiss-cpu missing

    print(f"[img-index] loading embeddings from {EMBEDDINGS_PATH.name} …")
    npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    ids: list[str] = [str(x) for x in npz["ids"]]
    vectors: np.ndarray = npz["vectors"].astype("float32")  # shape (N, 512)

    N, D = vectors.shape
    print(f"[img-index] building IndexFlatIP  N={N}  D={D}")

    index = faiss.IndexFlatIP(D)
    index.add(vectors)

    print(f"[img-index] writing {FAISS_PATH.name} …")
    faiss.write_index(index, str(FAISS_PATH))

    print(f"[img-index] writing {IDS_PATH.name} …")
    IDS_PATH.write_text(json.dumps(ids, indent=None))

    print(f"[img-index] done — {N} vectors  D={D}  ntotal={index.ntotal}")

    # Smoke check
    assert index.ntotal == N
    check_ids = json.loads(IDS_PATH.read_text())
    assert len(check_ids) == N
    print(f"[img-index] smoke check OK")


if __name__ == "__main__":
    main()

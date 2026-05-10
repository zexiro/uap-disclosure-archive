#!/usr/bin/env python3
"""Embed every record in ui/search-index.json with BAAI/bge-small-en-v1.5
(via fastembed / ONNX) and write a single .npz file the /api/ask endpoint
loads at startup for semantic retrieval.

Output:
  ui/embeddings.npz   — keys: ids (object array), vectors (float32 [N,384])
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "ui" / "search-index.json"
OUT_PATH = ROOT / "ui" / "embeddings.npz"

MAX_TEXT_CHARS = 12_000  # cap to keep batches reasonable; bge handles ~512 tokens

def embed_text_for(d: dict) -> str:
    """Build a single string per doc that captures what the model should match against."""
    parts = []
    if d.get("title"):
        parts.append(d["title"])
    if d.get("agency"):
        parts.append(f"Agency: {d['agency']}")
    if d.get("incident_date") and d["incident_date"] != "N/A":
        parts.append(f"Incident date: {d['incident_date']}")
    if d.get("incident_location") and d["incident_location"] != "N/A":
        parts.append(f"Location: {d['incident_location']}")
    if d.get("craft_shape"):
        parts.append(f"Craft: {d['craft_shape']}")
    if d.get("incident_id"):
        parts.append(f"Incident: {d['incident_id']}")
    if d.get("blurb"):
        parts.append(d["blurb"])
    if d.get("text"):
        parts.append(d["text"][:MAX_TEXT_CHARS])
    return " \n ".join(parts)


def main():
    t0 = time.time()
    docs = json.loads(INDEX_PATH.read_text())
    # Skip synthetic IMG records (they're parent-doc fragments, not searchable).
    docs = [d for d in docs if d.get("type") != "IMG"]
    print(f"[embed] {len(docs)} docs to embed")

    model = TextEmbedding()  # downloads bge-small-en-v1.5 once
    print(f"[embed] model: {model.model_name}")

    texts = [embed_text_for(d) for d in docs]
    ids = [d["id"] for d in docs]

    # Stream embed in batches; collect into one matrix.
    vecs = list(model.embed(texts, batch_size=16))
    matrix = np.asarray(vecs, dtype=np.float32)
    # Normalize once so cosine = dot
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    np.savez_compressed(OUT_PATH, ids=np.asarray(ids), vectors=matrix)
    print(f"[embed] wrote {OUT_PATH}  shape={matrix.shape}  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

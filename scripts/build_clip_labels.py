#!/usr/bin/env python3
"""CLIP zero-shot labels — Wave 3 of IMAGE_FORENSICS_PLAN.md.

For every image already embedded in ui/image_embeddings.npz (built by
build_image_embeddings.py using Qdrant/clip-ViT-B-32-vision), compute
cosine similarity against a fixed set of text prompts encoded with the
matching text tower (Qdrant/clip-ViT-B-32-text).

Outputs ui/clip_labels.json keyed by image filename so the lightbox
metadata panel can do a one-step lookup:

    {
      "<filename>.jpg": {
        "model": "Qdrant/clip-ViT-B-32",
        "synthetic_score": 0.32,        # P(syn|image) under CLIP zero-shot softmax
        "synthetic_max":   0.221,       # raw cosine to closest synthetic prompt
        "real_max":        0.276,       # raw cosine to closest real prompt
        "tags": [
          {"label": "document scan", "score": 0.314},
          {"label": "sketch or drawing", "score": 0.198},
          ...
        ]
      }
    }

Framing rule (per IMAGE_FORENSICS_PLAN.md): this is a SIMILARITY SCORE,
not a verdict. The UI surfaces it as such — "X% closer to AI-generated
prompts than real-photo prompts under CLIP zero-shot", with the model
name + prompt set visible. We never say "AI: yes/no".

No new model download, no new pip dep — fastembed already ships the
text tower and the existing .npz has all the image vectors we need.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "ui" / "image_embeddings.npz"
INDEX_PATH = ROOT / "ui" / "search-index.json"
OUT_PATH = ROOT / "ui" / "clip_labels.json"

MODEL_TEXT = "Qdrant/clip-ViT-B-32-text"
# CLIP zero-shot temperature. OpenAI's pre-trained CLIP uses ~100 during
# the contrastive loss, but at inference time that produces extremely
# peaky distributions ("96% synthetic" for slightly-CGI-looking grainy
# aerial photos). Dropping to 30 keeps the relative ordering intact but
# leaves room for nuance in the surfaced probability.
CLIP_TEMPERATURE = 30.0

# Two-class "synthetic vs real" prompt pool. We sum probabilities within
# each class after a temperature-scaled softmax across all 8 prompts;
# multiple paraphrases on each side reduce sensitivity to any single
# prompt wording.
SYNTHETIC_PROMPTS = [
    "an AI-generated synthetic image",
    "a computer-generated CGI render",
    "a stable diffusion image",
    "a midjourney generation",
]
REAL_PROMPTS = [
    "a real photograph of a physical scene",
    "a scanned paper document",
    "a camera photograph",
    "a real-world video frame",
    "an archival black-and-white photograph",
    "an aerial reconnaissance photograph",
    "a low-resolution analog photograph",
    "a vintage film photograph",
]

# Content tags surfaced as descriptive labels. Pick prompts the visitor
# can sanity-check at a glance: if CLIP says "orb of light" and the
# image is clearly a document, the score is doing the heavy lifting
# (and the visitor knows the model isn't infallible).
CONTENT_PROMPTS = [
    "an aircraft",
    "a drone",
    "a weather balloon",
    "an orb of light",
    "a triangular craft",
    "a disc-shaped craft",
    "a cigar-shaped craft",
    "a person",
    "a vehicle",
    "a scanned document page",
    "a hand-drawn sketch",
    "satellite imagery",
    "a blurry photograph",
    "a night sky photograph",
    "a slide from a presentation",
]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _build_id_to_filename(docs: list[dict]) -> dict[str, str]:
    """Map record IDs back to the filename the embedder actually consumed.

    Mirrors build_image_embeddings.resolve_image_path — first existing
    candidate wins, in this order: thumb_small → thumbnail_local → primary_local.
    """
    out: dict[str, str] = {}
    for d in docs:
        if d.get("type") != "IMG":
            continue
        candidates: list[str] = []
        if d.get("thumb_small"):
            candidates.append(d["thumb_small"])
        for k in ("thumbnail_local", "primary_local"):
            v = d.get(k)
            if isinstance(v, list):
                candidates.extend(v)
            elif isinstance(v, str):
                candidates.append(v)
        for rel in candidates:
            if rel and (ROOT / rel).exists():
                out[d["id"]] = Path(rel).name
                break
    return out


def main() -> int:
    if not EMB_PATH.exists():
        print(f"[clip-labels] {EMB_PATH} missing — run build_image_embeddings.py first")
        return 0
    if not INDEX_PATH.exists():
        print(f"[clip-labels] {INDEX_PATH} missing — run build_search_index.py first")
        return 0

    t0 = time.time()
    npz = np.load(EMB_PATH, allow_pickle=True)
    ids = list(npz["ids"].astype(str))
    img_vecs = _normalize(npz["vectors"].astype(np.float32))
    print(f"[clip-labels] loaded {len(ids)} image vectors")

    docs = json.loads(INDEX_PATH.read_text())
    id_to_fname = _build_id_to_filename(docs)
    print(f"[clip-labels] resolved {len(id_to_fname)} record-id → filename mappings")

    # Encode prompts (small, in-process — no batching needed).
    all_prompts = SYNTHETIC_PROMPTS + REAL_PROMPTS + CONTENT_PROMPTS
    encoder = TextEmbedding(model_name=MODEL_TEXT)
    prompt_vecs = _normalize(np.array(list(encoder.embed(all_prompts)), dtype=np.float32))

    # Cosine similarity matrix [N_images, N_prompts]
    sims = img_vecs @ prompt_vecs.T

    n_syn = len(SYNTHETIC_PROMPTS)
    n_real = len(REAL_PROMPTS)
    syn_slice = sims[:, :n_syn]
    real_slice = sims[:, n_syn : n_syn + n_real]
    cont_slice = sims[:, n_syn + n_real :]

    # Temperature-scaled softmax over (synthetic ∪ real) — standard CLIP
    # zero-shot recipe. Probability of "synthetic" = sum of softmax over
    # the synthetic-prompt slice.
    binary = np.concatenate([syn_slice, real_slice], axis=1) * CLIP_TEMPERATURE
    binary -= binary.max(axis=1, keepdims=True)  # numerical stability
    probs = np.exp(binary)
    probs /= probs.sum(axis=1, keepdims=True)
    syn_prob = probs[:, :n_syn].sum(axis=1)

    out: dict[str, dict] = {}
    for i, rid in enumerate(ids):
        fname = id_to_fname.get(rid)
        if not fname:
            continue
        cont_sims = cont_slice[i]
        order = np.argsort(cont_sims)[::-1]
        tags = []
        for j in order[:4]:
            score = float(cont_sims[j])
            if score < 0.18:  # below this cosine the label is basically noise
                continue
            label = CONTENT_PROMPTS[j]
            for prefix in ("an ", "a "):
                if label.startswith(prefix):
                    label = label[len(prefix):]
                    break
            tags.append({"label": label, "score": round(score, 3)})
        out[fname] = {
            "model": "Qdrant/clip-ViT-B-32",
            "synthetic_score": round(float(syn_prob[i]), 3),
            "synthetic_max": round(float(syn_slice[i].max()), 3),
            "real_max": round(float(real_slice[i].max()), 3),
            "tags": tags,
        }

    OUT_PATH.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")))
    elapsed = time.time() - t0
    print(f"[clip-labels] wrote {len(out)} entries → {OUT_PATH.relative_to(ROOT)}  [{elapsed:.1f}s]")

    # Tally — how many entries lean synthetic under this prompt set.
    over = sum(1 for v in out.values() if v["synthetic_score"] > 0.5)
    over_high = sum(1 for v in out.values() if v["synthetic_score"] > 0.7)
    print(f"[clip-labels]   {over} with synthetic_score > 0.5, {over_high} > 0.7")
    return 0


if __name__ == "__main__":
    sys.exit(main())

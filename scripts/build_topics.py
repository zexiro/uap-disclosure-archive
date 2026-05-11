#!/usr/bin/env python3
"""Auto-discover thematic topics across the disclosure archive using BERTopic.

Reuses the precomputed BGE-small embeddings from ui/embeddings.npz so that
no model download is needed at inference time.  HDBSCAN clusters the 384-dim
space; c-TF-IDF produces deterministic keyword labels.

Output
------
ui/topics.json
  {
    "topics": [{"id": 0, "label": "...", "keywords": [...], "size": N,
                "rep_records": ["id1", "id2"]}, ...],
    "by_record": {"<record_id>": <topic_id>}
  }
"""
import json
import sys
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
EMBED_PATH = ROOT / "ui" / "embeddings.npz"
INDEX_PATH = ROOT / "ui" / "search-index.json"
OUT_PATH   = ROOT / "ui" / "topics.json"


def build_doc_text(d: dict) -> str:
    parts = []
    if d.get("title"):
        parts.append(d["title"])
    if d.get("agency"):
        parts.append(d["agency"])
    if d.get("blurb"):
        parts.append(d["blurb"])
    if d.get("text"):
        parts.append(d["text"][:4000])
    return " ".join(parts) or d.get("id", "")


def main():
    # ── Load embeddings ──────────────────────────────────────────────────────
    data = np.load(EMBED_PATH, allow_pickle=True)
    ids: list[str] = data["ids"].tolist()
    vectors: np.ndarray = data["vectors"]            # (N, 384) float32, L2-normalised

    # ── Build matching document texts from search-index ──────────────────────
    all_docs = json.loads(INDEX_PATH.read_text())
    doc_map  = {d["id"]: d for d in all_docs}

    documents = [build_doc_text(doc_map[i]) if i in doc_map else i for i in ids]

    n = len(ids)
    print(f"[topics] {n} documents with embeddings")

    # ── BERTopic — pass precomputed embeddings; no re-embedding ─────────────
    # min_cluster_size=10 for ~150 docs gives ~8-15 topics typically.
    # If corpus grows to 1000+ docs, raise to 25-30.
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    # CountVectorizer with English stop words so c-TF-IDF surfaces domain terms.
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )

    topic_model = BERTopic(
        embedding_model=None,          # no internal embedding; we supply vectors
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=None,                # let HDBSCAN decide
        top_n_words=10,
        verbose=False,
        calculate_probabilities=False,
    )

    topic_ids, _ = topic_model.fit_transform(documents, embeddings=vectors)

    # ── Build output ─────────────────────────────────────────────────────────
    by_record: dict[str, int] = {ids[i]: int(topic_ids[i]) for i in range(n)}

    topic_info = topic_model.get_topic_info()
    topics_out = []

    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        label_words = topic_model.get_topic(tid)         # [(word, score), ...]

        if tid == -1:
            label = "uncategorised"
            keywords = []
        else:
            # c-TF-IDF top-3 keywords joined as a phrase
            kws = [w for w, _ in label_words[:3]] if label_words else []
            label = " · ".join(kws) if kws else f"topic_{tid}"
            keywords = [w for w, _ in label_words[:8]] if label_words else []

        # Representative record IDs for this topic
        size   = int(row["Count"])
        recs   = [ids[i] for i, t in enumerate(topic_ids) if t == tid]
        rep_records = recs[:5]

        topics_out.append({
            "id": tid,
            "label": label,
            "keywords": keywords,
            "size": size,
            "rep_records": rep_records,
        })

    # Sort: -1 (noise) last, others by size descending
    topics_out.sort(key=lambda t: (t["id"] == -1, -t["size"]))

    output = {"topics": topics_out, "by_record": by_record}
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"[topics] wrote {OUT_PATH}")

    # ── Smoke test summary ────────────────────────────────────────────────────
    named = [t for t in topics_out if t["id"] != -1]
    noise = next((t for t in topics_out if t["id"] == -1), None)
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {len(named)} named topics + noise cluster")
    print(f"Records assigned to named topics: {sum(t['size'] for t in named)}")
    if noise:
        print(f"Uncategorised (noise): {noise['size']}")
    print(f"\nTop 5 topics by size:")
    for t in named[:5]:
        print(f"  [{t['id']:2d}] size={t['size']:3d}  \"{t['label']}\"  kws={t['keywords'][:5]}")
    print("="*60)


if __name__ == "__main__":
    main()

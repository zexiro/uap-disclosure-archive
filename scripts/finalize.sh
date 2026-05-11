#!/usr/bin/env bash
# Run after `python3 scripts/download.py && python3 scripts/ocr.py` to:
#   - rebuild the Obsidian vault notes (now with extracted text)
#   - rebuild the search index (now with extracted text)
#   - print a summary of the corpus.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Re-running download to retry any failures…"
python3 scripts/download.py || true

echo "==> Re-running OCR on any PDFs added by retry…"
python3 scripts/ocr.py || true

echo "==> Computing relationship links (text + image)…"
python3 scripts/build_links.py

echo "==> Extracting in-PDF citation graph…"
python3 scripts/extract_citations.py

echo "==> Extracting named entities (spaCy NER over OCR text)…"
python3 scripts/extract_entities.py || true

echo "==> Building knowledge graph communities (Louvain + centrality)…"
python3 scripts/build_communities.py || true

echo "==> Building vault notes…"
rm -rf vault/Releases vault/Index vault/README.md
python3 scripts/build_vault.py

echo "==> Building search index…"
python3 scripts/build_search_index.py

echo "==> Building CLIP image embeddings for IMG records…"
python3 scripts/build_image_embeddings.py || true

echo "==> Building FAISS image similarity index…"
python3 scripts/build_image_index.py || true

echo "==> Discovering topics (BERTopic over BGE embeddings)…"
python3 scripts/build_topics.py || true

echo "==> Augmenting features (shapes, incidents, graph layout)…"
python3 scripts/build_features.py

echo "==> Emitting public API + Atom feed…"
python3 scripts/build_api.py

echo
echo "===== Corpus summary ====="
echo "PDFs:    $(ls raw/docs   2>/dev/null | grep -c '\.pdf$')"
echo "Images:  $(ls raw/images 2>/dev/null | grep -cE '\.(jpe?g|png|gif|webp)$')"
echo "Videos:  $(ls raw/videos 2>/dev/null | grep -c '\.mp4$')"
echo "Texts:   $(ls raw/text   2>/dev/null | grep -c '\.txt$')"
echo "Disk:    $(du -sh raw 2>/dev/null | awk '{print $1}')"
echo "Vault notes: $(ls vault/Releases 2>/dev/null | wc -l | tr -d ' ')"
echo
[ -f raw/download_errors.log ] && echo "Download errors: $(wc -l <raw/download_errors.log)" || true
[ -f raw/ocr_errors.log ] && echo "OCR errors: $(wc -l <raw/ocr_errors.log)" || true

echo
echo "→ Open ./vault in Obsidian, or run ‘make serve’ for the search UI."

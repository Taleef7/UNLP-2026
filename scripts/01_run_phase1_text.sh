#!/bin/bash
# Run Phase 1 text extraction and BM25 retrieval (CPU only, no GPU needed)
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026

echo "=== Phase 1: Text Retrieval ==="

echo "[1.1] Extracting text from PDFs..."
python scripts/01_extract_text.py

echo "[1.2] Rendering page images (background)..."
python scripts/01_render_pages.py &
RENDER_PID=$!

echo "[1.3] Building BM25 index and evaluating..."
python scripts/01_bm25_retrieval.py

echo "Waiting for page rendering (PID=$RENDER_PID)..."
wait $RENDER_PID
echo "Page rendering done."

echo "=== Phase 1 text pipeline complete ==="
echo "Next: run 01_bge_m3_retrieval.py (GPU) and 01_colpali_embed.py (GPU)"

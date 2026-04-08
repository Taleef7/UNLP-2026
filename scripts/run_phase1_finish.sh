#!/bin/bash
# Run remaining Phase 1 steps after colpali embedding finishes
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache
export PYTHONUNBUFFERED=1

echo "=== Phase 1 completion: ColPali retrieve + Hybrid fusion ==="

# Wait for embedding file to appear
EMB_META="embeddings/colsmol/metadata.json"
echo "Waiting for $EMB_META..."
while [ ! -f "$EMB_META" ]; do
    sleep 10
    echo -n "."
done
echo ""
echo "Embedding complete: $(wc -l < $EMB_META) pages in metadata"

# Run ColPali retrieval
echo "[1.6b] ColPali retrieval..."
MODEL=colsmol python scripts/01_colpali_retrieve.py 2>&1 | tee logs/colpali_retrieve.log

# Run hybrid fusion (requires bm25, bge_m3, and colpali retrieval)
echo "[1.7] Hybrid fusion (RRF)..."
python scripts/01_hybrid_fusion.py 2>&1 | tee logs/hybrid_fusion.log

echo "=== Phase 1 complete ==="

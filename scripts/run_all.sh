#!/bin/bash
# Master run script — execute phases in order
# Usage: bash scripts/run_all.sh [--phase 0|1|2|3]
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache

PHASE=${1:---phase all}
shift || true
PHASE_NUM=${2:-all}

echo "=== UNLP 2026 Pipeline ==="
echo "Phase: $PHASE_NUM"

run_phase0() {
    echo "[Phase 0] Setup and data audit"
    python scripts/00_profile_data.py
    python scripts/00_eval_harness.py
    python scripts/00_create_splits.py
    echo "Phase 0 complete."
}

run_phase1_cpu() {
    echo "[Phase 1 CPU] Text extraction + BM25"
    python scripts/01_extract_text.py
    python scripts/01_render_pages.py
    python scripts/01_bm25_retrieval.py
}

run_phase1_gpu() {
    echo "[Phase 1 GPU] Dense retrieval"
    python scripts/01_bge_m3_retrieval.py
    MODEL=colsmol python scripts/01_colpali_embed.py
    MODEL=colsmol python scripts/01_colpali_retrieve.py
    python scripts/01_hybrid_text.py
    python scripts/01_hybrid_fusion.py
}

run_phase2() {
    echo "[Phase 2] MCQ scoring"
    python scripts/02_mcq_logprobs.py
    python scripts/02_prompt_ablation.py
}

run_phase3() {
    echo "[Phase 3] Pipeline + submission"
    # Use best retrieval (BGE-M3)
    python scripts/03_pipeline.py \
        --retrieval precomputed \
        --precomputed_retrieval outputs/bge_m3_retrieval.json \
        --strategy direct \
        --output outputs/submission.csv

    echo "Evaluating submission..."
    python scripts/00_eval_harness.py outputs/submission.csv

    python scripts/03_profile_resources.py
}

case "$PHASE_NUM" in
    0) run_phase0 ;;
    1) run_phase1_cpu && run_phase1_gpu ;;
    2) run_phase2 ;;
    3) run_phase3 ;;
    all)
        run_phase0
        run_phase1_cpu
        run_phase1_gpu
        run_phase2
        run_phase3
        ;;
    *)
        echo "Usage: $0 --phase [0|1|2|3|all]"
        exit 1
        ;;
esac

echo "=== Done ==="

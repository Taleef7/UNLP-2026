#!/bin/bash
# Run Phase 2 (MCQ scoring) and Phase 3 (pipeline + submission)
# Assumes llama-cpp-python is installed and MamayLM GGUF is present
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache
export PYTHONUNBUFFERED=1

echo "=== Phase 2: MCQ Scoring ==="

# Verify llama-cpp and GGUF
python -c "from llama_cpp import Llama; print('llama-cpp OK')"
ls models/mamaylm/*.gguf

echo "[2.3] MCQ logprobs (oracle retrieval)..."
python scripts/02_mcq_logprobs.py 2>&1 | tee logs/mcq_eval.log

echo "[2.4] Prompt ablation..."
python scripts/02_prompt_ablation.py 2>&1 | tee logs/prompt_ablation.log

echo "=== Phase 3: Pipeline + Submission ==="

echo "[3.1] Full pipeline (BGE-M3 retrieval)..."
python scripts/03_pipeline.py \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_direct.csv \
    2>&1 | tee logs/pipeline.log

echo "Evaluating submission..."
python scripts/00_eval_harness.py outputs/submission_bge_direct.csv

echo "[3.3] Resource profiling..."
python scripts/03_profile_resources.py 2>&1 | tee logs/resource_profile.log

echo "=== Done ==="
echo "Submission: outputs/submission_bge_direct.csv"

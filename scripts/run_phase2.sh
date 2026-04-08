#!/bin/bash
# Phase 2: MCQ evaluation pipeline
# Run after llama-cpp-python is installed

set -e
BASE=/scratch/gilbreth/tamst01/unlp2026
cd "$BASE"

# Add local_packages to PYTHONPATH if needed
if [ -d "$BASE/local_packages" ]; then
    export PYTHONPATH="$BASE/local_packages:$PYTHONPATH"
    echo "PYTHONPATH set to include local_packages"
fi

# Verify llama-cpp works
echo "=== Verifying llama-cpp-python ==="
python scripts/verify_mcq.py 2>&1 | tee logs/verify_mcq.log
if [ $? -ne 0 ]; then
    echo "ERROR: llama-cpp-python verification failed"
    exit 1
fi
echo "=== verify_mcq passed ==="

# MCQ evaluation with oracle retrieval (val split)
echo ""
echo "=== Phase 2.3: MCQ Logprobs/Greedy (oracle retrieval, val split) ==="
python scripts/02_mcq_logprobs.py 2>&1 | tee logs/mcq_logprobs.log
echo "=== MCQ eval done ==="

# Prompt ablation (val split)
echo ""
echo "=== Phase 2.4: Prompt Ablation (val split) ==="
python scripts/02_prompt_ablation.py 2>&1 | tee logs/prompt_ablation.log
echo "=== Prompt ablation done ==="

# End-to-end pipeline with BGE-M3 retrieval (val split)
echo ""
echo "=== Phase 3.1: End-to-End Pipeline (BGE-M3, val split) ==="
python scripts/03_pipeline.py \
    --questions data/splits/val.csv \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_val.csv \
    2>&1 | tee logs/pipeline_val.log
echo "Validating..."
python scripts/validate_submission.py outputs/submission_bge_val.csv
echo "Evaluating val submission..."
python scripts/00_eval_harness.py outputs/submission_bge_val.csv 2>&1 | tee logs/eval_val.log
echo "=== Val pipeline done ==="

# End-to-end pipeline with BGE-M3 retrieval (full dev set)
echo ""
echo "=== Phase 3.1: End-to-End Pipeline (BGE-M3, full dev) ==="
python scripts/03_pipeline.py \
    --questions data/dev_questions.csv \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_dev.csv \
    2>&1 | tee logs/pipeline_dev.log
python scripts/validate_submission.py outputs/submission_bge_dev.csv
echo "Evaluating full dev submission..."
python scripts/00_eval_harness.py outputs/submission_bge_dev.csv 2>&1 | tee logs/eval_dev.log
echo "=== Dev pipeline done ==="

echo ""
echo "=== All Phase 2-3 steps complete ==="
echo "Check outputs/:"
ls -lh outputs/submission_*.csv outputs/mcq_results.json outputs/prompt_ablation_results.json 2>/dev/null

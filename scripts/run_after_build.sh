#!/bin/bash
# Run immediately after GPU build completes
# Checks if build is done, then runs the full Phase 2+3 pipeline

set -e
BASE=/scratch/gilbreth/tamst01/unlp2026
cd "$BASE"

# Wait for build to complete
BUILD_LOG="$BASE/logs/llama_cpp_build_final.log"
echo "Waiting for GPU build to complete..."
while ! grep -q "Successfully installed" "$BUILD_LOG" 2>/dev/null; do
    echo "  Build still running... $(date +%H:%M:%S)"
    sleep 30
done
echo "Build complete!"

# Kill CPU MCQ eval (now unneeded)
pkill -f "02_mcq_logprobs.py" 2>/dev/null && echo "Killed CPU MCQ eval"

# Set PYTHONPATH
export PYTHONPATH="$BASE/local_packages:$PYTHONPATH"

# Verify
echo "=== Verifying llama-cpp-python GPU ==="
python3 -c "
import sys
sys.path.insert(0, '$BASE/local_packages')
from llama_cpp import Llama
from pathlib import Path
import torch
gguf = list(Path('$BASE/models/mamaylm').glob('*.gguf'))[0]
llm = Llama(model_path=str(gguf), n_gpu_layers=-1, n_ctx=64, verbose=False)
resp = llm('A. Київ\nВідповідь:', max_tokens=1, temperature=0.0, stop=['\n'])
print('GPU test OK:', repr(resp['choices'][0]['text']))
vram = torch.cuda.memory_allocated() / 1024**3
print(f'GPU VRAM used: {vram:.2f}GB')
"
echo "=== Verify passed ==="

# Run MCQ eval on val split (oracle retrieval)
echo ""
echo "=== Phase 2.3: MCQ eval (oracle, val split) ==="
python3 scripts/02_mcq_logprobs.py 2>&1 | tee logs/mcq_gpu.log
echo "=== MCQ eval done ==="

# Run prompt ablation on val split
echo ""
echo "=== Phase 2.4: Prompt ablation (val split) ==="
python3 scripts/02_prompt_ablation.py 2>&1 | tee logs/prompt_ablation.log
echo "=== Ablation done ==="

# End-to-end pipeline on val split
echo ""
echo "=== Phase 3.1: E2E pipeline (BGE-M3, val split) ==="
python3 scripts/03_pipeline.py \
    --questions data/splits/val.csv \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_val.csv \
    2>&1 | tee logs/pipeline_val.log
python3 scripts/validate_submission.py outputs/submission_bge_val.csv
python3 scripts/00_eval_harness.py outputs/submission_bge_val.csv 2>&1 | tee logs/eval_val.log
echo "=== Val pipeline done ==="

# End-to-end pipeline on full dev set
echo ""
echo "=== Phase 3.1: E2E pipeline (BGE-M3, full dev) ==="
python3 scripts/03_pipeline.py \
    --questions data/dev_questions.csv \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_dev.csv \
    2>&1 | tee logs/pipeline_dev.log
python3 scripts/validate_submission.py outputs/submission_bge_dev.csv
python3 scripts/00_eval_harness.py outputs/submission_bge_dev.csv 2>&1 | tee logs/eval_dev.log
echo "=== Full dev pipeline done ==="

echo ""
echo "=== PHASE 2+3 COMPLETE ==="
echo "Key outputs:"
ls -lh outputs/submission_*.csv outputs/mcq_results.json outputs/prompt_ablation.json 2>/dev/null
cat logs/eval_val.log | grep "COMPOSITE"
cat logs/eval_dev.log | grep "COMPOSITE"

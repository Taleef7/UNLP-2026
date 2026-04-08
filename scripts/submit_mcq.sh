#!/bin/bash
#SBATCH --job-name=unlp-mcq
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gilbreth/tamst01/unlp2026/logs/%j_mcq.out
#SBATCH --error=/scratch/gilbreth/tamst01/unlp2026/logs/%j_mcq.err

cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Run MCQ logprobs strategy comparison
python scripts/02_mcq_logprobs.py

# Run prompt ablation
python scripts/02_prompt_ablation.py

# Run full pipeline with best retrieval
python scripts/03_pipeline.py \
    --retrieval precomputed \
    --precomputed_retrieval outputs/bge_m3_retrieval.json \
    --strategy direct \
    --output outputs/submission_bge_direct.csv

# Evaluate
python scripts/00_eval_harness.py outputs/submission_bge_direct.csv

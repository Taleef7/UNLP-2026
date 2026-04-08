#!/bin/bash
#SBATCH --job-name=unlp-bge-m3
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gilbreth/tamst01/unlp2026/logs/%j_bge_m3.out
#SBATCH --error=/scratch/gilbreth/tamst01/unlp2026/logs/%j_bge_m3.err

cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python scripts/01_bge_m3_retrieval.py

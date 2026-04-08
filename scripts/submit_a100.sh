#!/bin/bash
#SBATCH --job-name=unlp2026
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=/scratch/gilbreth/tamst01/unlp2026/logs/%j.out
#SBATCH --error=/scratch/gilbreth/tamst01/unlp2026/logs/%j.err

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache
cd /scratch/gilbreth/tamst01/unlp2026

python scripts/$1

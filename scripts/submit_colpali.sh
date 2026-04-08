#!/bin/bash
#SBATCH --job-name=unlp-colpali
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gilbreth/tamst01/unlp2026/logs/%j_colpali.out
#SBATCH --error=/scratch/gilbreth/tamst01/unlp2026/logs/%j_colpali.err

cd /scratch/gilbreth/tamst01/unlp2026
export HF_HOME=/scratch/gilbreth/tamst01/unlp2026/cache/hf_cache

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

MODEL=${MODEL:-colsmol}
echo "Running ColPali embedding with MODEL=$MODEL"

python scripts/01_colpali_embed.py
echo "Embedding done. Running retrieval evaluation..."
python scripts/01_colpali_retrieve.py

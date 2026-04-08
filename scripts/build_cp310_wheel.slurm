#!/bin/bash
#SBATCH --job-name=llama_cp310_wheel
#SBATCH --partition=a10
#SBATCH --account=pfw-cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gilbreth/tamst01/unlp2026/logs/slurm_cp310_wheel_%j.log
#SBATCH --error=/scratch/gilbreth/tamst01/unlp2026/logs/slurm_cp310_wheel_%j.log

set -e
WHEELS=/scratch/gilbreth/tamst01/unlp2026/kaggle_datasets/unlp2026-wheels
CONDA_PY310=/scratch/gilbreth/tamst01/conda_py310

echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader), $(date) ==="

# Load CUDA module
module load cuda/12.6.0 2>/dev/null || module load cuda 2>/dev/null
echo "CUDA: $(nvcc --version | grep release)"

# Use our Python 3.10 conda env
PYTHON=$CONDA_PY310/bin/python
PIP=$CONDA_PY310/bin/pip

echo "Python: $($PYTHON --version)"
echo "Building llama-cpp-python==0.3.16 for sm_60 (P100)..."

export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA_GRAPHS=OFF"
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK}"

$PIP wheel llama-cpp-python==0.3.16 \
    --no-cache-dir --no-deps \
    --wheel-dir "$WHEELS"

echo "=== Build complete. Wheels: ==="
ls -lh "$WHEELS"/*.whl

# Verify sm_60 target in new wheel
NEW_WHL=$(ls "$WHEELS"/llama_cpp_python-0.3.16-cp310-*.whl 2>/dev/null | head -1)
if [ -n "$NEW_WHL" ]; then
    echo "Verifying sm_60 target in $NEW_WHL..."
    $PIP install --target /tmp/verify_cp310 "$NEW_WHL" --no-deps -q
    strings /tmp/verify_cp310/llama_cpp/lib/libggml-cuda.so 2>/dev/null | grep "\.target" | head -3
    rm -rf /tmp/verify_cp310
    echo "=== DONE ==="
else
    echo "ERROR: cp310 wheel not found after build"
    exit 1
fi

#!/bin/bash
#SBATCH -J llamacpp_build
#SBATCH -A pfw-cs
#SBATCH -p a10
#SBATCH --qos=standby
#SBATCH -N 1 -n 1 -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 1:30:00
#SBATCH -o ${REPO_ROOT:-$PWD}/outputs/llamacpp_build_%j.log
#SBATCH -e ${REPO_ROOT:-$PWD}/outputs/llamacpp_build_%j.log
#
# Rebuild llama-cpp-python 0.3.16 with CUDA, FOR THIS CLUSTER'S CPU.
#
# WHY. The vendored wheel (kaggle_datasets/unlp2026-wheels, and the pruned copy that used to be in
# local_packages/) was compiled on Kaggle's Intel hardware: `libggml-cpu.so` contains 6109 AVX-512
# (zmm) instructions. Every Gilbreth GPU node is AMD Zen 3 -- a10 = EPYC 7313, a100-80gb = EPYC 7543 --
# and NONE of them has avx512f. Loading the model dies with `Illegal instruction (core dumped)`.
#
# So the system that produced our headline composite CANNOT BE EXECUTED ON THIS CLUSTER AT ALL, in the
# binary form it shipped in. That is a reproducibility fact worth stating plainly in the paper: the
# result was bound not just to a model and a seed we never recorded, but to a CPU instruction set.
#
# CAVEAT WE MUST DISCLOSE. A locally-built llama.cpp is not the same binary as the Kaggle one. With
# n_gpu_layers=-1 the arithmetic runs on CUDA, so the CPU backend is not doing the model math -- but
# this is a genuine deviation and it is not bit-identical. For the question we are actually asking it
# does not matter: sigma across seeds is a property of temperature-0.5 sampling, not of the CPU
# dispatch path. For any ABSOLUTE comparison against 0.8722 it matters, and we say so.
#
# Pin 0.3.16 to match the version the pipeline shipped with (local_packages/llama_cpp_python-0.3.16).
set -eu
cd ${REPO_ROOT:-$PWD}
module load cuda/12.6.0 2>/dev/null || true
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
unset PYTHONPATH

echo "[build] node=$(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
echo "[build] nvcc: $(which nvcc 2>/dev/null || echo none)"
echo "[build] target: local_packages/ (where pipeline_shared.py:40 looks)"

# Native arch: build for the CPU we will actually run on, not the one the wheel assumed.
export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=on -DCMAKE_CUDA_ARCHITECTURES=86"
export FORCE_CMAKE=1

rm -rf local_packages/llama_cpp local_packages/llama_cpp_python-0.3.16.dist-info
python3 -m pip install --no-binary llama-cpp-python --no-cache-dir \
    --target local_packages --upgrade --no-deps "llama-cpp-python==0.3.16"

echo "[build] verifying the CPU backend no longer uses AVX-512:"
for so in local_packages/llama_cpp/lib/libggml-cpu*.so; do
    echo "   $(basename "$so"): zmm=$(objdump -d "$so" 2>/dev/null | grep -c zmm || echo ?)"
done
python3 - <<'PY'
import sys; sys.path.insert(0, "local_packages")
import llama_cpp
print("[build] llama_cpp:", llama_cpp.__file__, "| has Llama:", hasattr(llama_cpp, "Llama"))
PY
echo "[build] done $(date)"

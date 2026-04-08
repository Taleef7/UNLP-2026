#!/bin/bash
# Phase 0.2: Install all Python dependencies
set -euo pipefail

echo "=== Installing UNLP 2026 dependencies ==="

# PyMuPDF for PDF processing
pip install pymupdf

# ColPali for visual retrieval
pip install colpali-engine

# llama-cpp-python with CUDA support (sm_86 for A30/A100)
# For P100 (sm_60), use CUDAARCHS="60"
CUDA_ARCH=${CUDA_ARCH:-"86"}
echo "Compiling llama-cpp-python for CUDA arch sm_$CUDA_ARCH ..."
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
    pip install llama-cpp-python --no-cache-dir --verbose 2>&1 | tail -20

echo "=== Verifying imports ==="
python -c "import fitz; print(f'PyMuPDF: {fitz.__version__}')"
python -c "from colpali_engine.models import ColSmol; print('ColPali: OK')"
python -c "from llama_cpp import Llama; print('llama-cpp-python: OK')"
python -c "from FlagEmbedding import BGEM3FlagModel; print('FlagEmbedding/BGE-M3: OK')"
python -c "from rank_bm25 import BM25Okapi; print('rank-bm25: OK')"
python -c "import faiss; print(f'faiss: {faiss.__version__}')"

echo "=== All dependencies installed ==="

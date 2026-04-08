#!/bin/bash
# Prepare Kaggle dataset directories for upload
# Run after GPU build completes and results are verified

set -e
BASE=/scratch/gilbreth/tamst01/unlp2026
KAGGLE_PREP="$BASE/kaggle_datasets"
HF_CACHE="$BASE/cache/hf_cache"

echo "=== Preparing Kaggle datasets ==="

# 1. MamayLM GGUF
echo ""
echo "[1/3] MamayLM GGUF..."
mkdir -p "$KAGGLE_PREP/mamaylm-gemma3-12b-gguf"
cp "$BASE/models/mamaylm/MamayLM-Gemma-3-12B-IT-v1.0.Q4_K_M.gguf" \
   "$KAGGLE_PREP/mamaylm-gemma3-12b-gguf/"
echo "  Copied: $(ls -lh $KAGGLE_PREP/mamaylm-gemma3-12b-gguf/*.gguf | awk '{print $5, $9}')"

# 2. BGE-M3 model (from HF cache)
echo ""
echo "[2/3] BGE-M3 model..."
BGE_SNAPSHOT=$(ls "$HF_CACHE/models--BAAI--bge-m3/snapshots/" 2>/dev/null | head -1)
if [ -z "$BGE_SNAPSHOT" ]; then
    echo "ERROR: BGE-M3 not found in HF cache"
    exit 1
fi
BGE_SRC="$HF_CACHE/models--BAAI--bge-m3/snapshots/$BGE_SNAPSHOT"
mkdir -p "$KAGGLE_PREP/bge-m3"
# Copy only the essential files (exclude ONNX and images)
cp "$BGE_SRC/config.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/tokenizer.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/tokenizer_config.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/special_tokens_map.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/sentencepiece.bpe.model" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/sentence_bert_config.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/modules.json" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/colbert_linear.pt" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/sparse_linear.pt" "$KAGGLE_PREP/bge-m3/"
cp "$BGE_SRC/pytorch_model.bin" "$KAGGLE_PREP/bge-m3/"
# Copy 1_Pooling directory
cp -r "$BGE_SRC/1_Pooling" "$KAGGLE_PREP/bge-m3/"
echo "  BGE-M3 size: $(du -sh $KAGGLE_PREP/bge-m3/ | awk '{print $1}')"

# 3. llama-cpp-python wheel for P100 (sm_60)
# Note: if pre-compiled wheel not available, Kaggle will compile from source (~15 min)
echo ""
echo "[3/3] llama-cpp-python wheels..."
mkdir -p "$KAGGLE_PREP/unlp2026-wheels"

# Check if we have a wheel built for sm_60
SM60_WHEEL=$(find "$BASE" -name "llama_cpp_python*sm60*.whl" 2>/dev/null | head -1)
if [ -n "$SM60_WHEEL" ]; then
    cp "$SM60_WHEEL" "$KAGGLE_PREP/unlp2026-wheels/"
    echo "  Copied sm_60 wheel: $(basename $SM60_WHEEL)"
else
    echo "  No pre-built sm_60 wheel found."
    echo "  Kaggle notebook will compile from source (adds ~15 min to runtime)."
    echo "  To build: CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=60' pip wheel llama-cpp-python"
fi

echo ""
echo "=== Dataset summary ==="
du -sh "$KAGGLE_PREP"/*/
echo ""
echo "=== Kaggle upload commands ==="
echo "# (requires kaggle CLI configured with API key)"
echo "kaggle datasets create -p $KAGGLE_PREP/mamaylm-gemma3-12b-gguf --dir-mode zip"
echo "kaggle datasets create -p $KAGGLE_PREP/bge-m3 --dir-mode zip"
echo "kaggle datasets create -p $KAGGLE_PREP/unlp2026-wheels --dir-mode zip"

#!/bin/bash
# Upload datasets to Kaggle
# Prerequisites: kaggle.json API key at ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/settings -> API -> Create New Token

set -e
BASE=/scratch/gilbreth/tamst01/unlp2026/kaggle_datasets

# Check API key
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found"
    echo "Go to https://www.kaggle.com/settings -> API -> Create New Token"
    echo "Then: mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "=== Uploading MamayLM GGUF (6.8GB) ==="
kaggle datasets create -p "$BASE/mamaylm-gemma3-12b-gguf" --dir-mode zip

echo "=== Uploading BGE-M3 (2.2GB) ==="
kaggle datasets create -p "$BASE/bge-m3" --dir-mode zip

echo "=== Uploading llama-cpp wheels ==="
kaggle datasets create -p "$BASE/unlp2026-wheels" --dir-mode zip

echo "=== Done! Check: https://www.kaggle.com/datasets ==="

#!/usr/bin/env python3
"""Phase 2.1: Download MamayLM GGUF and ColSmol model weights.

MamayLM-12B Q4_K_M: ~7.3GB GGUF for llama-cpp-python
ColSmol-500M: <1GB for Kaggle P100
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")


MODELS = {
    "mamaylm_gguf": {
        "repo_id": "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0-GGUF",
        "filename": None,  # auto-detect Q4_K_M file
        "filename_pattern": "Q4_K_M",
        "local_dir": BASE / "models" / "mamaylm",
        "description": "MamayLM-Gemma-3-12B Q4_K_M GGUF for llama-cpp-python",
    },
    "mamaylm_4b_gguf": {
        "repo_id": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF",
        "filename": None,
        "filename_pattern": "Q4_K_M",
        "local_dir": BASE / "models" / "mamaylm",
        "description": "MamayLM-Gemma-3-4B Q4_K_M GGUF (faster, smaller)",
    },
    "colsmol": {
        "repo_id": "vidore/colSmol-500M",
        "local_dir": BASE / "models" / "colsmol",
        "description": "ColSmol-500M for visual page retrieval (<1GB)",
    },
    "bge_m3": {
        "repo_id": "BAAI/bge-m3",
        "local_dir": BASE / "models" / "bge-m3",
        "description": "BGE-M3 multilingual embeddings",
    },
}


def find_gguf_filename(repo_id, pattern):
    """List repo files and find one matching pattern."""
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(repo_id))
    gguf_files = [f for f in files if f.endswith(".gguf") and pattern in f]
    if not gguf_files:
        raise ValueError(f"No GGUF file matching '{pattern}' in {repo_id}. Files: {files}")
    # Prefer exact Q4_K_M over others
    gguf_files.sort()
    print(f"  Found GGUF files: {gguf_files}")
    return gguf_files[0]


def download_gguf(repo_id, filename, local_dir, filename_pattern=None):
    """Download a single GGUF file."""
    local_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect filename if not specified
    if filename is None:
        pattern = filename_pattern or "Q4_K_M"
        filename = find_gguf_filename(repo_id, pattern)

    out_path = local_dir / filename
    if out_path.exists():
        print(f"  Already exists: {out_path} ({out_path.stat().st_size / 1e9:.1f}GB)")
        return out_path

    print(f"  Downloading {repo_id}/{filename}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    print(f"  Downloaded to {path} ({Path(path).stat().st_size / 1e9:.1f}GB)")
    return Path(path)


def download_model(repo_id, local_dir):
    """Download a full model repo."""
    local_dir.mkdir(parents=True, exist_ok=True)
    # Check if already downloaded
    if any(local_dir.iterdir()):
        print(f"  Already exists: {local_dir}")
        return

    print(f"  Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin", "*.pt"],  # Prefer safetensors
    )
    print(f"  Downloaded {repo_id}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()) + ["all"],
                        default=["all"])
    args = parser.parse_args()

    to_download = list(MODELS.keys()) if "all" in args.models else args.models

    for name in to_download:
        config = MODELS[name]
        print(f"\n[{name}] {config['description']}")

        if "filename" in config or "filename_pattern" in config:
            download_gguf(config["repo_id"], config.get("filename"),
                          config["local_dir"], config.get("filename_pattern"))
        else:
            download_model(config["repo_id"], config["local_dir"])

    print("\nDownload complete.")
    print("\nModel sizes:")
    for name in to_download:
        local_dir = MODELS[name]["local_dir"]
        if local_dir.exists():
            total = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            print(f"  {name}: {total / 1e9:.2f}GB")


if __name__ == "__main__":
    main()

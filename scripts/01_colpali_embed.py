#!/usr/bin/env python3
"""Phase 1.6: ColSmol/ColQwen2 visual page embedding for retrieval.

Uses ColSmol-500M (default, fits P100) or ColQwen2-v1.0 (better, for HPC).
Embeds all page images and saves multi-vector embeddings.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
EMB_DIR = BASE / "embeddings"

os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

# Model selection: set via env var MODEL=colsmol or MODEL=colqwen2
MODEL_NAME = os.environ.get("MODEL", "colsmol")

MODEL_IDS = {
    "colsmol": "vidore/colSmol-500M",
    "colqwen2": "vidore/colqwen2-v1.0",
}


def get_model_and_processor(model_name):
    # ColSmol-500M uses ColIdefics3 (SmolVLM-based architecture)
    # ColQwen2 uses ColQwen2
    if model_name == "colsmol":
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        model_id = MODEL_IDS["colsmol"]
        local_path = BASE / "models" / "colsmol"
        load_path = str(local_path) if (local_path.exists() and any(local_path.iterdir())) else model_id
        model = ColIdefics3.from_pretrained(
            load_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        processor = ColIdefics3Processor.from_pretrained(load_path)
    else:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        model_id = MODEL_IDS["colqwen2"]
        local_path = BASE / "models" / "colqwen2"
        load_path = str(local_path) if (local_path.exists() and any(local_path.iterdir())) else model_id
        model = ColQwen2.from_pretrained(
            load_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        processor = ColQwen2Processor.from_pretrained(load_path)

    return model, processor


def embed_pages_colpali(model, processor, pages_meta, batch_size=4):
    """Embed page images using ColPali model."""
    from PIL import Image

    all_embeddings = []  # list of tensors [n_patches, hidden]
    all_meta = []

    for i in range(0, len(pages_meta), batch_size):
        batch_meta = pages_meta[i:i + batch_size]
        images = []
        valid_meta = []

        for meta in batch_meta:
            img_path = Path(meta["img_path"])
            if not img_path.exists():
                print(f"  WARNING: Image not found: {img_path}")
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_meta.append(meta)
            except Exception as e:
                print(f"  WARNING: Could not load {img_path}: {e}")

        if not images:
            continue

        with torch.no_grad():
            batch_input = processor.process_images(images).to(model.device)
            embs = model(**batch_input)  # [batch, n_patches, hidden]
            # Move to CPU to save GPU memory
            for j, emb in enumerate(embs):
                all_embeddings.append(emb.cpu().float().numpy())
                all_meta.append(valid_meta[j])

        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(pages_meta))}/{len(pages_meta)} pages")

    return all_embeddings, all_meta


def embed_queries_colpali(model, processor, queries, batch_size=8):
    """Embed text queries using ColPali model."""
    all_query_embs = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        with torch.no_grad():
            batch_input = processor.process_queries(batch).to(model.device)
            embs = model(**batch_input)  # [batch, n_tokens, hidden]
            for emb in embs:
                all_query_embs.append(emb.cpu().float().numpy())

    return all_query_embs


def main():
    print("=" * 60)
    print(f"ColPali Page Embedding ({MODEL_NAME})")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: GPU required for ColPali embedding")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    # Load page manifest
    manifest_path = DATA / "page_images" / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: Run 01_render_pages.py first")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build pages metadata list
    pages_meta = []
    for doc_id, info in sorted(manifest.items()):
        img_dir = Path(info["image_dir"])
        for page_num in range(1, info["n_pages"] + 1):
            img_path = img_dir / f"page_{page_num:04d}.png"
            pages_meta.append({
                "doc_id": doc_id,
                "domain": info["domain"],
                "page_num": page_num,
                "img_path": str(img_path),
            })

    print(f"Total pages to embed: {len(pages_meta)}")

    # Load model
    print(f"\nLoading {MODEL_NAME} model...")
    start = time.time()
    model, processor = get_model_and_processor(MODEL_NAME)
    print(f"Model loaded in {time.time() - start:.1f}s")
    print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Output dir
    out_dir = EMB_DIR / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Embed pages
    print("\nEmbedding pages...")
    start = time.time()
    embeddings, valid_meta = embed_pages_colpali(model, processor, pages_meta, batch_size=4)
    elapsed = time.time() - start
    print(f"Embedded {len(embeddings)} pages in {elapsed:.1f}s "
          f"({len(embeddings)/elapsed:.1f} pages/sec)")

    # Save embeddings and metadata
    import numpy as np
    print("Saving embeddings...")
    for i, (emb, meta) in enumerate(zip(embeddings, valid_meta)):
        # Save each page embedding as a numpy array
        key = f"{meta['doc_id']}_p{meta['page_num']:04d}"
        np.save(str(out_dir / f"{key}.npy"), emb)

    # Save metadata
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(valid_meta, f, indent=1, ensure_ascii=False)

    print(f"Saved {len(embeddings)} page embeddings to {out_dir}")


if __name__ == "__main__":
    main()

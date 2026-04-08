#!/usr/bin/env python3
"""Phase 3.3: Resource profiling — time and VRAM at each pipeline stage.

Estimates P100 runtime by applying a ~0.6x slowdown factor vs A30.
"""

import json
import os
import time
from pathlib import Path

import torch

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
OUT = BASE / "outputs"
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

# Kaggle P100 slowdown factor relative to A30
# P100: 10.6 TFLOPS FP32 / A30: 10.3 TFLOPS FP32 ≈ similar for FP32
# But P100 has slower mem bandwidth: 732 vs 933 GB/s → ~0.78x
# Plus P100 no BF16 acceleration → ~0.6-0.7x overall for LLM inference
P100_FACTOR = 0.65


def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def get_total_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0


def profile_text_extraction():
    """Profile PyMuPDF text extraction."""
    import fitz
    pdf_dir = BASE / "data" / "raw_pdfs"
    pdfs = list(pdf_dir.rglob("*.pdf"))

    start = time.time()
    total_pages = 0
    for pdf_path in pdfs:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            page.get_text()
            total_pages += 1
        doc.close()
    elapsed = time.time() - start

    return {
        "stage": "text_extraction",
        "device": "cpu",
        "n_pdfs": len(pdfs),
        "n_pages": total_pages,
        "elapsed_s": elapsed,
        "pages_per_sec": total_pages / elapsed,
        "estimated_p100_s": elapsed,  # CPU-only, same on Kaggle
    }


def profile_page_rendering():
    """Profile PyMuPDF page rendering at 144 DPI."""
    import fitz
    pdf_dir = BASE / "data" / "raw_pdfs"
    pdfs = list(pdf_dir.rglob("*.pdf"))[:5]  # Sample 5 docs

    mat = fitz.Matrix(144/72, 144/72)
    start = time.time()
    total_pages = 0
    for pdf_path in pdfs:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            page.get_pixmap(matrix=mat)
            total_pages += 1
        doc.close()
    elapsed = time.time() - start

    # Extrapolate to all docs
    all_pdfs = list(pdf_dir.rglob("*.pdf"))
    scale = len(all_pdfs) / len(pdfs)

    return {
        "stage": "page_rendering",
        "device": "cpu",
        "n_pdfs_sampled": len(pdfs),
        "n_pages_sampled": total_pages,
        "elapsed_s": elapsed,
        "pages_per_sec": total_pages / elapsed,
        "estimated_full_s": elapsed * scale,
        "estimated_p100_s": elapsed * scale,  # CPU-only
    }


def profile_bm25():
    """Profile BM25 indexing and retrieval."""
    import csv
    import json
    from rank_bm25 import BM25Okapi

    # Load pages
    text_dir = BASE / "data" / "extracted_text"
    pages = []
    for jp in sorted(text_dir.glob("*.json")):
        if jp.name == "manifest.json":
            continue
        with open(jp) as f:
            doc = json.load(f)
        for p in doc["pages"]:
            pages.append(p["text"])

    with open(BASE / "data" / "dev_questions.csv") as f:
        questions = list(csv.DictReader(f))

    # Index
    t0 = time.time()
    tokenized = [t.lower().split() for t in pages]
    bm25 = BM25Okapi(tokenized)
    index_time = time.time() - t0

    # Query
    t0 = time.time()
    for q in questions:
        bm25.get_scores(q["Question"].lower().split())
    query_time = time.time() - t0

    return {
        "stage": "bm25",
        "device": "cpu",
        "n_pages": len(pages),
        "n_questions": len(questions),
        "index_time_s": index_time,
        "query_time_s": query_time,
        "total_s": index_time + query_time,
        "estimated_p100_s": index_time + query_time,  # CPU
    }


def profile_bge_m3():
    """Profile BGE-M3 embedding (GPU)."""
    if not torch.cuda.is_available():
        return {"stage": "bge_m3", "error": "No GPU"}

    import json
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    # Load small sample
    text_dir = BASE / "data" / "extracted_text"
    texts = []
    for jp in sorted(text_dir.glob("*.json"))[:3]:
        if jp.name == "manifest.json":
            continue
        with open(jp) as f:
            doc = json.load(f)
        for p in doc["pages"]:
            texts.append(p["text"][:512])

    torch.cuda.reset_peak_memory_stats()
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    vram_after_load = torch.cuda.memory_allocated() / 1024**3

    t0 = time.time()
    embs = model.encode(texts[:20], batch_size=8, max_length=512,
                        return_dense=True, return_sparse=False, return_colbert_vecs=False)
    sample_time = time.time() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Extrapolate to full set
    n_total = sum(1 for _ in (BASE / "data" / "extracted_text").glob("*.json"))
    # Rough: 1121 total pages
    n_pages_total = 1121
    scale = n_pages_total / 20
    estimated_full = sample_time * scale

    del model
    torch.cuda.empty_cache()

    return {
        "stage": "bge_m3",
        "device": "gpu",
        "n_sample": 20,
        "sample_time_s": sample_time,
        "pages_per_sec": 20 / sample_time,
        "vram_model_gb": vram_after_load,
        "vram_peak_gb": vram_peak,
        "estimated_full_s": estimated_full,
        "estimated_p100_s": estimated_full / P100_FACTOR,
    }


def profile_colsmol():
    """Profile ColSmol embedding (GPU)."""
    if not torch.cuda.is_available():
        return {"stage": "colsmol", "error": "No GPU"}

    try:
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    except ImportError:
        return {"stage": "colsmol", "error": "colpali-engine not installed"}

    img_dir = BASE / "data" / "page_images"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        return {"stage": "colsmol", "error": "No page images found"}

    from PIL import Image

    # Sample a few images
    sample_imgs = []
    for doc_dir in sorted(img_dir.iterdir())[:2]:
        if doc_dir.is_dir():
            for img_path in sorted(doc_dir.glob("*.png"))[:5]:
                sample_imgs.append(Image.open(img_path).convert("RGB"))

    if not sample_imgs:
        return {"stage": "colsmol", "error": "No images found"}

    torch.cuda.reset_peak_memory_stats()
    model = ColIdefics3.from_pretrained("vidore/colSmol-500M",
                                        dtype=torch.bfloat16, device_map="cuda").eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-500M")
    vram_load = torch.cuda.memory_allocated() / 1024**3

    t0 = time.time()
    with torch.no_grad():
        batch = processor.process_images(sample_imgs).to(model.device)
        _ = model(**batch)
    sample_time = time.time() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    del model
    torch.cuda.empty_cache()

    n_imgs = len(sample_imgs)
    scale = 1121 / n_imgs  # total pages
    return {
        "stage": "colsmol",
        "device": "gpu",
        "n_sample": n_imgs,
        "sample_time_s": sample_time,
        "pages_per_sec": n_imgs / sample_time,
        "vram_model_gb": vram_load,
        "vram_peak_gb": vram_peak,
        "estimated_full_s": sample_time * scale,
        "estimated_p100_s": sample_time * scale / P100_FACTOR,
    }


def profile_mamaylm(n_questions=20):
    """Profile MamayLM inference speed (GPU)."""
    candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    gguf_path = candidates[0] if candidates else BASE / "models" / "mamaylm" / "mamaylm.gguf"
    if not gguf_path.exists():
        return {"stage": "mamaylm", "error": f"GGUF not found at {gguf_path}"}
    try:
        from llama_cpp import Llama
    except ImportError:
        return {"stage": "mamaylm", "error": "llama-cpp-python not installed"}

    torch.cuda.reset_peak_memory_stats()
    llm = Llama(model_path=str(gguf_path), n_gpu_layers=-1, n_ctx=2048, verbose=False)
    vram_load = torch.cuda.memory_allocated() / 1024**3

    prompt = "Питання: Яка столиця України?\nА. Київ\nВ. Харків\nС. Одеса\nВідповідь (лише буква):"
    t0 = time.time()
    for _ in range(n_questions):
        llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
    elapsed = time.time() - t0

    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    del llm
    torch.cuda.empty_cache()

    # Total questions in test set (estimated)
    n_test = 461
    scale = n_test / n_questions
    return {
        "stage": "mamaylm",
        "device": "gpu",
        "n_sample": n_questions,
        "sample_time_s": elapsed,
        "questions_per_sec": n_questions / elapsed,
        "vram_model_gb": vram_load,
        "vram_peak_gb": vram_peak,
        "estimated_full_s": elapsed * scale,
        "estimated_p100_s": elapsed * scale / P100_FACTOR,
    }


def print_budget(profiles):
    print("\n" + "=" * 65)
    print(f"{'Stage':<20} {'Device':<6} {'Est A30 (s)':>12} {'Est P100 (s)':>13} {'VRAM (GB)':>10}")
    print("-" * 65)
    total_p100 = 0
    for p in profiles:
        if "error" in p:
            print(f"  {p['stage']:<18} {'N/A':<6} {'ERROR':>12}  {p['error']}")
            continue
        a30 = p.get("estimated_full_s", p.get("total_s", p.get("elapsed_s", 0)))
        p100 = p.get("estimated_p100_s", a30)
        vram = p.get("vram_peak_gb", 0)
        total_p100 += p100
        print(f"  {p['stage']:<18} {p['device']:<6} {a30:>12.0f}  {p100:>12.0f}  {vram:>9.1f}")
    print("-" * 65)
    print(f"  {'TOTAL':<18} {'':6} {'':>12}  {total_p100:>12.0f}")
    print(f"\n  P100 budget: {total_p100/3600:.2f}h / 9.00h limit")
    if total_p100 < 9 * 3600:
        print("  Status: FITS within 9-hour limit ✓")
    else:
        print(f"  Status: EXCEEDS by {(total_p100 - 9*3600)/3600:.2f}h — optimize needed!")


def main():
    print("=" * 65)
    print("UNLP 2026 Resource Profiler")
    print("=" * 65)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({get_total_vram_gb():.1f} GB)")
    print(f"P100 slowdown factor: {P100_FACTOR}x\n")

    profiles = []

    print("[1/5] Text extraction...")
    profiles.append(profile_text_extraction())
    print(f"  Done: {profiles[-1]['elapsed_s']:.1f}s")

    print("[2/5] Page rendering (sample)...")
    profiles.append(profile_page_rendering())
    print(f"  Done: {profiles[-1].get('elapsed_s', 0):.1f}s")

    print("[3/5] BM25 indexing + retrieval...")
    profiles.append(profile_bm25())
    print(f"  Done: {profiles[-1].get('total_s', 0):.1f}s")

    print("[4/5] BGE-M3 embedding...")
    profiles.append(profile_bge_m3())
    e = profiles[-1]
    print(f"  Done: {e.get('sample_time_s', 'N/A')}s (sample), "
          f"VRAM={e.get('vram_peak_gb', 0):.1f}GB")

    print("[5/5] ColSmol embedding (sample)...")
    profiles.append(profile_colsmol())
    e = profiles[-1]
    print(f"  Done: {e.get('sample_time_s', 'N/A')}s (sample), "
          f"VRAM={e.get('vram_peak_gb', 0):.1f}GB")

    # MamayLM only if available
    gguf_candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    gguf = gguf_candidates[0] if gguf_candidates else None
    if gguf and gguf.exists():
        print("[6/6] MamayLM inference...")
        profiles.append(profile_mamaylm())
        e = profiles[-1]
        print(f"  Done: {e.get('questions_per_sec', 'N/A')} q/s, "
              f"VRAM={e.get('vram_peak_gb', 0):.1f}GB")

    print_budget(profiles)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "resource_budget.json", "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nSaved to {OUT / 'resource_budget.json'}")


if __name__ == "__main__":
    main()

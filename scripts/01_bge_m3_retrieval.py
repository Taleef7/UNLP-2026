#!/usr/bin/env python3
"""Phase 1.4: BGE-M3 dense retrieval over extracted page texts.

Uses BAAI/bge-m3 for multilingual dense embeddings + FAISS similarity search.
"""

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
EMB_DIR = BASE / "embeddings" / "bge-m3"
OUT = BASE / "outputs"
MODEL_DIR = BASE / "models" / "bge-m3"

os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

MAX_LENGTH = 512  # BGE-M3 max is 8192, but 512 is enough for page chunks


def load_pages():
    """Load all extracted page texts."""
    text_dir = DATA / "extracted_text"
    pages = []
    for json_path in sorted(text_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc["doc_id"]
        domain = doc["domain"]
        for page in doc["pages"]:
            text = page["text"].strip()
            if len(text) < 10:
                text = f"[Empty page from {doc_id}]"
            pages.append({
                "doc_id": doc_id,
                "domain": domain,
                "page_num": page["page_num"],
                "text": text[:2000],  # Truncate very long pages for embedding
            })
    return pages


def embed_pages(model, pages, batch_size=32):
    """Embed page texts with BGE-M3."""
    texts = [p["text"] for p in pages]
    print(f"Embedding {len(texts)} pages (batch_size={batch_size})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    # embeddings is a dict with 'dense_vecs'
    dense = embeddings["dense_vecs"]
    elapsed = time.time() - start
    print(f"Embedded in {elapsed:.1f}s ({len(texts)/elapsed:.1f} pages/sec)")
    return np.array(dense, dtype=np.float32)


def embed_queries(model, queries, batch_size=32):
    """Embed queries with BGE-M3."""
    print(f"Embedding {len(queries)} queries...")
    start = time.time()
    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = embeddings["dense_vecs"]
    elapsed = time.time() - start
    print(f"Embedded in {elapsed:.1f}s")
    return np.array(dense, dtype=np.float32)


def evaluate_retrieval(questions, pages, index, page_embeddings, model, top_k=10):
    """Evaluate dense retrieval."""
    queries = [q["Question"] for q in questions]
    query_embs = embed_queries(model, queries)

    # Normalize for cosine similarity (FAISS IndexFlatIP)
    faiss.normalize_L2(query_embs)

    # Search
    scores, indices = index.search(query_embs, top_k)

    results = []
    doc_at_1 = 0
    doc_at_k = 0
    page_at_1 = 0
    page_at_k = 0

    for i, q in enumerate(questions):
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        top_pages = [pages[idx] for idx in indices[i]]
        top_scores = scores[i].tolist()

        if top_pages[0]["doc_id"] == true_doc:
            doc_at_1 += 1
        if any(p["doc_id"] == true_doc for p in top_pages):
            doc_at_k += 1
        if top_pages[0]["doc_id"] == true_doc and top_pages[0]["page_num"] == true_page:
            page_at_1 += 1
        if any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages):
            page_at_k += 1

        best_page_for_doc = None
        for p in top_pages:
            if p["doc_id"] == true_doc:
                best_page_for_doc = p["page_num"]
                break
        page_prox = max(0, 1 - abs(best_page_for_doc - true_page) / max(n_pages, 1)) if best_page_for_doc else 0.0

        results.append({
            "question_id": q["Question_ID"],
            "domain": q["Domain"],
            "doc_at_1": top_pages[0]["doc_id"] == true_doc,
            "page_at_1": top_pages[0]["doc_id"] == true_doc and top_pages[0]["page_num"] == true_page,
            "doc_at_k": any(p["doc_id"] == true_doc for p in top_pages),
            "page_at_k": any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages),
            "page_proximity": page_prox,
            "top_results": [
                {"doc_id": p["doc_id"], "page_num": p["page_num"], "score": s}
                for p, s in zip(top_pages, top_scores)
            ],
        })

    N = len(results)
    metrics = {
        "n_questions": N,
        "top_k": top_k,
        "doc_at_1": doc_at_1 / N,
        "doc_at_k": doc_at_k / N,
        "page_at_1": page_at_1 / N,
        "page_at_k": page_at_k / N,
        "avg_page_proximity": sum(r["page_proximity"] for r in results) / N,
    }

    # Per-domain
    domain_results = defaultdict(list)
    for r in results:
        domain_results[r["domain"]].append(r)
    per_domain = {}
    for domain, dr in sorted(domain_results.items()):
        n = len(dr)
        per_domain[domain] = {
            "n": n,
            "doc_at_1": sum(r["doc_at_1"] for r in dr) / n,
            "doc_at_k": sum(r["doc_at_k"] for r in dr) / n,
            "page_at_1": sum(r["page_at_1"] for r in dr) / n,
            "page_at_k": sum(r["page_at_k"] for r in dr) / n,
            "avg_page_proximity": sum(r["page_proximity"] for r in dr) / n,
        }
    metrics["per_domain"] = per_domain

    return metrics, results


def main():
    print("=" * 60)
    print("BGE-M3 Dense Retrieval")
    print("=" * 60)

    # Load pages
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")

    # Load model
    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # Check for cached embeddings
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMB_DIR / "page_embeddings.npy"
    meta_path = EMB_DIR / "page_metadata.json"

    if emb_path.exists():
        print("Loading cached embeddings...")
        page_embs = np.load(str(emb_path))
        assert page_embs.shape[0] == len(pages), "Embedding count mismatch, re-embedding..."
    else:
        page_embs = embed_pages(model, pages, batch_size=32)
        np.save(str(emb_path), page_embs)
        # Save metadata
        meta = [{"doc_id": p["doc_id"], "domain": p["domain"], "page_num": p["page_num"]}
                for p in pages]
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"Saved embeddings to {emb_path}")

    # Build FAISS index (cosine similarity via normalized IP)
    print("Building FAISS index...")
    dim = page_embs.shape[1]
    faiss.normalize_L2(page_embs)
    index = faiss.IndexFlatIP(dim)
    index.add(page_embs)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # Save FAISS index
    faiss_path = EMB_DIR / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))

    # Load questions
    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        all_questions = list(csv.DictReader(f))

    # Evaluate
    for top_k in [1, 5, 10]:
        metrics, results = evaluate_retrieval(all_questions, pages, index, page_embs, model, top_k=top_k)
        print(f"\n--- top_k={top_k} ---")
        print(f"  Doc@1={metrics['doc_at_1']:.4f}  Doc@{top_k}={metrics['doc_at_k']:.4f}")
        print(f"  Page@1={metrics['page_at_1']:.4f}  Page@{top_k}={metrics['page_at_k']:.4f}")
        print(f"  Avg page proximity: {metrics['avg_page_proximity']:.4f}")
        for domain, d in metrics["per_domain"].items():
            print(f"    {domain}: Doc@1={d['doc_at_1']:.4f} Doc@{top_k}={d['doc_at_k']:.4f} "
                  f"Page@1={d['page_at_1']:.4f} Page@{top_k}={d['page_at_k']:.4f}")

    # Save top-10 retrieval results
    _, results_10 = evaluate_retrieval(all_questions, pages, index, page_embs, model, top_k=10)
    retrieval_output = {}
    for r in results_10:
        retrieval_output[r["question_id"]] = r["top_results"]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bge_m3_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_output, f, indent=1, ensure_ascii=False)

    with open(OUT / "bge_m3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()

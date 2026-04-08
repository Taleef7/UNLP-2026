#!/usr/bin/env python3
"""Phase 1.6 (part 2): ColPali retrieval using MaxSim scoring.

Loads saved page embeddings and evaluates retrieval quality.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
EMB_DIR = BASE / "embeddings"
OUT = BASE / "outputs"

os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

MODEL_NAME = os.environ.get("MODEL", "colsmol")


def maxsim_score(query_emb, page_emb):
    """MaxSim: for each query token, max cosine similarity over page patches."""
    # query_emb: [n_tokens, hidden]
    # page_emb:  [n_patches, hidden]
    q_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    p_norm = page_emb / (np.linalg.norm(page_emb, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ p_norm.T  # [n_tokens, n_patches]
    return float(sim.max(axis=1).sum())


def maxsim_batch(query_emb, page_embs_stacked):
    """Vectorized MaxSim: score one query against all pages at once.

    query_emb: [n_q_tokens, hidden]
    page_embs_stacked: [n_pages, n_patches, hidden]
    Returns: [n_pages] scores
    """
    # Normalize query: [n_q, h]
    q_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
    # Normalize pages: [n_pages, n_patches, h]
    page_norms = np.linalg.norm(page_embs_stacked, axis=2, keepdims=True) + 1e-8
    p_norm = page_embs_stacked / page_norms
    # Batched matmul: [n_pages, n_q, n_patches]
    # q_norm[1, n_q, h] × p_norm[n_pages, h, n_patches] → [n_pages, n_q, n_patches]
    sim = np.einsum('qh,pth->pqt', q_norm, p_norm)  # [n_pages, n_q, n_patches]
    # MaxSim: max over patches for each q_token, sum over q_tokens
    return sim.max(axis=2).sum(axis=1)  # [n_pages]


def embed_queries(model, processor, queries, batch_size=8):
    """Embed queries using ColPali model."""
    import torch
    all_query_embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        with torch.no_grad():
            batch_input = processor.process_queries(batch).to(model.device)
            embs = model(**batch_input)
            for emb in embs:
                all_query_embs.append(emb.cpu().float().numpy())
    return all_query_embs


def evaluate_colpali_retrieval(questions, page_embeddings, page_meta, model, processor, top_k=10):
    """Evaluate ColPali retrieval."""
    import time
    queries = [q["Question"] for q in questions]
    print(f"Embedding {len(queries)} queries...")
    query_embs = embed_queries(model, processor, queries, batch_size=8)

    # Stack and GPU-normalize page embeddings for fast MaxSim
    import torch
    print("Moving page embeddings to GPU...")
    t0 = time.time()
    max_patches = max(e.shape[0] for e in page_embeddings)
    hidden = page_embeddings[0].shape[1]
    pages_arr = np.zeros((len(page_embeddings), max_patches, hidden), dtype=np.float32)
    for i, e in enumerate(page_embeddings):
        pages_arr[i, :e.shape[0], :] = e

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = "cuda"
        pages_gpu = torch.from_numpy(pages_arr).to(device).half()
        pages_gpu = torch.nn.functional.normalize(pages_gpu, dim=2)
        print(f"GPU pages: {tuple(pages_gpu.shape)} | "
              f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB | "
              f"Time: {time.time()-t0:.1f}s")
    else:
        # CPU fallback (slow)
        pages_arr = pages_arr / (np.linalg.norm(pages_arr, axis=2, keepdims=True) + 1e-8)
        print(f"CPU fallback: {pages_arr.shape}, {time.time()-t0:.1f}s")

    results = []
    t_score = time.time()
    for qi, q in enumerate(questions):
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        q_emb = query_embs[qi]

        # GPU-accelerated vectorized MaxSim
        if use_gpu:
            q_t = torch.from_numpy(q_emb).to(device).half()
            q_t = torch.nn.functional.normalize(q_t, dim=1)  # [n_q, hidden]
            with torch.no_grad():
                sim = torch.einsum('qh,pth->pqt', q_t, pages_gpu)  # [n_pages, n_q, n_patches]
                scores_t = sim.max(dim=2).values.sum(dim=1)  # [n_pages]
            scores = scores_t.cpu().numpy().astype(np.float32)
        else:
            q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
            sim = np.einsum('qh,pth->pqt', q_norm, pages_arr)
            scores = sim.max(axis=2).sum(axis=1)

        top_indices = np.argsort(scores)[::-1][:top_k].tolist()

        top_pages = [page_meta[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        doc_at_1 = top_pages[0]["doc_id"] == true_doc
        page_at_1 = doc_at_1 and top_pages[0]["page_num"] == true_page
        doc_at_k = any(p["doc_id"] == true_doc for p in top_pages)
        page_at_k = any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages)

        best_page = next((p["page_num"] for p in top_pages if p["doc_id"] == true_doc), None)
        page_prox = max(0, 1 - abs(best_page - true_page) / max(n_pages, 1)) if best_page else 0.0

        results.append({
            "question_id": q["Question_ID"],
            "domain": q["Domain"],
            "doc_at_1": doc_at_1,
            "page_at_1": page_at_1,
            "doc_at_k": doc_at_k,
            "page_at_k": page_at_k,
            "page_proximity": page_prox,
            "top_results": [{"doc_id": p["doc_id"], "page_num": p["page_num"], "score": float(s)}
                            for p, s in zip(top_pages, top_scores)],
        })

        if qi % 50 == 0:
            print(f"  Processed {qi}/{len(questions)} questions")

    N = len(results)
    metrics = {
        "model": MODEL_NAME, "n": N, "top_k": top_k,
        "doc_at_1": sum(r["doc_at_1"] for r in results) / N,
        "doc_at_k": sum(r["doc_at_k"] for r in results) / N,
        "page_at_1": sum(r["page_at_1"] for r in results) / N,
        "page_at_k": sum(r["page_at_k"] for r in results) / N,
        "avg_page_proximity": sum(r["page_proximity"] for r in results) / N,
    }
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
    print(f"ColPali Retrieval ({MODEL_NAME})")
    print("=" * 60)

    emb_dir = EMB_DIR / MODEL_NAME
    meta_path = emb_dir / "metadata.json"

    if not meta_path.exists():
        print(f"ERROR: Run 01_colpali_embed.py first (MODEL={MODEL_NAME})")
        return

    # Load metadata
    with open(meta_path) as f:
        page_meta = json.load(f)
    print(f"Loading {len(page_meta)} page embeddings...")

    # Load all embeddings
    page_embeddings = []
    for meta in page_meta:
        key = f"{meta['doc_id']}_p{meta['page_num']:04d}"
        emb_path = emb_dir / f"{key}.npy"
        if emb_path.exists():
            page_embeddings.append(np.load(str(emb_path)))
        else:
            print(f"WARNING: Missing embedding for {key}")
            page_embeddings.append(None)

    # Filter out None
    valid = [(e, m) for e, m in zip(page_embeddings, page_meta) if e is not None]
    page_embeddings, page_meta = zip(*valid) if valid else ([], [])
    page_embeddings = list(page_embeddings)
    page_meta = list(page_meta)
    print(f"Loaded {len(page_embeddings)} valid page embeddings")

    # Load model for query embedding
    import torch

    print(f"\nLoading {MODEL_NAME} for query embedding...")
    if MODEL_NAME == "colsmol":
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        local_path = BASE / "models" / "colsmol"
        load_path = str(local_path) if (local_path.exists() and any(local_path.iterdir())) else "vidore/colSmol-500M"
        model = ColIdefics3.from_pretrained(load_path, dtype=torch.bfloat16, device_map="cuda").eval()
        processor = ColIdefics3Processor.from_pretrained(load_path)
    else:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        local_path = BASE / "models" / "colqwen2"
        load_path = str(local_path) if (local_path.exists() and any(local_path.iterdir())) else "vidore/colqwen2-v1.0"
        model = ColQwen2.from_pretrained(load_path, dtype=torch.bfloat16, device_map="cuda").eval()
        processor = ColQwen2Processor.from_pretrained(load_path)

    # Load questions
    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        all_questions = list(csv.DictReader(f))

    # Evaluate
    metrics, results = evaluate_colpali_retrieval(all_questions, page_embeddings, page_meta,
                                                   model, processor, top_k=10)

    print(f"\n--- ColPali ({MODEL_NAME}) top_k=10 ---")
    print(f"  Doc@1={metrics['doc_at_1']:.4f}  Doc@10={metrics['doc_at_k']:.4f}")
    print(f"  Page@1={metrics['page_at_1']:.4f}  Page@10={metrics['page_at_k']:.4f}")
    print(f"  Avg page proximity: {metrics['avg_page_proximity']:.4f}")
    for domain, d in metrics["per_domain"].items():
        print(f"    {domain}: Doc@1={d['doc_at_1']:.4f} Doc@10={d['doc_at_k']:.4f} "
              f"Page@1={d['page_at_1']:.4f} Page@10={d['page_at_k']:.4f}")

    # Save
    retrieval_output = {r["question_id"]: r["top_results"] for r in results}
    suffix = f"_{MODEL_NAME}"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"colpali{suffix}_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_output, f, indent=1, ensure_ascii=False)
    with open(OUT / f"colpali{suffix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

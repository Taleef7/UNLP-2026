#!/usr/bin/env python3
"""Phase 1.5: Hybrid text retrieval — BM25 + BGE-M3 via Reciprocal Rank Fusion (RRF).

RRF score: sum(1 / (k + rank_i)) across retrieval systems.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Combine multiple ranked lists via RRF.

    Args:
        ranked_lists: list of lists of (doc_id, page_num) in rank order
        k: RRF constant (default 60)

    Returns:
        list of (doc_id, page_num, rrf_score) sorted by score descending
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def evaluate_fusion(questions, fused_results, gt_by_id, top_k=10):
    """Evaluate fused retrieval results."""
    results = []
    for q in questions:
        qid = q["Question_ID"]
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        top = fused_results.get(qid, [])[:top_k]
        top_pages = [{"doc_id": d, "page_num": p} for (d, p), s in top]

        if not top_pages:
            results.append({
                "question_id": qid, "domain": q["Domain"],
                "doc_at_1": False, "page_at_1": False,
                "doc_at_k": False, "page_at_k": False,
                "page_proximity": 0.0,
            })
            continue

        doc_at_1 = top_pages[0]["doc_id"] == true_doc
        page_at_1 = doc_at_1 and top_pages[0]["page_num"] == true_page
        doc_at_k = any(p["doc_id"] == true_doc for p in top_pages)
        page_at_k = any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages)

        best_page = next((p["page_num"] for p in top_pages if p["doc_id"] == true_doc), None)
        page_prox = max(0, 1 - abs(best_page - true_page) / max(n_pages, 1)) if best_page else 0.0

        results.append({
            "question_id": qid, "domain": q["Domain"],
            "doc_at_1": doc_at_1, "page_at_1": page_at_1,
            "doc_at_k": doc_at_k, "page_at_k": page_at_k,
            "page_proximity": page_prox,
        })

    N = len(results)
    metrics = {
        "n": N, "top_k": top_k,
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
    print("Hybrid Text Retrieval (BM25 + BGE-M3 RRF)")
    print("=" * 60)

    # Load retrieval outputs
    bm25_path = OUT / "bm25_retrieval.json"
    bge_path = OUT / "bge_m3_retrieval.json"

    if not bm25_path.exists():
        print("ERROR: Run 01_bm25_retrieval.py first")
        return
    if not bge_path.exists():
        print("WARNING: BGE-M3 results not found, using BM25 only")
        use_bge = False
    else:
        use_bge = True

    with open(bm25_path) as f:
        bm25_results = json.load(f)
    if use_bge:
        with open(bge_path) as f:
            bge_results = json.load(f)

    # Load questions
    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        all_questions = list(csv.DictReader(f))
    gt_by_id = {q["Question_ID"]: q for q in all_questions}

    # Fuse for each question
    fused = {}
    for q in all_questions:
        qid = q["Question_ID"]
        bm25_ranked = [(r["doc_id"], r["page_num"]) for r in bm25_results.get(qid, [])]
        ranked_lists = [bm25_ranked]

        if use_bge:
            bge_ranked = [(r["doc_id"], r["page_num"]) for r in bge_results.get(qid, [])]
            ranked_lists.append(bge_ranked)

        fused[qid] = reciprocal_rank_fusion(ranked_lists, k=60)

    # Evaluate different k values
    for rrf_k in [10, 30, 60]:
        # Recompute with different RRF k
        fused_k = {}
        for q in all_questions:
            qid = q["Question_ID"]
            bm25_ranked = [(r["doc_id"], r["page_num"]) for r in bm25_results.get(qid, [])]
            ranked_lists = [bm25_ranked]
            if use_bge:
                bge_ranked = [(r["doc_id"], r["page_num"]) for r in bge_results.get(qid, [])]
                ranked_lists.append(bge_ranked)
            fused_k[qid] = reciprocal_rank_fusion(ranked_lists, k=rrf_k)

        metrics, _ = evaluate_fusion(all_questions, fused_k, gt_by_id, top_k=10)
        print(f"\n--- RRF k={rrf_k} ---")
        print(f"  Doc@1={metrics['doc_at_1']:.4f}  Doc@10={metrics['doc_at_k']:.4f}")
        print(f"  Page@1={metrics['page_at_1']:.4f}  Page@10={metrics['page_at_k']:.4f}")
        print(f"  Avg page proximity: {metrics['avg_page_proximity']:.4f}")

    # Save best (k=60) fusion results
    metrics_best, results_best = evaluate_fusion(all_questions, fused, gt_by_id, top_k=10)
    output = {qid: [{"doc_id": d, "page_num": p, "score": s}
                     for (d, p), s in fused[qid][:10]]
              for qid in fused}

    with open(OUT / "hybrid_text_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)
    with open(OUT / "hybrid_text_metrics.json", "w") as f:
        json.dump(metrics_best, f, indent=2)

    print(f"\nSaved to {OUT / 'hybrid_text_retrieval.json'}")


if __name__ == "__main__":
    main()

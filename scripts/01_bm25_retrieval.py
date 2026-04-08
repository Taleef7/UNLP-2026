#!/usr/bin/env python3
"""Phase 1.3: BM25 baseline retrieval over extracted page texts."""

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"


def load_pages():
    """Load all extracted page texts into a flat list."""
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
            pages.append({
                "doc_id": doc_id,
                "domain": domain,
                "page_num": page["page_num"],
                "text": page["text"],
            })
    return pages


def tokenize(text):
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def evaluate_retrieval(questions, pages, bm25, page_index, top_k=10, split_name="all"):
    """Evaluate BM25 retrieval on questions."""
    results = []
    doc_at_1 = 0
    doc_at_k = 0
    page_at_1 = 0
    page_at_k = 0

    for q in questions:
        query = q["Question"]
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        # BM25 retrieval
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_pages = [pages[i] for i in top_indices]

        # Doc@1
        if top_pages[0]["doc_id"] == true_doc:
            doc_at_1 += 1
        # Doc@K
        if any(p["doc_id"] == true_doc for p in top_pages):
            doc_at_k += 1
        # Page@1
        if top_pages[0]["doc_id"] == true_doc and top_pages[0]["page_num"] == true_page:
            page_at_1 += 1
        # Page@K
        if any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages):
            page_at_k += 1

        # Page proximity for best doc match
        best_page_for_doc = None
        for p in top_pages:
            if p["doc_id"] == true_doc:
                best_page_for_doc = p["page_num"]
                break

        if best_page_for_doc is not None:
            page_prox = max(0, 1 - abs(best_page_for_doc - true_page) / max(n_pages, 1))
        else:
            page_prox = 0.0

        results.append({
            "question_id": q["Question_ID"],
            "domain": q["Domain"],
            "doc_at_1": top_pages[0]["doc_id"] == true_doc,
            "page_at_1": top_pages[0]["doc_id"] == true_doc and top_pages[0]["page_num"] == true_page,
            "doc_at_k": any(p["doc_id"] == true_doc for p in top_pages),
            "page_at_k": any(p["doc_id"] == true_doc and p["page_num"] == true_page for p in top_pages),
            "page_proximity": page_prox,
            "top_doc": top_pages[0]["doc_id"],
            "top_page": top_pages[0]["page_num"],
        })

    N = len(results)
    metrics = {
        "split": split_name,
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
    print("BM25 Retrieval")
    print("=" * 60)

    # Load pages
    pages = load_pages()
    print(f"Loaded {len(pages)} pages from {len(set(p['doc_id'] for p in pages))} documents")

    # Build BM25 index
    print("Building BM25 index...")
    start = time.time()
    tokenized_pages = [tokenize(p["text"]) for p in pages]
    bm25 = BM25Okapi(tokenized_pages)
    print(f"BM25 index built in {time.time() - start:.1f}s")

    # Build page index for fast lookup
    page_index = {}
    for i, p in enumerate(pages):
        page_index[(p["doc_id"], p["page_num"])] = i

    # Load questions
    val_path = DATA / "splits" / "val.csv"
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_questions = list(csv.DictReader(f))
        print(f"Val questions: {len(val_questions)}")
    else:
        val_questions = None

    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        all_questions = list(csv.DictReader(f))

    # Evaluate
    for top_k in [1, 5, 10]:
        metrics, results = evaluate_retrieval(all_questions, pages, bm25, page_index,
                                              top_k=top_k, split_name="dev_all")
        print(f"\n--- top_k={top_k} (all dev) ---")
        print(f"  Doc@1={metrics['doc_at_1']:.4f}  Doc@{top_k}={metrics['doc_at_k']:.4f}")
        print(f"  Page@1={metrics['page_at_1']:.4f}  Page@{top_k}={metrics['page_at_k']:.4f}")
        print(f"  Avg page proximity: {metrics['avg_page_proximity']:.4f}")
        for domain, d in metrics["per_domain"].items():
            print(f"    {domain}: Doc@1={d['doc_at_1']:.4f} Doc@{top_k}={d['doc_at_k']:.4f} "
                  f"Page@1={d['page_at_1']:.4f} Page@{top_k}={d['page_at_k']:.4f}")

    # Save top-10 results for pipeline use
    metrics_10, results_10 = evaluate_retrieval(all_questions, pages, bm25, page_index,
                                                 top_k=10, split_name="dev_all")

    # Save retrieval predictions (top-10 per question for downstream use)
    print("\nSaving BM25 retrieval results...")
    retrieval_output = {}
    for q in all_questions:
        tokenized_query = tokenize(q["Question"])
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        retrieval_output[q["Question_ID"]] = [
            {
                "doc_id": pages[i]["doc_id"],
                "page_num": pages[i]["page_num"],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bm25_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_output, f, indent=1, ensure_ascii=False)

    with open(OUT / "bm25_metrics.json", "w") as f:
        json.dump(metrics_10, f, indent=2)

    print(f"Saved to {OUT / 'bm25_retrieval.json'}")
    print(f"Saved to {OUT / 'bm25_metrics.json'}")


if __name__ == "__main__":
    main()

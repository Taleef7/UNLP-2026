#!/usr/bin/env python3
"""Phase 1.7: Full hybrid fusion — 3-way RRF (BM25 + BGE-M3 + ColPali).

This is the paper's main novelty: dual-pathway (text + vision) retrieval fusion.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"

MODEL_NAME = "colsmol"  # or colqwen2


def reciprocal_rank_fusion(ranked_lists, k=60):
    """RRF over multiple ranked lists of (doc_id, page_num) tuples."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def evaluate_fusion(questions, fused_results, top_k=10):
    """Evaluate fused retrieval results."""
    results = []
    for q in questions:
        qid = q["Question_ID"]
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        top = [(d_p, s) for d_p, s in fused_results.get(qid, [])[:top_k]]
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


def load_retrieval(path):
    """Load retrieval JSON as {qid: [(doc_id, page_num), ...]} ranked list."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {qid: [(r["doc_id"], r["page_num"]) for r in results]
            for qid, results in data.items()}


def main():
    print("=" * 60)
    print("Full Hybrid Fusion (BM25 + BGE-M3 + ColPali)")
    print("=" * 60)

    # Load available retrieval systems
    systems = {}
    system_files = {
        "bm25": OUT / "bm25_retrieval.json",
        "bge_m3": OUT / "bge_m3_retrieval.json",
        f"colpali_{MODEL_NAME}": OUT / f"colpali_{MODEL_NAME}_retrieval.json",
    }

    for name, path in system_files.items():
        ranked = load_retrieval(path)
        if ranked:
            systems[name] = ranked
            print(f"Loaded: {name} ({len(ranked)} questions)")
        else:
            print(f"Missing: {name} (skipping)")

    if not systems:
        print("ERROR: No retrieval results found. Run retrieval scripts first.")
        sys.exit(1)

    # Load questions
    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        all_questions = list(csv.DictReader(f))

    # Evaluate all fusion combinations
    system_names = list(systems.keys())
    print(f"\nAvailable systems: {system_names}")

    all_metrics = {}

    # 1. Individual systems
    for name in system_names:
        fused = {q["Question_ID"]: [((d, p), 1.0) for d, p in systems[name].get(q["Question_ID"], [])]
                 for q in all_questions}
        # Convert to expected format
        fused2 = {}
        for q in all_questions:
            qid = q["Question_ID"]
            ranked = systems[name].get(qid, [])
            fused2[qid] = [((d, p), 1.0 / (i + 1)) for i, (d, p) in enumerate(ranked)]
        metrics, _ = evaluate_fusion(all_questions, fused2)
        all_metrics[f"single_{name}"] = metrics
        print(f"\n  {name}: Doc@1={metrics['doc_at_1']:.4f}  Page@1={metrics['page_at_1']:.4f}  "
              f"PageProx={metrics['avg_page_proximity']:.4f}")

    # 2. All pairwise combinations
    for i, n1 in enumerate(system_names):
        for n2 in system_names[i + 1:]:
            for rrf_k in [60]:
                fused = {}
                for q in all_questions:
                    qid = q["Question_ID"]
                    ranked_lists = [systems[n1].get(qid, []), systems[n2].get(qid, [])]
                    fused[qid] = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
                metrics, _ = evaluate_fusion(all_questions, fused)
                label = f"rrf_{n1}+{n2}_k{rrf_k}"
                all_metrics[label] = metrics
                print(f"\n  {n1}+{n2} (k={rrf_k}): Doc@1={metrics['doc_at_1']:.4f}  "
                      f"Page@1={metrics['page_at_1']:.4f}  PageProx={metrics['avg_page_proximity']:.4f}")

    # 3. 3-way fusion (if all systems available)
    if len(system_names) >= 3:
        for rrf_k in [10, 30, 60, 100]:
            fused = {}
            for q in all_questions:
                qid = q["Question_ID"]
                ranked_lists = [systems[n].get(qid, []) for n in system_names]
                fused[qid] = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
            metrics, _ = evaluate_fusion(all_questions, fused)
            label = f"rrf_3way_k{rrf_k}"
            all_metrics[label] = metrics
            print(f"\n  3-way RRF (k={rrf_k}): Doc@1={metrics['doc_at_1']:.4f}  "
                  f"Page@1={metrics['page_at_1']:.4f}  PageProx={metrics['avg_page_proximity']:.4f}")

    # Find best configuration
    best_label = max(all_metrics, key=lambda k: all_metrics[k]["avg_page_proximity"])
    best_metrics = all_metrics[best_label]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_label}")
    print(f"  Doc@1={best_metrics['doc_at_1']:.4f}  Doc@10={best_metrics['doc_at_k']:.4f}")
    print(f"  Page@1={best_metrics['page_at_1']:.4f}  Page@10={best_metrics['page_at_k']:.4f}")
    print(f"  Avg page proximity: {best_metrics['avg_page_proximity']:.4f}")

    # Save best fusion results
    # Determine which systems to use for best config
    if "3way" in best_label:
        ranked_lists_fn = lambda qid: [systems[n].get(qid, []) for n in system_names]
        k = int(best_label.split("_k")[-1])
    elif "+" in best_label:
        parts = best_label.replace("rrf_", "").split("+")
        n1, n2_k = parts[0], parts[1]
        n2 = n2_k.split("_k")[0]
        k = int(n2_k.split("_k")[1])
        ranked_lists_fn = lambda qid: [systems[n1].get(qid, []), systems[n2].get(qid, [])]
    else:
        n = best_label.replace("single_", "")
        ranked_lists_fn = lambda qid: [systems[n].get(qid, [])]
        k = 60

    best_fused = {}
    for q in all_questions:
        qid = q["Question_ID"]
        best_fused[qid] = reciprocal_rank_fusion(ranked_lists_fn(qid), k=k)

    output = {qid: [{"doc_id": d, "page_num": p, "score": float(s)}
                     for (d, p), s in results[:10]]
              for qid, results in best_fused.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "hybrid_fusion_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)
    with open(OUT / "hybrid_fusion_metrics.json", "w") as f:
        json.dump({"all": all_metrics, "best": {"label": best_label, "metrics": best_metrics}},
                  f, indent=2)
    print(f"\nSaved to {OUT}/hybrid_fusion_retrieval.json")


if __name__ == "__main__":
    main()

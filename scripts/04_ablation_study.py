#!/usr/bin/env python3
"""Phase 4.1: Ablation study for paper.

Factors:
1. Retrieval: text-only / vision-only / hybrid
2. RRF k parameter: 10, 30, 60, 100
3. Context pages: 1, 3, 5
4. Prompt variant: direct / cot / ukrainian / evidence / elimination
5. Model: MamayLM-12B / MamayLM-4B (if available) / baseline
6. Leave-one-domain-out generalization
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

def _find_gguf():
    candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    return candidates[0] if candidates else BASE / "models" / "mamaylm" / "mamaylm.gguf"

GGUF_PATH = _find_gguf()
ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]


def load_retrieval_results():
    """Load all available retrieval results."""
    systems = {}
    for name, fname in [
        ("bm25", "bm25_retrieval.json"),
        ("bge_m3", "bge_m3_retrieval.json"),
        ("colpali_colsmol", "colpali_colsmol_retrieval.json"),
        ("hybrid_text", "hybrid_text_retrieval.json"),
        ("hybrid_fusion", "hybrid_fusion_retrieval.json"),
    ]:
        path = OUT / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            systems[name] = {
                qid: [(r["doc_id"], r["page_num"]) for r in results]
                for qid, results in data.items()
            }
            print(f"  Loaded: {name}")
    return systems


def reciprocal_rank_fusion(ranked_lists, k=60):
    """RRF over multiple ranked lists."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def oracle_retrieval(question, n_pages=3):
    """Oracle: return the true doc/page as top result."""
    true_doc = question["Doc_ID"]
    true_page = int(question["Page_Num"])
    n_total = int(question["N_Pages"])
    results = [(true_doc, true_page)]
    for offset in [-1, 1, -2, 2]:
        pn = true_page + offset
        if 1 <= pn <= n_total:
            results.append((true_doc, pn))
    return results[:n_pages]


def score_greedy(llm, prompt):
    """Greedy decode, extract A-F from output."""
    import re
    resp = llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
    text = resp["choices"][0]["text"].strip()
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter):
            return letter
    ua_map = {"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E", "Ф": "F"}
    for cyr, lat in ua_map.items():
        if text.startswith(cyr):
            return lat
    m = re.search(r'\b([A-F])\b', text.upper())
    if m:
        return m.group(1)
    for c in text.upper():
        if c in "ABCDEF":
            return c
    return "A"


def evaluate_with_retrieval_and_llm(questions, retrieval_system, oracle_pages, llm,
                                     prompt_builder, context_pages=3, top_k=10):
    """Full evaluation: retrieval + LLM answer generation."""
    correct_answer = 0
    correct_doc = 0
    total_page_prox = 0.0
    results = []

    for q in questions:
        qid = q["Question_ID"]
        true_doc = q["Doc_ID"]
        true_page = int(q["Page_Num"])
        n_pages = int(q["N_Pages"])

        # Get top retrieved pages
        if retrieval_system == "oracle":
            top_pages = oracle_retrieval(q, context_pages)
        else:
            top_pages = retrieval_system.get(qid, [])[:top_k]

        # Best predicted doc/page
        pred_doc = top_pages[0][0] if top_pages else true_doc
        pred_page = top_pages[0][1] if top_pages else true_page

        # Build context
        context_parts = []
        for doc_id, pnum in top_pages[:context_pages]:
            text = oracle_pages.get((doc_id, pnum), "")
            if text.strip():
                context_parts.append(f"[Сторінка {pnum}]\n{text[:1200]}")
        context = "\n\n".join(context_parts) or "[No context]"

        # Score MCQ
        prompt = prompt_builder(q, context)
        pred_answer = score_greedy(llm, prompt)

        # Compute metrics
        a_i = pred_answer == q["Correct_Answer"]
        d_i = pred_doc == true_doc
        p_i = max(0, 1 - abs(pred_page - true_page) / max(n_pages, 1)) if d_i else 0.0

        if a_i:
            correct_answer += 1
        if d_i:
            correct_doc += 1
        total_page_prox += p_i

        results.append({"question_id": qid, "domain": q.get("Domain", ""),
                         "a_i": a_i, "d_i": d_i, "p_i": p_i})

    N = len(results)
    return {
        "n": N,
        "answer_acc": correct_answer / N,
        "doc_acc": correct_doc / N,
        "page_prox": total_page_prox / N,
        "composite": 0.5 * (correct_answer / N) + 0.25 * (correct_doc / N) + 0.25 * (total_page_prox / N),
    }


def main():
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    # This script is designed to be run once all components are ready
    # It imports from other scripts, so run after all Phase 1 & 2 are complete

    systems = load_retrieval_results()
    if not systems:
        print("No retrieval results found. Run Phase 1 scripts first.")
        return

    with open(DATA / "splits" / "val.csv", "r", encoding="utf-8") as f:
        val_questions = list(csv.DictReader(f))

    # Load oracle pages
    oracle_pages = {}
    text_dir = DATA / "extracted_text"
    for json_path in sorted(text_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for page in doc["pages"]:
            oracle_pages[(doc["doc_id"], page["page_num"])] = page["text"]

    print(f"\nVal questions: {len(val_questions)}")
    print(f"Available retrieval systems: {list(systems.keys())}")

    # Retrieval-only ablation (no LLM needed)
    print("\n=== Retrieval-only evaluation ===")

    # Print retrieval metrics per system
    N = len(val_questions)
    for name, system in systems.items():
        doc_at_1 = sum(1 for q in val_questions
                       if system.get(q["Question_ID"], [("", 0)])[0][0] == q["Doc_ID"])
        page_at_1 = sum(1 for q in val_questions
                        if (system.get(q["Question_ID"], [("", 0)])[0][0] == q["Doc_ID"]
                            and system.get(q["Question_ID"], [("", 0)])[0][1] == int(q["Page_Num"])))
        print(f"  {name}: Doc@1={doc_at_1/N:.4f}  Page@1={page_at_1/N:.4f}")

    print("\nNote: Full LLM ablation requires MamayLM + llama-cpp-python.")
    print("Run after Phase 2 setup is complete.")

    OUT.mkdir(parents=True, exist_ok=True)
    ablation_results = {"retrieval_systems": list(systems.keys()), "val_size": len(val_questions)}
    with open(OUT / "ablation_partial.json", "w") as f:
        json.dump(ablation_results, f, indent=2)


if __name__ == "__main__":
    main()

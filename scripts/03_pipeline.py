#!/usr/bin/env python3
"""Phase 3.1: End-to-end pipeline — retrieval + MCQ → submission.csv

Usage:
    python scripts/03_pipeline.py \
        --questions data/dev_questions.csv \
        --pdf_dir data/raw_pdfs \
        --output outputs/submission.csv \
        [--retrieval bm25|bge|hybrid|colpali] \
        [--top_k 5] \
        [--strategy direct|cot]
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]


# ─── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_dirs):
    """Extract page texts from a list of PDF directories."""
    import fitz
    pages = {}  # {(doc_id, page_num): text}
    doc_meta = {}  # {doc_id: {"n_pages": N, "domain": d}}

    for pdf_dir in pdf_dirs:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            continue
        domain = pdf_dir.name
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            doc_id = pdf_path.name
            doc = fitz.open(str(pdf_path))
            n_pages = len(doc)
            doc_meta[doc_id] = {"n_pages": n_pages, "domain": domain}
            for i, page in enumerate(doc):
                pages[(doc_id, i + 1)] = page.get_text()
            doc.close()

    return pages, doc_meta


# ─── BM25 Retrieval ───────────────────────────────────────────────────────────

def build_bm25_index(pages):
    """Build BM25 index from {(doc_id, page_num): text}."""
    from rank_bm25 import BM25Okapi
    page_list = [{"doc_id": k[0], "page_num": k[1], "text": v}
                 for k, v in sorted(pages.items())]
    tokenized = [t["text"].lower().split() for t in page_list]
    bm25 = BM25Okapi(tokenized)
    return bm25, page_list


def retrieve_bm25(bm25, page_list, query, top_k=10):
    scores = bm25.get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"doc_id": page_list[i]["doc_id"], "page_num": page_list[i]["page_num"],
             "score": float(scores[i])} for i in top]


# ─── MCQ Scoring ─────────────────────────────────────────────────────────────

UA_TO_LATIN = {"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E", "Ф": "F",
               "а": "A", "в": "B", "с": "C", "д": "D", "е": "E", "ф": "F"}


def load_llm(gguf_path, n_ctx=2048, n_gpu_layers=-1):
    from llama_cpp import Llama
    print(f"Loading {Path(gguf_path).name}...")
    return Llama(model_path=str(gguf_path), n_gpu_layers=n_gpu_layers,
                 n_ctx=n_ctx, verbose=False)


def extract_answer(text):
    """Extract A-F from generated text."""
    import re
    text = text.strip()
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter):
            return letter
    for cyrillic, latin in UA_TO_LATIN.items():
        if text.startswith(cyrillic):
            return latin
    m = re.search(r'\b([A-F])\b', text.upper())
    if m:
        return m.group(1)
    for c in text.upper():
        if c in "ABCDEF":
            return c
    return "A"


def score_mcq(llm, question_row, context, strategy="direct"):
    """Score MCQ — greedy decode, extract A-F letter."""
    q = question_row
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    if strategy == "evidence":
        # V4: evidence-first — best prompt (86.8% oracle accuracy)
        prompt = (f"Документ:\n{context}\n\n"
                  f"Питання: {q['Question']}\n\n"
                  f"Варіанти:\n{options}\n\n"
                  f"Знайди відповідний фрагмент тексту, потім вибери правильний варіант.\n"
                  f"Правильний варіант:")
    else:
        # V1: direct (default)
        prompt = f"{context}\n\nПитання: {q['Question']}\n\nВаріанти:\n{options}\n\nВідповідь (лише буква):"
    resp = llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
    return extract_answer(resp["choices"][0]["text"])


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_context_from_pages(pages, doc_id, top_retrieval, n_pages_total, top_k=3):
    """Build context string from retrieved pages."""
    parts = []
    seen = set()
    for r in top_retrieval[:top_k]:
        key = (r["doc_id"], r["page_num"])
        if key not in seen and r["doc_id"] == doc_id:
            text = pages.get(key, "")
            if text.strip():
                parts.append(f"[Сторінка {r['page_num']}]\n{text[:1200]}")
                seen.add(key)
    # If nothing matched, use top results regardless of doc
    if not parts:
        for r in top_retrieval[:top_k]:
            key = (r["doc_id"], r["page_num"])
            if key not in seen:
                text = pages.get(key, "")
                if text.strip():
                    parts.append(f"[Сторінка {r['page_num']}]\n{text[:1200]}")
                    seen.add(key)
    return "\n\n".join(parts) if parts else "[Контекст недоступний]"


def main():
    parser = argparse.ArgumentParser(description="UNLP 2026 end-to-end pipeline")
    parser.add_argument("--questions", default=str(BASE / "data" / "dev_questions.csv"))
    parser.add_argument("--pdf_dir", default=str(BASE / "data" / "raw_pdfs"))
    parser.add_argument("--output", default=str(BASE / "outputs" / "submission.csv"))
    parser.add_argument("--retrieval", default="bm25", choices=["bm25", "precomputed"])
    parser.add_argument("--precomputed_retrieval", default=None,
                        help="Path to precomputed retrieval JSON")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--context_pages", type=int, default=3)
    parser.add_argument("--strategy", default="direct", choices=["direct", "cot", "evidence"])
    _gguf_candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    _default_gguf = str(_gguf_candidates[0]) if _gguf_candidates else \
        str(BASE / "models" / "mamaylm" / "mamaylm.gguf")
    parser.add_argument("--gguf", default=_default_gguf)
    parser.add_argument("--n_ctx", type=int, default=4096)
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 60)
    print("UNLP 2026 Pipeline")
    print("=" * 60)
    print(f"Questions:  {args.questions}")
    print(f"Retrieval:  {args.retrieval}")
    print(f"Strategy:   {args.strategy}")
    print(f"Top-k:      {args.top_k}")

    # Load questions
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    print(f"\nLoaded {len(questions)} questions")

    # Extract text from all domain PDFs
    pdf_root = Path(args.pdf_dir)
    pdf_dirs = sorted(d for d in pdf_root.iterdir() if d.is_dir())
    print(f"Extracting text from {len(pdf_dirs)} domains...")
    t0 = time.time()
    pages, doc_meta = extract_text_from_pdfs(pdf_dirs)
    print(f"Extracted {len(pages)} pages in {time.time()-t0:.1f}s")

    # Build/load retrieval
    if args.retrieval == "precomputed" and args.precomputed_retrieval:
        print(f"Loading precomputed retrieval: {args.precomputed_retrieval}")
        with open(args.precomputed_retrieval) as f:
            retrieval_data = json.load(f)
    else:
        print("Building BM25 index...")
        t0 = time.time()
        bm25, page_list = build_bm25_index(pages)
        print(f"BM25 index built in {time.time()-t0:.1f}s")
        retrieval_data = None

    # Retrieve pages for each question
    print("Retrieving pages...")
    retrievals = {}
    for q in questions:
        qid = q["Question_ID"]
        if retrieval_data and qid in retrieval_data:
            retrievals[qid] = retrieval_data[qid]
        else:
            retrievals[qid] = retrieve_bm25(bm25, page_list, q["Question"], top_k=args.top_k)

    # Load LLM
    if not Path(args.gguf).exists():
        print(f"ERROR: GGUF not found at {args.gguf}")
        print("Run: python scripts/02_download_models.py --models mamaylm_gguf")
        sys.exit(1)

    llm = load_llm(args.gguf, n_ctx=args.n_ctx)

    # Generate predictions
    print(f"Generating answers ({args.strategy})...")
    predictions = []
    t0 = time.time()

    for i, q in enumerate(questions):
        qid = q["Question_ID"]
        top_results = retrievals[qid][:args.top_k]

        # Best retrieved doc/page
        if top_results:
            pred_doc = top_results[0]["doc_id"]
            pred_page = top_results[0]["page_num"]
        else:
            # Fallback: first doc in same domain
            pred_doc = list(doc_meta.keys())[0] if doc_meta else "unknown.pdf"
            pred_page = 1

        # Build context from retrieved pages
        context = build_context_from_pages(pages, pred_doc, top_results,
                                            doc_meta.get(pred_doc, {}).get("n_pages", 1),
                                            top_k=args.context_pages)

        # Score MCQ
        pred_answer = score_mcq(llm, q, context, strategy=args.strategy)

        predictions.append({
            "Question_ID": qid,
            "Correct_Answer": pred_answer,
            "Doc_ID": pred_doc,
            "Page_Num": str(pred_page),
        })

        if i % 50 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / max(i, 1) * (len(questions) - i)
            print(f"  {i}/{len(questions)} — {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining")

    # Save submission
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Question_ID", "Correct_Answer",
                                                "Doc_ID", "Page_Num"])
        writer.writeheader()
        writer.writerows(predictions)

    total_elapsed = time.time() - total_start
    print(f"\nDone! {len(predictions)} predictions in {total_elapsed:.1f}s")
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()

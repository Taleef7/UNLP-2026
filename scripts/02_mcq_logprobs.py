#!/usr/bin/env python3
"""Phase 2.3: MCQ scoring via llama-cpp-python.

Primary strategy: greedy decode (fast, ~1s/q after warmup)
- Generate 1-3 tokens, extract first A-F letter
- Temperature=0 for deterministic output

Optional: logits_all=True for full probability distribution (slow, ~5-20s/q)

Evaluates with ORACLE retrieval to isolate answer quality from retrieval quality.
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]
# Ukrainian letter mappings (model may output Cyrillic А/Б/В/Г/Д/Е)
UA_TO_LATIN = {"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E", "Ф": "F",
               "а": "A", "в": "B", "с": "C", "д": "D", "е": "E", "ф": "F"}


def _find_gguf():
    candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    return candidates[0] if candidates else BASE / "models" / "mamaylm" / "mamaylm.gguf"

GGUF_PATH = _find_gguf()


def load_llm(n_ctx=2048, logits_all=False):
    from llama_cpp import Llama
    print(f"Loading {GGUF_PATH.name} (n_ctx={n_ctx}, logits_all={logits_all})...")
    start = time.time()
    llm = Llama(
        model_path=str(GGUF_PATH),
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        logits_all=logits_all,
        verbose=False,
    )
    print(f"Loaded in {time.time() - start:.1f}s")
    return llm


def load_oracle_pages():
    text_dir = DATA / "extracted_text"
    pages = {}
    for json_path in sorted(text_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc_id = doc["doc_id"]
        for page in doc["pages"]:
            pages[(doc_id, page["page_num"])] = page["text"]
    return pages


def build_context(question_row, oracle_pages, n_context_pages=2):
    true_doc = question_row["Doc_ID"]
    true_page = int(question_row["Page_Num"])
    n_pages = int(question_row["N_Pages"])
    parts = []
    for offset in range(n_context_pages):
        pn = true_page + offset
        if 1 <= pn <= n_pages:
            text = oracle_pages.get((true_doc, pn), "")
            if text.strip():
                parts.append(text[:1200])
    return "\n\n".join(parts) or "[Контекст недоступний]"


def build_prompt(question_row, context):
    q = question_row
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return (f"{context}\n\n"
            f"Питання: {q['Question']}\n\n"
            f"Варіанти:\n{options}\n\n"
            f"Відповідь (лише буква):")


def extract_answer_from_text(text):
    """Extract A-F letter from generated text."""
    text = text.strip()
    # Direct match
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter):
            return letter
    # Check for Cyrillic equivalents
    for cyrillic, latin in UA_TO_LATIN.items():
        if text.startswith(cyrillic):
            return latin
    # Regex search for standalone letter
    match = re.search(r'\b([A-F])\b', text.upper())
    if match:
        return match.group(1)
    # Last resort: any A-F character
    for char in text.upper():
        if char in "ABCDEF":
            return char
    return "A"  # fallback


def score_greedy(llm, prompt, max_tokens=3):
    """Strategy 1: Greedy decode, extract A-F from output. Fast (~1s/q)."""
    resp = llm(prompt, max_tokens=max_tokens, temperature=0.0, stop=["\n", "."])
    text = resp["choices"][0]["text"]
    return extract_answer_from_text(text)


def score_logprobs(llm, prompt):
    """Strategy 2: Full logprobs over A-F tokens. Slow but robust.
    Requires model initialized with logits_all=True.
    """
    resp = llm(prompt, max_tokens=1, logprobs=50, temperature=0.0)
    top_lp = resp["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0] or {}

    scores = {}
    for letter in ANSWER_CHOICES:
        # Match Latin A-F and Cyrillic equivalents
        for tok in [letter, f" {letter}", f"\n{letter}",
                    f" {letter.lower()}"]:
            if tok in top_lp:
                scores[letter] = float(top_lp[tok])
                break
        if letter not in scores:
            # Cyrillic
            cyrillic = {"A": "А", "B": "Б", "C": "В", "D": "Г", "E": "Д", "F": "Е"}
            for tok in [cyrillic.get(letter, ""), f" {cyrillic.get(letter, '')}"]:
                if tok in top_lp:
                    scores[letter] = float(top_lp[tok])
                    break
        if letter not in scores:
            scores[letter] = -100.0

    return max(scores, key=lambda k: scores[k])


def evaluate_strategy(questions, oracle_pages, llm, strategy="greedy", max_questions=None):
    if max_questions:
        questions = questions[:max_questions]

    correct = 0
    results = []
    start = time.time()

    for i, q in enumerate(questions):
        context = build_context(q, oracle_pages)
        prompt = build_prompt(q, context)

        if strategy == "greedy":
            pred = score_greedy(llm, prompt)
        elif strategy == "logprobs":
            pred = score_logprobs(llm, prompt)
        else:
            pred = score_greedy(llm, prompt)

        is_correct = pred == q["Correct_Answer"]
        if is_correct:
            correct += 1
        results.append({
            "question_id": q["Question_ID"],
            "domain": q.get("Domain", ""),
            "true": q["Correct_Answer"],
            "pred": pred,
            "correct": is_correct,
        })

        if i % 20 == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            remaining = (len(questions) - i) / rate
            print(f"  [{strategy}] {i}/{len(questions)}: "
                  f"acc={correct/(i+1):.4f} | {rate:.1f} q/s | ~{remaining:.0f}s left")

    elapsed = time.time() - start
    accuracy = correct / len(questions) if questions else 0

    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    metrics = {
        "strategy": strategy,
        "n": len(questions),
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "questions_per_sec": len(questions) / elapsed,
        "per_domain": {
            d: sum(r["correct"] for r in dr) / len(dr)
            for d, dr in sorted(by_domain.items())
        },
    }
    return metrics, results


def main():
    if not GGUF_PATH.exists():
        print(f"ERROR: GGUF not found at {GGUF_PATH}")
        print("Run: python scripts/02_download_models.py --models mamaylm_gguf")
        sys.exit(1)

    print("Loading oracle pages...")
    oracle_pages = load_oracle_pages()
    print(f"Loaded {len(oracle_pages)} pages")

    val_path = DATA / "splits" / "val.csv"
    questions_path = val_path if val_path.exists() else DATA / "dev_questions.csv"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    print(f"Questions: {len(questions)}")

    all_results = {}

    # Strategy 1: Greedy (primary — fast)
    print("\n--- Strategy: greedy (fast) ---")
    llm = load_llm(n_ctx=2048, logits_all=False)
    metrics, results = evaluate_strategy(questions, oracle_pages, llm, strategy="greedy")
    all_results["greedy"] = {"metrics": metrics}
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Speed: {metrics['questions_per_sec']:.2f} q/s")
    print(f"  Per domain: {metrics['per_domain']}")
    del llm

    # Summary
    print("\n" + "=" * 60)
    print("MCQ Scoring Results (Oracle Retrieval)")
    print("=" * 60)
    for strategy, data in all_results.items():
        m = data["metrics"]
        print(f"  {strategy:15s}: accuracy={m['accuracy']:.4f}  "
              f"speed={m['questions_per_sec']:.2f}q/s  "
              f"time={m['elapsed_seconds']:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "mcq_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT}/mcq_results.json")


if __name__ == "__main__":
    main()

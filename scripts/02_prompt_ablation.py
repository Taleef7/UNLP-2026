#!/usr/bin/env python3
"""Phase 2.4: Prompt engineering ablation.

Tests 5 prompt variants on val split with oracle retrieval.
Results feed directly into paper ablation table.
"""

import csv
import json
import os
import time
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


def build_prompt_v1_direct(q, context):
    """V1: Direct — minimal prompt, just context + question + options."""
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return f"""{context}

{q['Question']}
{options}

Відповідь:"""


def build_prompt_v2_cot(q, context):
    """V2: Chain-of-thought reasoning."""
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return f"""Прочитай текст та дай відповідь на питання.

Текст:
{context}

Питання: {q['Question']}

Варіанти:
{options}

Розмірковуй крок за кроком, потім вкажи відповідь: "Відповідь: X"."""


def build_prompt_v3_ukrainian(q, context):
    """V3: Explicit Ukrainian-language instruction."""
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return f"""Ти — асистент, що відповідає на питання виключно українською мовою.

Контекст документа:
{context}

Питання: {q['Question']}

Варіанти відповідей:
{options}

Вибери правильну відповідь та вкажи лише букву:"""


def build_prompt_v4_evidence(q, context):
    """V4: Evidence-first — model must cite relevant text."""
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return f"""Документ:
{context}

Питання: {q['Question']}

Варіанти:
{options}

Знайди відповідний фрагмент тексту, потім вибери правильний варіант.
Правильний варіант:"""


def build_prompt_v5_elimination(q, context):
    """V5: Elimination — rule out wrong answers."""
    options = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return f"""Контекст:
{context}

Питання: {q['Question']}

Варіанти:
{options}

Виключи невірні варіанти і вкажи правильний (лише букву):"""


PROMPT_BUILDERS = {
    "v1_direct": build_prompt_v1_direct,
    "v2_cot": build_prompt_v2_cot,
    "v3_ukrainian": build_prompt_v3_ukrainian,
    "v4_evidence": build_prompt_v4_evidence,
    "v5_elimination": build_prompt_v5_elimination,
}


def load_oracle_pages():
    """Load ground-truth page texts."""
    text_dir = DATA / "extracted_text"
    pages = {}
    for json_path in sorted(text_dir.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for page in doc["pages"]:
            pages[(doc["doc_id"], page["page_num"])] = page["text"]
    return pages


def get_context(q, oracle_pages, n_pages=2):
    """Get oracle context pages."""
    true_doc = q["Doc_ID"]
    true_page = int(q["Page_Num"])
    n_total = int(q["N_Pages"])
    parts = []
    for offset in range(0, n_pages):
        pn = true_page + offset
        if 1 <= pn <= n_total:
            text = oracle_pages.get((true_doc, pn), "")
            if text.strip():
                parts.append(text[:1200])
    return "\n\n".join(parts) if parts else "[No context available]"


def score_greedy(llm, prompt):
    """Score by greedy decode — extract A-F from generated text."""
    import re
    response = llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
    text = response["choices"][0]["text"].strip()
    # Try Latin A-F
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter):
            return letter
    # Try Cyrillic
    UA_MAP = {"А": "A", "В": "B", "С": "C", "Д": "D", "Е": "E", "Ф": "F"}
    for cyr, lat in UA_MAP.items():
        if text.startswith(cyr):
            return lat
    m = re.search(r'\b([A-F])\b', text.upper())
    if m:
        return m.group(1)
    # Fallback: any A-F char
    for c in text.upper():
        if c in "ABCDEF":
            return c
    return "A"


def score_logprobs_greedy(llm, prompt):
    """Alias: same as score_greedy for compatibility."""
    return score_greedy(llm, prompt)


def evaluate_prompt(questions, oracle_pages, llm, prompt_builder, variant_name):
    """Evaluate one prompt variant."""
    correct = 0
    results = []
    start = time.time()

    for i, q in enumerate(questions):
        context = get_context(q, oracle_pages)
        prompt = prompt_builder(q, context)
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

    elapsed = time.time() - start
    from collections import defaultdict
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    metrics = {
        "variant": variant_name,
        "n": len(results),
        "accuracy": correct / len(results) if results else 0,
        "elapsed": elapsed,
        "per_domain": {
            d: sum(r["correct"] for r in dr) / len(dr)
            for d, dr in sorted(by_domain.items())
        },
    }
    return metrics, results


def main():
    from llama_cpp import Llama

    if not GGUF_PATH.exists():
        print(f"ERROR: GGUF not found at {GGUF_PATH}")
        return

    oracle_pages = load_oracle_pages()

    val_path = DATA / "splits" / "val.csv"
    questions_path = val_path if val_path.exists() else DATA / "dev_questions.csv"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    print(f"Evaluating {len(questions)} questions across {len(PROMPT_BUILDERS)} prompt variants")

    llm = Llama(model_path=str(GGUF_PATH), n_gpu_layers=-1, n_ctx=2048, verbose=False)

    all_metrics = {}
    for variant, builder in PROMPT_BUILDERS.items():
        print(f"\n--- {variant} ---")
        metrics, results = evaluate_prompt(questions, oracle_pages, llm, builder, variant)
        all_metrics[variant] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}  Time: {metrics['elapsed']:.1f}s")
        for d, acc in metrics["per_domain"].items():
            print(f"    {d}: {acc:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Prompt Ablation Results")
    print("=" * 60)
    print(f"{'Variant':<20} {'Accuracy':>10}")
    print("-" * 32)
    for variant, m in sorted(all_metrics.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{variant:<20} {m['accuracy']:>10.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "prompt_ablation.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT}/prompt_ablation.json")


if __name__ == "__main__":
    main()

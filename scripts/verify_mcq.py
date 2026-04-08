#!/usr/bin/env python3
"""Quick verification: test MamayLM greedy MCQ on a few questions."""
import csv
import os
import sys
import time
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

def main():
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed")
        sys.exit(1)

    gguf_candidates = list((BASE / "models" / "mamaylm").glob("*.gguf"))
    if not gguf_candidates:
        print("ERROR: No GGUF found")
        sys.exit(1)
    gguf = gguf_candidates[0]
    print(f"Loading {gguf.name}...")

    t0 = time.time()
    llm = Llama(model_path=str(gguf), n_gpu_layers=-1, n_ctx=512, verbose=False)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Test 3 quick prompts
    import re
    prompts = [
        "Питання: Яка столиця України?\nА. Київ\nВ. Харків\nС. Одеса\nD. Варшава\nE. Берлін\nF. Будапешт\nВідповідь (лише буква):",
        "Яка найбільша планета сонячної системи?\nA. Земля\nB. Марс\nC. Юпітер\nD. Сатурн\nE. Уран\nF. Нептун\nВідповідь (лише буква):",
    ]
    ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]

    print("\nTest prompts:")
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        resp = llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
        text = resp["choices"][0]["text"].strip()
        elapsed = time.time() - t0
        # Extract A-F
        pred = "?"
        for letter in ANSWER_CHOICES:
            if text.upper().startswith(letter):
                pred = letter
                break
        if pred == "?":
            m = re.search(r'\b([A-F])\b', text.upper())
            if m: pred = m.group(1)
        print(f"  [{i+1}] Output: '{text}' → Pred: {pred} ({elapsed:.2f}s)")

    # Load a few val questions and test
    val_path = BASE / "data" / "splits" / "val.csv"
    oracle_dir = BASE / "data" / "extracted_text"

    if val_path.exists() and oracle_dir.exists():
        import json
        # Load oracle pages
        oracle_pages = {}
        for json_path in sorted(oracle_dir.glob("*.json")):
            if json_path.name == "manifest.json":
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            for pg in doc["pages"]:
                oracle_pages[(doc["doc_id"], pg["page_num"])] = pg["text"]

        with open(val_path, "r", encoding="utf-8") as f:
            questions = list(csv.DictReader(f))[:5]

        print(f"\nTest on {len(questions)} val questions (oracle retrieval):")
        correct = 0
        for q in questions:
            true_doc = q["Doc_ID"]
            true_page = int(q["Page_Num"])
            context = oracle_pages.get((true_doc, true_page), "")[:800]
            opts = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
            prompt = f"{context}\n\nПитання: {q['Question']}\n\nВаріанти:\n{opts}\n\nВідповідь (лише буква):"
            resp = llm(prompt, max_tokens=3, temperature=0.0, stop=["\n", "."])
            text = resp["choices"][0]["text"].strip()
            pred = "A"
            for letter in ANSWER_CHOICES:
                if text.upper().startswith(letter):
                    pred = letter
                    break
            is_correct = pred == q["Correct_Answer"]
            if is_correct:
                correct += 1
            print(f"  Q{q['Question_ID']}: pred={pred} true={q['Correct_Answer']} {'✓' if is_correct else '✗'}")

        print(f"\nAccuracy: {correct}/{len(questions)} = {correct/len(questions):.2%}")

    print("\n✓ llama-cpp-python working correctly")


if __name__ == "__main__":
    main()

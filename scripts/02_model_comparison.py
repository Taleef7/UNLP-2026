#!/usr/bin/env python3
"""Phase 2.5: Model comparison — MamayLM-12B vs alternatives.

Compares on val split with oracle retrieval:
  - MamayLM-12B-v2.1 Q4_K_M (primary)
  - MamayLM-4B (if available, smaller/faster)
  - Fallback: transformers bitsandbytes NF4 (if llama-cpp unavailable)

Reports: accuracy, tokens/sec, VRAM, total time.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"
os.environ["HF_HOME"] = str(BASE / "cache" / "hf_cache")

ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]

def find_gguf(model_dir, pattern):
    """Find a GGUF file in model_dir matching pattern."""
    candidates = list(Path(model_dir).glob(f"*{pattern}*.gguf"))
    return candidates[0] if candidates else None


MODELS = {
    "mamaylm_12b_q4": {
        "type": "gguf",
        "path": find_gguf(BASE / "models" / "mamaylm", "12B") or
                find_gguf(BASE / "models" / "mamaylm", "Q4_K_M"),
        "hf_repo": "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0-GGUF",
        "n_ctx": 4096,
        "description": "MamayLM Gemma-3 12B Q4_K_M GGUF",
    },
    "mamaylm_4b_q4": {
        "type": "gguf",
        "path": find_gguf(BASE / "models" / "mamaylm", "4B"),
        "hf_repo": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF",
        "n_ctx": 4096,
        "description": "MamayLM Gemma-3 4B Q4_K_M GGUF (faster)",
    },
}


def load_oracle_pages():
    pages = {}
    for jp in sorted((DATA / "extracted_text").glob("*.json")):
        if jp.name == "manifest.json":
            continue
        with open(jp, encoding="utf-8") as f:
            doc = json.load(f)
        for p in doc["pages"]:
            pages[(doc["doc_id"], p["page_num"])] = p["text"]
    return pages


def get_context(q, oracle_pages, n=2):
    parts = []
    for offset in range(n):
        pn = int(q["Page_Num"]) + offset
        text = oracle_pages.get((q["Doc_ID"], pn), "")
        if text.strip():
            parts.append(text[:1200])
    return "\n\n".join(parts) or "[No context]"


def build_prompt(q, context):
    opts = "\n".join(f"{l}. {q[l]}" for l in ANSWER_CHOICES)
    return (f"{context}\n\nПитання: {q['Question']}\n\n"
            f"Варіанти:\n{opts}\n\nВідповідь (лише буква):")


def score_gguf(llm, prompt):
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


def evaluate_gguf_model(model_config, questions, oracle_pages):
    """Evaluate a GGUF model."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return {"error": "llama-cpp-python not installed"}

    path = model_config["path"]
    if not path.exists():
        return {"error": f"Model not found: {path}"}

    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    llm = Llama(model_path=str(path), n_gpu_layers=-1,
                n_ctx=model_config["n_ctx"], verbose=False)
    load_time = time.time() - t_load
    vram_gb = torch.cuda.memory_allocated() / 1024**3

    correct = 0
    t_infer = time.time()
    for q in questions:
        context = get_context(q, oracle_pages)
        prompt = build_prompt(q, context)
        pred = score_gguf(llm, prompt)
        if pred == q["Correct_Answer"]:
            correct += 1
    infer_time = time.time() - t_infer

    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    del llm
    torch.cuda.empty_cache()

    n = len(questions)
    return {
        "description": model_config["description"],
        "n_questions": n,
        "accuracy": correct / n,
        "load_time_s": load_time,
        "infer_time_s": infer_time,
        "questions_per_sec": n / infer_time,
        "vram_model_gb": vram_gb,
        "vram_peak_gb": vram_peak,
        "total_time_s": load_time + infer_time,
    }


def main():
    print("=" * 65)
    print("Model Comparison")
    print("=" * 65)

    oracle_pages = load_oracle_pages()

    val_path = DATA / "splits" / "val.csv"
    with open(val_path if val_path.exists() else DATA / "dev_questions.csv",
              encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    print(f"Questions: {len(questions)}\n")

    results = {}
    for name, config in MODELS.items():
        if not config["path"].exists():
            print(f"[{name}] Skipping — not downloaded")
            continue
        print(f"[{name}] Evaluating {config['description']}...")
        r = evaluate_gguf_model(config, questions, oracle_pages)
        results[name] = r
        if "error" not in r:
            print(f"  Accuracy: {r['accuracy']:.4f}")
            print(f"  Speed:    {r['questions_per_sec']:.2f} q/s")
            print(f"  VRAM:     {r['vram_peak_gb']:.1f} GB")
            print(f"  Time:     {r['total_time_s']:.0f}s total "
                  f"(load={r['load_time_s']:.0f}s, infer={r['infer_time_s']:.0f}s)")
        else:
            print(f"  Error: {r['error']}")

    if results:
        print("\n" + "=" * 65)
        print(f"{'Model':<25} {'Acc':>6} {'q/s':>6} {'VRAM':>7} {'Time':>7}")
        print("-" * 65)
        for name, r in results.items():
            if "error" in r:
                print(f"  {name:<23} ERROR: {r['error']}")
            else:
                print(f"  {name:<23} {r['accuracy']:>6.4f} {r['questions_per_sec']:>6.2f} "
                      f"{r['vram_peak_gb']:>6.1f}G {r['total_time_s']:>6.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/model_comparison.json")


if __name__ == "__main__":
    main()

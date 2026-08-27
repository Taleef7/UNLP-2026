#!/usr/bin/env python3
"""C4 — Why the generation-budget bug is a MULTILINGUAL bug, not a generic one.

Our answering stage capped generation at `max_tokens=3` / `4` (pipeline_shared.py:2036,2043). In
English that is a perfectly reasonable budget for an MCQ answer: `"Answer: A"` is three tokens. The
question this script settles is whether it is reasonable in Ukrainian.

It is not, and the reason is TOKENIZER FERTILITY. Subword vocabularies are fitted overwhelmingly on
English and other high-resource Latin-script text, so the same semantic content costs more tokens in
Ukrainian, Russian, Hindi, Telugu... A generation budget is a budget in TOKENS, but what the developer
reasons about is CONTENT. The two come apart exactly in the languages that can least afford it.

This is the finding that makes artifact #3 a multilingual result rather than a private embarrassment:

    A fixed generation budget, chosen (reasonably) against English, is a SILENT, LANGUAGE-CORRELATED
    CONSTRAINT that binds hardest in low-resource, non-Latin-script languages -- and when it binds,
    a parser with a default-value fallback converts it into a plausible-looking accuracy number
    instead of an error.

We measure, for each language and tokenizer, the token cost of the minimal answer strings a model must
emit, and report which of them fit inside the budgets our own pipeline actually used.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp   # item C10: an artifact must remember the code that wrote it

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The minimal answer surface a model must produce, in each language, for an MCQ answer.
# `bare` = the letter alone; `slot` = the natural "Answer: X" phrasing an instructed model emits.
ANSWERS = {
    "English":    {"bare": "A", "slot": "Answer: A",       "script": "Latin"},
    "Ukrainian":  {"bare": "A", "slot": "Відповідь: А",    "script": "Cyrillic"},
    "Russian":    {"bare": "A", "slot": "Ответ: А",        "script": "Cyrillic"},
    "German":     {"bare": "A", "slot": "Antwort: A",      "script": "Latin"},
    "French":     {"bare": "A", "slot": "Réponse : A",     "script": "Latin"},
    "Spanish":    {"bare": "A", "slot": "Respuesta: A",    "script": "Latin"},
    "Arabic":     {"bare": "A", "slot": "الإجابة: A",       "script": "Arabic"},
    "Hindi":      {"bare": "A", "slot": "उत्तर: A",          "script": "Devanagari"},
    "Bengali":    {"bare": "A", "slot": "উত্তর: A",          "script": "Bengali"},
    "Telugu":     {"bare": "A", "slot": "సమాధానం: A",       "script": "Telugu"},
    "Japanese":   {"bare": "A", "slot": "答え: A",           "script": "Japanese"},
    "Korean":     {"bare": "A", "slot": "답변: A",           "script": "Hangul"},
    "Chinese":    {"bare": "A", "slot": "答案: A",           "script": "Han"},
    "Thai":       {"bare": "A", "slot": "คำตอบ: A",          "script": "Thai"},
    "Swahili":    {"bare": "A", "slot": "Jibu: A",         "script": "Latin"},
}

# A short chain-of-thought preamble: what a reasoning prompt makes the model emit BEFORE the answer.
COT_PREAMBLE = {
    "English":   "Let us think step by step. The passage states that",
    "Ukrainian": "Поміркуймо крок за кроком. У тексті сказано, що",
    "Russian":   "Давайте подумаем шаг за шагом. В тексте сказано, что",
}

# The budgets our own pipeline actually used (pipeline_shared.score_mcq).
BUDGETS = {"completion path (max_tokens=3)": 3, "chat path (max_tokens=4)": 4}


def main():
    ap = argparse.ArgumentParser()
    # The default list USED to name google/gemma-2-2b-it and meta-llama/Llama-3.2-1B-Instruct. Both are
    # GATED on the Hub, both raised OSError, and the loop below caught the exception, printed one line
    # to stderr, and carried on with a SHORTER table. The artifact we actually shipped was produced by
    # an ad-hoc --models override (SmolLM2 + bloom) that nobody recorded -- so the committed default
    # could not reproduce the committed result, and docs/results/C4_answer_budget_fertility.md ended up
    # describing the DEFAULT list in its limitations while its headline table described the OVERRIDE.
    # A swallowed exception is what let those two drift apart in silence.
    # These four are ungated and are the four the paper reports.
    ap.add_argument("--models", default="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0,"
                                        "Qwen/Qwen2.5-1.5B-Instruct,"
                                        "HuggingFaceTB/SmolLM2-360M-Instruct,"
                                        "bigscience/bloom-560m")
    ap.add_argument("--allow-skip", action="store_true",
                    help="Permit a tokenizer to fail to load and be omitted. OFF by default: a "
                         "silently shorter table is how this experiment misreported itself once.")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/answer_budget_fertility.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    report = {"budgets": BUDGETS, "models": {}}

    skipped = []
    for mid in args.models.split(","):
        try:
            tk = AutoTokenizer.from_pretrained(mid.strip())
        except Exception as e:
            if not args.allow_skip:
                sys.exit(f"\n[FATAL] tokenizer {mid.strip()} failed to load: {type(e).__name__}: "
                         f"{str(e).splitlines()[0]}\n"
                         f"Refusing to write a table that silently omits a model. Fix access to the "
                         f"model, drop it from --models, or pass --allow-skip to record the omission "
                         f"explicitly in the artifact.\n")
            print(f"[skip] {mid}: {type(e).__name__}", file=sys.stderr)
            skipped.append({"model": mid.strip(), "error": type(e).__name__})
            continue
        name = mid.strip().split("/")[-1]
        print(f"\n=== {name} ===")
        print(f"{'language':<11}{'script':<12}{'bare':>6}{'slot':>6}  "
              f"{'fits max_tokens=3?':<20}{'fits 4?':<9}")
        rows = {}
        for lang, a in ANSWERS.items():
            n_bare = len(tk.encode(a["bare"], add_special_tokens=False))
            n_slot = len(tk.encode(a["slot"], add_special_tokens=False))
            fits3 = n_slot <= 3
            fits4 = n_slot <= 4
            rows[lang] = {"script": a["script"], "slot": a["slot"],
                          "tokens_bare": n_bare, "tokens_slot": n_slot,
                          "fits_3": fits3, "fits_4": fits4}
            print(f"{lang:<11}{a['script']:<12}{n_bare:>6}{n_slot:>6}  "
                  f"{('YES' if fits3 else 'NO'):<20}{('YES' if fits4 else 'NO'):<9}")

        en = rows["English"]["tokens_slot"]
        infl = {l: rows[l]["tokens_slot"] / en for l in rows}
        latin = [v for l, v in infl.items() if rows[l]["script"] == "Latin"]
        nonlat = [v for l, v in infl.items() if rows[l]["script"] != "Latin"]
        print(f"\n  token cost of the answer slot, relative to English:")
        print(f"    Latin-script languages     : mean {np.mean(latin):.2f}x")
        print(f"    non-Latin-script languages : mean {np.mean(nonlat):.2f}x")
        n_fail3 = sum(1 for r in rows.values() if not r["fits_3"])
        print(f"  languages whose answer slot does NOT fit our max_tokens=3 budget: "
              f"{n_fail3}/{len(rows)}")

        cot = {}
        for lang, txt in COT_PREAMBLE.items():
            cot[lang] = len(tk.encode(txt, add_special_tokens=False))
        if cot:
            print(f"  CoT preamble length (tokens BEFORE the model can reach its answer):")
            for lang, n in cot.items():
                print(f"    {lang:<11}{n:>4}   -> exceeds a 3-4 token budget by {n/3.5:.0f}x")

        report["models"][name] = {"rows": rows, "cot_preamble_tokens": cot,
                                  "mean_inflation_latin": float(np.mean(latin)),
                                  "mean_inflation_nonlatin": float(np.mean(nonlat)),
                                  "n_langs_failing_budget3": n_fail3, "n_langs": len(rows)}

    # Record what was actually run, so the artifact cannot disagree with the invocation again.
    report["_requested_models"] = [m.strip() for m in args.models.split(",")]
    if skipped:
        report["_skipped_models"] = skipped
    json.dump(stamp(report, "answer_budget_fertility.py"), open(args.out, "w"),
              ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}  ({len(report['models'])} model(s)"
          + (f", {len(skipped)} SKIPPED" if skipped else "") + ")")


if __name__ == "__main__":
    main()

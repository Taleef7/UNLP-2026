#!/usr/bin/env python3
"""End-to-end run on committed code, DETERMINISTIC, with the answer-extraction path instrumented.

WHY THIS SHAPE. We cannot re-run "the shipped system": v7_full_dev ran 2026-03-26 and the repository's
first commit is 2026-04-08, so the code that produced 0.8634 / 0.8722 was never version controlled and
no longer exists (models and corpus ARE byte-identical; the code is not). Any absolute comparison
against 0.8634 would be comparing two different systems, and b5_analyze.py records that refusal.

What we CAN do is better: establish a reproducible baseline on committed code and measure our fixes
against IT. B5 showed greedy (vote_temp=0) is exactly deterministic -- 0/461 answer disagreements across
two different seeds -- so sigma is 0 and ANY delta between two greedy runs is real, not noise. That is a
strictly stronger footing than the number we lost.

WHAT WE THOUGHT THE BUG WAS, AND WHAT IT ACTUALLY IS.

We expected the GENERATION BUDGET (the token-budget hypothesis). The completion path was hardcoded to
`max_tokens=3`; "Answer: A" is exactly 3 tokens in English under every tokenizer we measured, and 5
under MamayLM's Ukrainian-adapted vocabulary. Tidy story. We ran budget=3 against budget=8, both
deterministic.

**The two runs came out byte-identical, and `truncated == 0` in BOTH.** The budget NEVER BINDS. N6's
arithmetic is true about the tokenizer and simply does not operate here. That is a negative result and
we report it as one rather than keep the tidier story.

The real defect was in the same three lines and we had walked past it. The v4_evidence prompt ends with
`"Правильний варіант:"` -- no trailing space. The model's natural continuation is `"\nA"`. But `"\n"` is
IN THE STOP LIST, so llama.cpp halts at token ZERO and returns the EMPTY STRING with
`finish_reason="stop"` -- not "length", which is why no truncation check would ever have caught it.
`extract_answer("")` then falls through every branch to its unconditional `return "A"`.

Measured, deterministically: **36 of 1383 calls return "", all of them, across 13 questions. We answered
"A" to all 13.** Their gold answers are A,A,B,C,C,D,D,D,E,E,F,F,F -- two right by luck, banked as
knowledge (+0.43pp of our reported accuracy).

It stayed invisible because three things hid it at once: the model appeared to answer, "A" is a
plausible answer, and Zheng et al. (2024) show LLMs are ALREADY biased toward option A -- so a parser
failure is camouflaged as a known model behaviour.

THE CELLS (all greedy => sigma is exactly 0 by B5 GATE 2, so every delta is real):
    e2e_b3    budget=3, stop=["\n","."]   the shipped configuration
    e2e_b8    budget=8, stop=["\n","."]   the budget hypothesis -- a NO-OP, reported as such
    e2e_fix   budget=8, stop=["."]        the actual fix: let the model speak

`answer_trace.json` records, per call: the raw generation, WHICH extractor branch fired, and the
finish_reason. That is what turned "the fallback is probably firing" into a count, and it lets the
abstain counterfactual be computed offline instead of costing another GPU run.
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))

PRESET = "v7_baseline"
QUESTIONS = REPO / "data/dev_questions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vote-temp", type=float, default=0.0,
                    help="0.0 = greedy = deterministic (B5 GATE 2 proved 0/461 churn across seeds)")
    ap.add_argument("--max-tokens", type=int, default=3,
                    help="completion-path budget. 3 = the shipped constant; 8 = enough for Ukrainian.")
    ap.add_argument("--stop-on-newline", dest="stop_nl", action="store_true", default=True,
                    help="shipped behaviour: \\n is a stop token (and fires at token 0 -> empty -> 'A')")
    ap.add_argument("--no-stop-on-newline", dest="stop_nl", action="store_false",
                    help="THE FIX: let the model emit its leading newline and actually answer")
    ap.add_argument("--reset-kv", dest="reset_kv", action="store_true", default=False,
                    help="THE SECOND FIX: llama.py:882 cancels its own reset on a prefix cache hit, so "
                         "votes 2-3 re-evaluate ONE token against a warm cache while vote 1 evaluates "
                         "the full prompt. Different logits, same prompt. This forces a true reset.")
    ap.add_argument("--passes", type=int, default=None,
                    help="override base_passes. With vote_temp=0 all passes are identical, so 1 is "
                         "equivalent and 3x cheaper; default leaves the preset alone.")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    from pipeline_shared import run_pipeline_from_preset

    out = REPO / "outputs/benchmarks" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    llm = {"seed": args.seed, "vote_temp": args.vote_temp,
           "max_tokens_completion": args.max_tokens, "stop_on_newline": args.stop_nl,
           "reset_kv_cache": args.reset_kv}
    if args.passes is not None:
        llm["base_passes"] = args.passes
    overrides = {"llm": llm}

    print(f"[e2e] preset={PRESET} {llm} -> {out}", flush=True)
    run_pipeline_from_preset(
        PRESET, questions_path=QUESTIONS, output_dir=out, env="local",
        n_questions=args.n, overrides=overrides,
        run_metadata={"e2e_tag": args.tag, **{f"e2e_{k}": v for k, v in llm.items()}},
    )

    s = json.load(open(out / "summary.json"))
    keys = ("answer_accuracy", "doc_accuracy", "page_proximity", "composite_score")
    missing = [k for k in keys if s.get(k) is None]
    if missing:
        sys.exit(f"[FATAL] summary.json missing {missing} -- refusing to record an empty run")

    tr = json.load(open(out / "answer_trace.json"))
    calls = tr["calls"]
    fb = sum(1 for c in calls if c["branch"] == "FALLBACK_A")
    tc = sum(1 for c in calls if c["truncated"])
    print(f"[e2e] " + "  ".join(f"{k}={s[k]:.6f}" for k in keys), flush=True)
    print(f"[e2e] calls={len(calls)}  truncated={tc} ({tc/max(len(calls),1):.1%})  "
          f"FALLBACK_A={fb} ({fb/max(len(calls),1):.1%})", flush=True)

    json.dump({**{k: s[k] for k in keys},
               "tag": args.tag, "seed": args.seed, "vote_temp": args.vote_temp,
               "max_tokens_completion": args.max_tokens,
               "n_questions": s["n_questions"],
               "n_calls": len(calls), "n_truncated": tc, "n_fallback_A": fb},
              open(out / "e2e_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()

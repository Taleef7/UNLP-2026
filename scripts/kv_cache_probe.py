#!/usr/bin/env python3
"""Why do three IDENTICAL greedy calls on the SAME prompt return different text?

Observed in e2e_fix (vote_temp=0, so all three passes are temperature=0.0 on a byte-identical prompt):

    qid  62: [' B', '\\nB', '\\nB']     call 1 leads with a SPACE, calls 2-3 lead with a NEWLINE
    qid 237: [' D', '\\nD', '\\nD']
    qid 408: [' F', ' A',  ' A']       calls 2-3 return a DIFFERENT LETTER entirely

That must not happen. Greedy decoding is a deterministic function of the prompt, so either the prompt
differs between calls (it does not -- `raw_prompt` is computed once in score_mcq and reused) or the
model is not in the same state. The hypothesis is llama.cpp's PROMPT CACHE: on the second call the
entire prompt is already in the KV cache, llama-cpp-python evaluates zero or one new tokens, and
sampling proceeds from a state that is not identical to the freshly-evaluated one.

This matters well beyond a cosmetic quirk:

  * In the SHIPPED configuration "\\n" is a stop token, so a call that leads with a newline halts at
    token ZERO and returns "". `extract_answer("")` returns "A". So calls 2 and 3 both become "A" and,
    under `Counter(votes).most_common`, TWO EMPTY STRINGS OUTVOTE THE MODEL'S CORRECT FIRST ANSWER.
  * On qid 408 there is no empty string at all: the model itself drifts F -> A -> A, and the vote lets
    the two drifted calls beat the correct one.
  * It means the three "votes" are not three draws from one distribution. Vote 1 and votes 2-3 are
    drawn from different states. The ensemble is not an ensemble.

THIS SCRIPT DOES NOT ASSUME THE CAUSE. It measures it. For each of the four affected questions it runs
the identical prompt three times under each of two regimes:

    A) as the pipeline does it -- three back-to-back calls, cache left alone
    B) with `llm.reset()` before every call -- cache cleared, each call sees a fresh state

If the cache is the cause, regime A reproduces the observed disagreement and regime B returns three
identical strings. If regime B ALSO disagrees, the cause is something else and the hypothesis is dead.
Either way we report what we measured. A control question (three calls that already agree) is included
so that a "reset makes everything identical" result cannot be trivially true.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "scripts"))
from provenance import stamp

OUT = REPO / "outputs/kv_cache_probe.json"
# The four questions where three identical greedy calls disagreed, plus two that agreed (controls).
AFFECTED = ["62", "237", "408", "447"]
CONTROLS = ["1", "2"]


def main():
    import csv
    from pipeline_shared import PipelineRunner, build_prompt, load_preset, extract_answer_traced

    rows = {r["Question_ID"]: r for r in csv.DictReader(open(REPO / "data/dev_questions.csv"))}
    qids = AFFECTED + CONTROLS

    # Rebuild the exact context each question saw, by replaying the ranking the e2e run recorded.
    ctx = {}
    with open(REPO / "outputs/benchmarks/e2e_fix/ranking_details.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q = str(d.get("Question_ID", d.get("qid", "")))
            if q in qids and "context" in d:
                ctx[q] = d["context"]
    missing = [q for q in qids if q not in ctx]
    if missing:
        print(f"[probe] ranking_details.jsonl has no `context` field for {missing}; "
              f"keys present: {sorted(d.keys())}", flush=True)
        sys.exit("[probe] cannot reconstruct the prompt without the context -- aborting rather than "
                 "probing a DIFFERENT prompt and reporting it as the same one")

    preset = load_preset("v7_baseline")
    runner = PipelineRunner(preset, env="local")
    runner.load_llm()
    llm = runner.llm
    cap = int(preset["llm"].get("max_tokens_completion", 3))

    rep = {"cap": cap, "questions": {}}
    for q in qids:
        prompt = build_prompt(rows[q], ctx[q], preset["llm"].get("prompt_variant", "v4_evidence"))
        out = {"gold": rows[q]["Correct_Answer"], "affected": q in AFFECTED}

        # regime A: exactly what the pipeline does -- three back-to-back calls, cache untouched
        a = []
        for _ in range(3):
            r = llm(prompt, max_tokens=cap, temperature=0.0, stop=["\n", "."])
            a.append(r["choices"][0]["text"])
        out["no_reset"] = a
        out["no_reset_identical"] = (a[0] == a[1] == a[2])

        # regime B: clear the KV cache before every call
        b = []
        for _ in range(3):
            llm.reset()
            r = llm(prompt, max_tokens=cap, temperature=0.0, stop=["\n", "."])
            b.append(r["choices"][0]["text"])
        out["with_reset"] = b
        out["with_reset_identical"] = (b[0] == b[1] == b[2])

        out["letters_no_reset"] = [extract_answer_traced(t)[0] for t in a]
        out["letters_with_reset"] = [extract_answer_traced(t)[0] for t in b]
        rep["questions"][q] = out
        print(f"[probe] qid {q:>4} gold={out['gold']}  "
              f"no_reset={a!r} identical={out['no_reset_identical']}  "
              f"with_reset={b!r} identical={out['with_reset_identical']}", flush=True)

    aff = [rep["questions"][q] for q in AFFECTED]
    ctl = [rep["questions"][q] for q in CONTROLS]
    rep["verdict"] = {
        "affected_disagree_without_reset": sum(1 for x in aff if not x["no_reset_identical"]),
        "affected_disagree_with_reset": sum(1 for x in aff if not x["with_reset_identical"]),
        "controls_disagree_without_reset": sum(1 for x in ctl if not x["no_reset_identical"]),
        "n_affected": len(aff), "n_controls": len(ctl),
    }
    v = rep["verdict"]
    cache_is_cause = (v["affected_disagree_without_reset"] > 0
                      and v["affected_disagree_with_reset"] == 0)
    rep["verdict"]["kv_cache_is_the_cause"] = cache_is_cause
    rep["verdict"]["note"] = (
        "CONFIRMED: identical greedy calls disagree when the prompt cache is warm and agree when it is "
        "reset. The three 'votes' are not three draws from one distribution."
        if cache_is_cause else
        "NOT CONFIRMED: resetting the cache did not make the calls agree (or they agreed anyway). The "
        "KV-cache hypothesis does not explain the disagreement and must not be reported as if it did.")
    print(f"\n[probe] {rep['verdict']['note']}", flush=True)
    json.dump(stamp(rep, "kv_cache_probe.py"), open(OUT, "w"), indent=2)
    print(f"[probe] saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

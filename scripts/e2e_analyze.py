#!/usr/bin/env python3
"""The empty-string bug: what it cost us, and what fixing it buys. Four deterministic runs.

    e2e_b3     budget=3,  stop=["\\n","."]   the shipped configuration
    e2e_b8     budget=8,  stop=["\\n","."]   the answer-budget hypothesis
    e2e_fix    budget=8,  stop=["."]         the ACTUAL fix
    e2e_fix64  budget=64, stop=["."]         does a REAL budget close the last 6? (it does not)

vote_temp=0 throughout, and B5 GATE 2 established that greedy has 0/461 answer churn across two
different seeds -- sigma is exactly 0, so every delta below is real rather than noise. That is the whole
reason this comparison is worth anything, and it is why we measured sigma before running it.

Written to survive a hostile reviewer: their objections run as CHECKS, rather than waiting to be raised.

  CHECK 1  REGRESSION. e2e_b3 is bit-for-bit b5_g201's configuration, and greedy is seed-independent, so
           it must reproduce b5_g201 EXACTLY. If it does not, the answer-trace instrumentation changed
           the shipped code path and nothing downstream is trustworthy.

  CHECK 2  IS THE BUDGET CONTRAST SURGICAL? `stop` ends generation at the first newline/period, so
           raising the cap can only move calls that were TRUNCATED. Every untruncated generation must be
           byte-identical between b3 and b8. A reviewer will ask; "it shouldn't change" is not an answer.

  CHECK 3  DOES THE BUDGET HYPOTHESIS EVEN APPLY? It requires truncation to be happening. If
           truncated == 0 in the baseline, the mechanism does not operate and the budget contrast is
           VACUOUS -- it cannot help and its null result says nothing about the token-budget hypothesis's arithmetic.
           (An earlier version of this check tested `truncated_fixed <= truncated_base` and PASSED on
           0 <= 0. It reported success for an experiment that could not have failed. Exactly the
           degenerate-test failure this project keeps finding in its own work -- see A1/A2.)

  CHECK 4  DOES THE FIX REACH THE MECHANISM? The claim is: stop-on-newline fires at token 0 -> empty
           string -> FALLBACK_A. So removing "\\n" from the stop list must drive the empty-string count,
           and FALLBACK_A, toward zero. If the fallbacks persist, the diagnosis is wrong no matter what
           the accuracy does.

  CHECK 5  ARE THE 6 RESIDUAL FALLBACKS BUDGET-BOUND? After the fix, the 6 remaining FALLBACK_A calls all
           finish with reason="length" -- so the obvious reading is that the model started quoting the
           document and ran out of room. e2e_fix64 gives it 8x the budget. If that reading is right,
           truncation AND the fallbacks both go to zero. It is NOT right: truncation goes to zero and the
           fallbacks DO NOT MOVE. Given room, the model finishes its sentence and still never emits a
           letter. The budget was never the binding constraint -- not at 3, not at 8, not at 64. the token-budget hypothesis is
           refuted a second time, on its strongest possible test.

  CHECK 6  THE FIRST-VS-LAST PARSER HAZARD IN extract_answer, MEASURED NOT ASSUMED.
           `extract_answer` takes the FIRST \\b[A-F]\\b. Under an evidence-then-answer prompt the answer is
           LAST, so 'Текст містить пункт B. Правильна відповідь: E' would parse to B. Before running
           e2e_fix64 we predicted a longer budget might expose this. It did not: ZERO generations in ANY
           cell contain more than one Latin [A-F]. The bug is real in the code and UNEXPRESSED in the run.
           We report it as a latent hazard. We do not claim to have fixed it, and we do not claim it is
           harmless -- only that under this prompt the model never writes text that triggers it.

  CHECK 7  THE HOMOGLYPH HYPOTHESIS, KILLED. Plan Step 5.2 proposed measuring how often MamayLM emits
           Cyrillic А/В/С/Е instead of Latin A/B/C/E -- a silent, script-correlated eval failure, and one
           of the two experiments meant to earn the "language-correlated" framing. The `prefix_cyrillic`
           extractor branch fires ZERO times in every cell. It does not happen. Reported as a null.
"""
import collections, csv, json, re, sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from provenance import stamp

B = REPO / "outputs/benchmarks"
OUT = REPO / "outputs/e2e_stop_bug.json"
SHIPPED, BUDGET, FIX, FIX64, REGRESS = "e2e_b3", "e2e_b8", "e2e_fix", "e2e_fix64", "b5_g201"
RESET = "e2e_reset"          # optional 5th cell; CHECK 10 runs only once it exists
CELLS = (SHIPPED, BUDGET, FIX, FIX64)
# Pre-registered and committed BEFORE the job was submitted.
PREREG_DISAGREE_QIDS = ["62", "237", "408", "447"]
PREREG_P3_QID, PREREG_P3_LETTER = "408", "F"      # greedy says F, gold is F, the warm-cache votes say A
PREREG_P4_ANSWER_ACC = 0.8872017353579176         # call-1-only accuracy of e2e_fix. SOFT.
PREREG_P5_FALLBACK_A = 6                          # qids 197 and 203 answer in prose. SOFT.
SHIPPED_DOC_ACC = 0.9327548806941431              # llm is used ONLY in score_mcq -> retrieval untouched
METRICS = ("answer_accuracy", "doc_accuracy", "page_proximity", "composite_score")
LATIN = re.compile(r"\b([A-F])\b")


def summary(t): return json.load(open(B / t / "summary.json"))
def trace(t):   return json.load(open(B / t / "answer_trace.json"))["calls"]
def perq(t):    return {r["Question_ID"]: r for r in csv.DictReader(open(B / t / "per_question.csv"))}


def mcnemar(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def stats(t):
    c = trace(t)
    multi = [x for x in c if len(LATIN.findall(x["text"])) > 1]
    return {
        "n_calls": len(c),
        "empty": sum(1 for x in c if x["text"].strip() == ""),
        "truncated": sum(1 for x in c if x["truncated"]),
        "fallback_A": sum(1 for x in c if x["branch"] == "FALLBACK_A"),
        "cyrillic_letter_answers": sum(1 for x in c if x["branch"] == "prefix_cyrillic"),
        "generations_with_multiple_letters": len(multi),
        "of_those_first_ne_last": sum(1 for x in multi
                                      if LATIN.findall(x["text"])[0] != LATIN.findall(x["text"])[-1]),
    }


def main():
    for t in CELLS:
        if not (B / t / "e2e_result.json").exists():
            sys.exit(f"[e2e] {t} not finished")
    rep, S, T, P = {}, {}, {}, {}
    for t in CELLS:
        S[t], T[t], P[t] = summary(t), stats(t), perq(t)
    print("=" * 88)

    # ---- CHECK 1: regression ----
    sr, pr = summary(REGRESS), perq(REGRESS)
    common = sorted(set(P[SHIPPED]) & set(pr))
    nd = sum(1 for q in common if P[SHIPPED][q]["pred_answer"] != pr[q]["pred_answer"])
    same = all(abs(S[SHIPPED][m] - sr[m]) < 1e-12 for m in METRICS)
    ok1 = same and nd == 0
    rep["check1_regression"] = {"vs": REGRESS, "metrics_identical": same,
                               "answer_disagreements": nd, "n": len(common), "pass": ok1}
    print(f"CHECK 1  regression: {SHIPPED} must reproduce {REGRESS} exactly")
    print(f"         composite {S[SHIPPED]['composite_score']:.6f} vs {sr['composite_score']:.6f}   "
          f"per-question disagreements {nd}/{len(common)}   -> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        sys.exit("instrumentation changed the shipped code path -- stop")

    # ---- CHECK 2: budget contrast surgical ----
    cb, cf = trace(SHIPPED), trace(BUDGET)
    moved = [i for i, (a, b) in enumerate(zip(cb, cf))
             if not a["truncated"] and a["text"] != b["text"]]
    ok2 = (len(cb) == len(cf)) and not moved
    rep["check2_budget_surgical"] = {"untruncated_that_moved": len(moved), "pass": ok2}
    print(f"\nCHECK 2  budget contrast surgical? untruncated generations must be byte-identical")
    print(f"         calls={len(cb)}   untruncated generations that changed: {len(moved)}   "
          f"-> {'PASS' if ok2 else 'FAIL (confounded)'}")

    # ---- CHECK 3: does the budget hypothesis even apply? ----
    tr_base = T[SHIPPED]["truncated"]
    ok3 = tr_base > 0
    rep["check3_budget_hypothesis_applies"] = {
        "truncated_in_baseline": tr_base, "applies": ok3,
        "note": ("VACUOUS: nothing was ever truncated, so the budget cannot bind and its null result "
                 "is not evidence about N6's tokenizer arithmetic -- only that the stop sequence fires "
                 "first." if not ok3 else "budget can bind"),
    }
    print(f"\nCHECK 3  does the BUDGET hypothesis even apply? (needs truncation > 0 in the baseline)")
    print(f"         truncated in {SHIPPED}: {tr_base}   -> "
          f"{'applies' if ok3 else 'VACUOUS -- the budget can never bind; the contrast is uninformative'}")
    if not ok3:
        print(f"         The budget contrast is a NULL BY CONSTRUCTION, not a refutation of N6's")
        print(f"         arithmetic. We report it as a negative result, not as evidence for anything.")

    # ---- CHECK 4: does the fix reach the mechanism? ----
    ok4 = T[FIX]["empty"] < T[SHIPPED]["empty"] and T[FIX]["fallback_A"] < T[SHIPPED]["fallback_A"]
    rep["check4_fix_reaches_mechanism"] = {"cells": {t: T[t] for t in CELLS}, "pass": ok4}
    print(f"\nCHECK 4  does removing '\\n' from the stop list reach the mechanism?")
    print(f"         {'cell':<11}{'calls':>7}{'empty':>7}{'trunc':>7}{'FALLBACK_A':>12}{'cyrillic':>10}")
    for t in CELLS:
        print(f"         {t:<11}{T[t]['n_calls']:>7}{T[t]['empty']:>7}{T[t]['truncated']:>7}"
              f"{T[t]['fallback_A']:>12}{T[t]['cyrillic_letter_answers']:>10}")
    print(f"         -> {'PASS -- empty strings and fallbacks both collapse' if ok4 else 'FAIL -- diagnosis is wrong'}")

    # ---- CHECK 5: are the 6 residual fallbacks budget-bound? ----
    tf, t64 = T[FIX], T[FIX64]
    budget_bound = t64["fallback_A"] < tf["fallback_A"]
    freed = [i for i, (a, b) in enumerate(zip(trace(FIX), trace(FIX64))) if a["text"] != b["text"]]
    rep["check5_residual_fallbacks_budget_bound"] = {
        "fallback_A_at_budget_8": tf["fallback_A"], "fallback_A_at_budget_64": t64["fallback_A"],
        "truncated_at_budget_8": tf["truncated"], "truncated_at_budget_64": t64["truncated"],
        "generations_that_grew": len(freed),
        "metrics_identical_to_budget_8": all(abs(S[FIX64][m] - S[FIX][m]) < 1e-12 for m in METRICS),
        "budget_bound": budget_bound,
        "note": ("The budget was never the binding constraint -- not at 3, not at 8, not at 64. Given 8x "
                 "the room the model finishes its sentence and STILL never emits a letter: it answers by "
                 "quoting the document in Ukrainian prose. This refutes N6 a second time, on its "
                 "strongest test." if not budget_bound else "budget did bind after all"),
    }
    print(f"\nCHECK 5  are the 6 residual FALLBACK_A calls budget-bound? (all finish_reason='length' at 8)")
    print(f"         budget  8 -> truncated {tf['truncated']:>2}   FALLBACK_A {tf['fallback_A']:>2}")
    print(f"         budget 64 -> truncated {t64['truncated']:>2}   FALLBACK_A {t64['fallback_A']:>2}   "
          f"({len(freed)} generations grew; metrics identical: "
          f"{rep['check5_residual_fallbacks_budget_bound']['metrics_identical_to_budget_8']})")
    print(f"         -> {'budget-bound' if budget_bound else 'NO. Truncation goes to 0 and the fallbacks DO NOT MOVE.'}")
    if not budget_bound:
        print(f"         Given room, the model finishes its sentence and still emits no letter -- it")
        print(f"         answers by quoting the document. The budget never bound: not at 3, 8, or 64.")

    # ---- CHECK 6: the first-vs-last parser hazard (E3), measured ----
    hz = {t: (T[t]["generations_with_multiple_letters"], T[t]["of_those_first_ne_last"]) for t in CELLS}
    expressed = any(v[1] for v in hz.values())
    rep["check6_first_vs_last_hazard"] = {
        "generations_with_multiple_letters": {t: hz[t][0] for t in CELLS},
        "of_those_where_first_differs_from_last": {t: hz[t][1] for t in CELLS},
        "expressed": expressed,
        "note": ("LATENT, NOT EXPRESSED. extract_answer takes the FIRST \\b[A-F]\\b, which is wrong for an "
                 "evidence-then-answer generation. No generation in any cell contains two different "
                 "letters, so the rule is never exercised. The code is still wrong; we report it as an "
                 "unexpressed hazard rather than claiming it is fixed or harmless."),
    }
    print(f"\nCHECK 6  the first-vs-last parser hazard (E3) -- measured, not assumed")
    for t in CELLS:
        print(f"         {t:<11} generations with >1 latin [A-F]: {hz[t][0]:>3}   "
              f"of those, first != last: {hz[t][1]:>3}")
    print(f"         -> {'EXPRESSED -- the parser is taking letters out of the evidence' if expressed else 'LATENT: never exercised. Wrong in code, silent in this run.'}")

    # ---- CHECK 7: the homoglyph hypothesis ----
    cyr = {t: T[t]["cyrillic_letter_answers"] for t in CELLS}
    total = sum(T[t]["n_calls"] for t in CELLS)
    rep["check7_homoglyph_hypothesis"] = {
        "prefix_cyrillic_hits": cyr, "total_calls_across_cells": total,
        "occurs": any(cyr.values()),
        "note": ("NULL. Plan Step 5.2 proposed the homoglyph experiment as one of two pillars of the "
                 "'language-correlated' framing: MamayLM emits Cyrillic А/В/С/Е, the Latin-only parser "
                 "misses it, accuracy silently drops. It does not happen -- 0 hits in every cell. The "
                 "hypothesis is dead and we report it as a null."),
    }
    print(f"\nCHECK 7  homoglyph answers (Cyrillic А/В/С/Е instead of Latin A/B/C/E)?")
    print(f"         prefix_cyrillic branch hits across all {total} calls: {sum(cyr.values())}")
    print(f"         -> {'occurs' if any(cyr.values()) else 'NULL -- it never happens. Plan Step 5.2 is dead; reported as a negative result.'}")

    # ---- CHECK 8: the warm-cache signature ----
    # llama.py:882 cancels its own reset on a prefix hit. Calls 2 and 3 therefore share ONE code path
    # (evaluate 1 token against a fully warm cache) while call 1 evaluates a divergent suffix. That is a
    # falsifiable prediction, not a story: calls 2 and 3 must ALWAYS agree; only call 1 may differ.
    sig = {}
    for t in CELLS:
        byq = {}
        for c in trace(t):
            byq.setdefault(c["qid"], []).append(c)
        sig[t] = {
            "c2_vs_c3_disagree": sum(1 for cs in byq.values() if cs[1]["text"] != cs[2]["text"]),
            "c1_differs_from_c2": sum(1 for cs in byq.values() if cs[0]["text"] != cs[1]["text"]),
            "n_questions": len(byq),
        }
    holds = all(v["c2_vs_c3_disagree"] == 0 for v in sig.values())
    rep["check8_warm_cache_signature"] = {
        "per_cell": sig, "prediction_holds": holds,
        "note": ("Calls 2 and 3 never once disagree, in any cell. They are not independent samples -- "
                 "they are numerically identical twins produced by the same warm-cache code path, while "
                 "call 1 takes a different one. The three 'votes' are not three draws from one "
                 "distribution." if holds else
                 "FALSIFIED: calls 2 and 3 disagree somewhere, so the warm-cache account is wrong."),
    }
    print(f"\nCHECK 8  the warm-cache signature (llama.py:882 cancels its own reset on a prefix hit)")
    print(f"         prediction: calls 2 and 3 share the warm path -> must ALWAYS agree; only call 1 differs")
    for t in CELLS:
        s = sig[t]
        print(f"         {t:<11} calls 2 vs 3 disagree: {s['c2_vs_c3_disagree']:>3}/{s['n_questions']}"
              f"   call 1 differs from 2-3: {s['c1_differs_from_c2']:>3}/{s['n_questions']}")
    print(f"         -> {'HOLDS across ' + str(sum(v['n_questions'] for v in sig.values())) + ' vote-triples' if holds else 'FALSIFIED'}")

    # ---- CHECK 9: does the 3-pass vote ever earn its cost? ----
    print(f"\nCHECK 9  does the 3-pass vote ever earn its keep? (`Counter.most_common`: calls 2-3 can")
    print(f"         only overrule call 1 by AGREEING with each other -- which the warm cache guarantees)")
    gold = {q: r["true_answer"] for q, r in P[SHIPPED].items()}
    for t in CELLS:
        byq = {}
        for c in trace(t):
            byq.setdefault(c["qid"], []).append(c)
        n = len(byq)
        over = [(q, cs[0]["letter"],
                 collections.Counter([c["letter"] for c in cs]).most_common(1)[0][0])
                for q, cs in byq.items()
                if collections.Counter([c["letter"] for c in cs]).most_common(1)[0][0] != cs[0]["letter"]]
        broke = sum(1 for q, g, w in over if g == gold[q])
        saved = sum(1 for q, g, w in over if w == gold[q])
        acc_vote = sum(1 for q, cs in byq.items()
                       if collections.Counter([c["letter"] for c in cs]).most_common(1)[0][0] == gold[q]) / n
        acc_g1 = sum(1 for q, cs in byq.items() if cs[0]["letter"] == gold[q]) / n
        rep.setdefault("check9_vote_integrity", {})[t] = {
            "vote_overruled_call1": len(over), "vote_was_right": saved,
            "vote_destroyed_a_correct_answer": broke,
            "accuracy_3vote": acc_vote, "accuracy_call1_only": acc_g1,
            "delta_pp_of_dropping_the_vote": (acc_g1 - acc_vote) * 100,
        }
        print(f"         {t:<11} overruled call 1 on {len(over)} question(s); right {saved} time(s), "
              f"destroyed a correct answer {broke} time(s)")
        print(f"         {'':<11} 3-vote {acc_vote:.6f}  vs  call-1 only {acc_g1:.6f}   "
              f"({(acc_g1 - acc_vote) * 100:+.2f}pp, and 3x cheaper)")
    rep["check9_vote_integrity"]["note"] = (
        "READ THE RETRACTION BELOW BEFORE USING THESE NUMBERS. The 'call-1 only' column here treats call 1 "
        "as the model's true greedy answer. IT IS NOT. e2e_reset proved call 1 is ITSELF a prefix-cache "
        "hit against the PREVIOUS question's prompt, so it is contaminated too -- just differently. The "
        "clean single-greedy-call baseline is e2e_reset, not this column.")
    rep["check9_vote_integrity"]["RETRACTED_CLAIM"] = {
        "claim": "On qid 408 the vote DESTROYED A CORRECT ANSWER: the model said F, gold is F, and the "
                 "two warm-cache calls said A, so A won 2-1.",
        "status": "WRONG. RETRACTED 2026-07-14 by e2e_reset.",
        "why": "Call 1's ' F' was not the model's belief -- it was a cache artifact. With a TRUE reset the "
               "model answers ' A' on all three calls (gold F) and is simply wrong. There was no correct "
               "answer to destroy. We mistook numerical noise for the model's real answer, which is the "
               "same error, in miniature, that this whole paper is about.",
        "what_survives": "The three calls DO diverge from a byte-identical prompt (461/461 -> identical "
                         "only after a true reset), and calls 2-3 always agree (0/1844). The MECHANISM in "
                         "N5 is confirmed. What does not survive is the claim that the vote destroyed a "
                         "correct answer, and the claim that dropping the vote buys +0.22pp.",
    }

    # ---- the counterfactual: accuracy manufactured by the silent guess ----
    print("\n" + "=" * 88)
    print("HOW MUCH ACCURACY WAS THE SILENT `return \"A\"` MANUFACTURING?\n")
    for t in CELLS:
        fbq = {c["qid"] for c in trace(t) if c["branch"] == "FALLBACK_A"}
        pq = P[t]
        decided = [q for q in fbq if q in pq and pq[q]["pred_answer"] == "A"]
        lucky = [q for q in decided if pq[q]["true_answer"] == "A"]
        n = len(pq)
        right = sum(1 for q in pq if pq[q]["pred_answer"] == pq[q]["true_answer"])
        rep.setdefault("fallback_counterfactual", {})[t] = {
            "questions_decided_by_a_fallback": len(decided),
            "of_those_right_by_luck": len(lucky),
            "reported_answer_accuracy": right / n,
            "if_fallbacks_abstained": (right - len(lucky)) / n,
            "manufactured_pp": len(lucky) / n * 100,
        }
        f = rep["fallback_counterfactual"][t]
        print(f"  {t:<10} decided by a blind 'A': {f['questions_decided_by_a_fallback']:>3}   "
              f"right by luck: {f['of_those_right_by_luck']:>2}   "
              f"reported {f['reported_answer_accuracy']:.4f} -> honest "
              f"{f['if_fallbacks_abstained']:.4f}  ({-f['manufactured_pp']:+.2f}pp)")

    # ---- headline ----
    print("\n" + "=" * 88)
    print("RESULTS (all deterministic; sigma = 0, so every delta is real)\n")
    print(f"  {'cell':<11}{'answer':>11}{'doc':>10}{'page':>10}{'composite':>12}   note")
    notes = {SHIPPED: "shipped", BUDGET: "budget 3->8 (the hypothesis)  NO-OP",
             FIX: "stop-on-newline removed  THE FIX", FIX64: "budget 8->64 on top of the fix  NO-OP"}
    for t in CELLS:
        print(f"  {t:<11}{S[t]['answer_accuracy']:>11.6f}{S[t]['doc_accuracy']:>10.6f}"
              f"{S[t]['page_proximity']:>10.6f}{S[t]['composite_score']:>12.6f}   {notes[t]}")
    print()
    for t in (BUDGET, FIX, FIX64):
        n01 = sum(1 for q in P[t] if P[t][q]["pred_answer"] == P[t][q]["true_answer"]
                  and P[SHIPPED][q]["pred_answer"] != P[SHIPPED][q]["true_answer"])
        n10 = sum(1 for q in P[t] if P[SHIPPED][q]["pred_answer"] == P[SHIPPED][q]["true_answer"]
                  and P[t][q]["pred_answer"] != P[t][q]["true_answer"])
        p = mcnemar(n01, n10)
        rep.setdefault("vs_shipped", {})[t] = {
            "delta_answer": S[t]["answer_accuracy"] - S[SHIPPED]["answer_accuracy"],
            "delta_composite": S[t]["composite_score"] - S[SHIPPED]["composite_score"],
            "mcnemar": {"wins": n01, "losses": n10, "exact_p": p},
        }
        v = rep["vs_shipped"][t]
        print(f"  {t:<10} vs {SHIPPED}:  answer {v['delta_answer']:+.4f}  "
              f"composite {v['delta_composite']:+.4f}   McNemar {n01}W/{n10}L  p={p:.4f}")
    print(f"\n  McNemar's floor: with k discordant pairs the smallest attainable p is 2/2^k. At k=2 that is")
    print(f"  0.5000 -- the test CANNOT reach significance here no matter which way the pairs fall. The")
    print(f"  claim is about the MECHANISM (36 empty strings -> 0), not about a significant score gain.")

    # ---- CHECK 10: the pre-registered cache test (only once e2e_reset exists) ----
    if (B / RESET / "e2e_result.json").exists():
        print("\n" + "=" * 88)
        print("CHECK 10  e2e_reset vs docs/PREREG_e2e_reset.md (predictions committed BEFORE the run)\n")
        rs, rt, rp = summary(RESET), stats(RESET), perq(RESET)
        byq = {}
        for c in trace(RESET):
            byq.setdefault(c["qid"], []).append(c)
        ident = [q for q, cs in byq.items() if cs[0]["text"] == cs[1]["text"] == cs[2]["text"]]
        disagree = sorted(set(byq) - set(ident))

        p1 = len(ident) == len(byq)
        p2 = all(q not in disagree for q in PREREG_DISAGREE_QIDS)
        got408 = collections.Counter([c["letter"] for c in byq[PREREG_P3_QID]]).most_common(1)[0][0] \
            if PREREG_P3_QID in byq else None
        p3 = got408 == PREREG_P3_LETTER
        p4 = abs(rs["answer_accuracy"] - PREREG_P4_ANSWER_ACC) < 1e-9
        p5 = rt["fallback_A"] == PREREG_P5_FALLBACK_A
        surgical = abs(rs["doc_accuracy"] - SHIPPED_DOC_ACC) < 1e-12

        print(f"  P1 HARD  three identical greedy generations on every question")
        print(f"           {len(ident)}/{len(byq)} identical"
              f"{'' if p1 else '   still disagreeing: ' + str(disagree[:8])}   -> {'PASS' if p1 else 'FAIL'}")
        print(f"  P2 HARD  the four known disagreers ({', '.join(PREREG_DISAGREE_QIDS)}) collapse"
              f"   -> {'PASS' if p2 else 'FAIL'}")
        print(f"  P3 MED   qid {PREREG_P3_QID} resolves to {PREREG_P3_LETTER} (gold {PREREG_P3_LETTER}); "
              f"got {got408}   -> {'PASS' if p3 else 'MISS'}")
        print(f"  P4 SOFT  answer_accuracy == {PREREG_P4_ANSWER_ACC:.6f}; "
              f"got {rs['answer_accuracy']:.6f}   -> {'PASS' if p4 else 'MISS (does NOT falsify; a true reset moves call 1 too)'}")
        print(f"  P5 SOFT  FALLBACK_A == {PREREG_P5_FALLBACK_A}; got {rt['fallback_A']}"
              f"   -> {'PASS' if p5 else 'MISS'}")
        print(f"  SURGICAL doc_accuracy unchanged ({SHIPPED_DOC_ACC:.6f}); got {rs['doc_accuracy']:.6f}"
              f"   -> {'PASS' if surgical else 'FAIL -- the reset touched retrieval, which it cannot'}")

        verdict = ("CONFIRMED: a true reset makes the three identical greedy calls return identical text. "
                   "The prefix-cache hit at llama.py:882 was the cause."
                   if (p1 and p2) else
                   "RETRACTED: the calls still disagree after a true reset, so the KV-cache account "
                   "is WRONG and must be rewritten or withdrawn.")
        print(f"\n  VERDICT: {verdict}")
        rep["check10_prereg_reset"] = {
            "hard_P1_all_identical": {"identical": len(ident), "n": len(byq),
                                      "still_disagreeing": disagree, "pass": p1},
            "hard_P2_known_disagreers_collapse": {"qids": PREREG_DISAGREE_QIDS, "pass": p2},
            "med_P3_qid408": {"expected": PREREG_P3_LETTER, "got": got408, "pass": p3},
            "soft_P4_answer_accuracy": {"expected": PREREG_P4_ANSWER_ACC,
                                        "got": rs["answer_accuracy"], "pass": p4,
                                        "note": "a MISS does not falsify: a true reset also re-evaluates "
                                                "call 1, whose answers may themselves move"},
            "soft_P5_fallback_A": {"expected": PREREG_P5_FALLBACK_A, "got": rt["fallback_A"], "pass": p5},
            "surgical_doc_accuracy_unchanged": {"expected": SHIPPED_DOC_ACC,
                                                "got": rs["doc_accuracy"], "pass": surgical},
            "metrics": {m: rs[m] for m in METRICS},
            "vs_fix": {m: rs[m] - S[FIX][m] for m in METRICS},
            "hypothesis_confirmed": bool(p1 and p2),
            "verdict": verdict,
        }
        print(f"\n  {'cell':<11}{'answer':>11}{'doc':>10}{'page':>10}{'composite':>12}")
        for t in (SHIPPED, FIX):
            print(f"  {t:<11}{S[t]['answer_accuracy']:>11.6f}{S[t]['doc_accuracy']:>10.6f}"
                  f"{S[t]['page_proximity']:>10.6f}{S[t]['composite_score']:>12.6f}")
        print(f"  {RESET:<11}{rs['answer_accuracy']:>11.6f}{rs['doc_accuracy']:>10.6f}"
              f"{rs['page_proximity']:>10.6f}{rs['composite_score']:>12.6f}")

    # ---- THE LADDER: what each bug was worth, and what the clean system actually scores ----
    if (B / RESET / "e2e_result.json").exists():
        print("\n" + "=" * 88)
        print("THE LADDER -- every rung removes a bug, and the score goes DOWN when we finish\n")
        gold_ = {q: r["true_answer"] for q, r in P[SHIPPED].items()}
        rungs = [(SHIPPED, "shipped: BOTH bugs"),
                 (FIX, "stop fixed; the cache bug REMAINS"),
                 (RESET, "both fixed -- the CLEAN system")]
        print(f"  {'cell':<11}{'correct':>9}{'answer':>11}{'composite':>12}{'blindA':>8}{'lucky':>7}"
              f"{'honest':>10}   note")
        ladder = {}
        for t, note in rungs:
            s_, pq_ = summary(t), perq(t)
            fbq = {c["qid"] for c in trace(t) if c["branch"] == "FALLBACK_A"}
            blind = [q for q in fbq if pq_[q]["pred_answer"] == "A"]
            lucky = [q for q in blind if gold_[q] == "A"]
            n_ = len(pq_)
            right = sum(1 for q in pq_ if pq_[q]["pred_answer"] == gold_[q])
            ladder[t] = {"correct": right, "n": n_, "answer_accuracy": s_["answer_accuracy"],
                         "composite": s_["composite_score"], "decided_by_blind_A": len(blind),
                         "right_by_luck": len(lucky),
                         "honest_if_abstained": (right - len(lucky)) / n_, "note": note}
            print(f"  {t:<11}{right:>6}/{n_}{s_['answer_accuracy']:>11.6f}{s_['composite_score']:>12.6f}"
                  f"{len(blind):>8}{len(lucky):>7}{(right - len(lucky)) / n_:>10.6f}   {note}")
        d_fix = ladder[RESET]["correct"] - ladder[FIX]["correct"]
        ladder["interpretation"] = (
            f"Fixing the cache bug COSTS us {-d_fix} question(s): {ladder[FIX]['correct']} -> "
            f"{ladder[RESET]['correct']} of {ladder[RESET]['n']}. The numerical nondeterminism was "
            f"NET-FAVOURABLE to our score. Both bugs were manufacturing accuracy, and the fully corrected "
            f"system scores LOWER than the half-corrected one. This is the honest result and it is the "
            f"one we report. The clean system still beats the shipped one "
            f"({ladder[RESET]['correct']} vs {ladder[SHIPPED]['correct']} correct), but it beats it by "
            f"less than the half-fix appeared to, and for a reason that has nothing to do with the model "
            f"getting better: we stopped guessing, and we stopped letting arithmetic noise vote.")
        ladder["vote_is_now_a_provable_no_op"] = (
            "In e2e_reset all three calls return IDENTICAL text on 461/461 questions. At vote_temp=0 the "
            "3-pass vote is therefore EXACTLY a no-op -- not 'unhelpful', but mathematically incapable of "
            "changing any answer -- while costing 3x the inference. That is a logical identity given P1, "
            "and it REPLACES the earlier (wrong) claim that dropping the vote buys +0.22pp.")
        print(f"\n  {ladder['interpretation']}")
        print(f"\n  {ladder['vote_is_now_a_provable_no_op']}")
        rep["the_ladder"] = ladder

    rep["cells"] = {t: {m: S[t][m] for m in METRICS} for t in CELLS}
    if (B / RESET / "e2e_result.json").exists():
        rep["cells"][RESET] = {m: summary(RESET)[m] for m in METRICS}
    rep["sigma_note"] = ("vote_temp=0 => sigma exactly 0 (B5 GATE 2: 0/461 churn across seeds). "
                         "Deltas are real, not noise.")
    # Hash-pin every file this report was computed from. Re-run any cell and this artifact goes
    # STALE_INPUT instead of quietly continuing to look authoritative.
    cells_present = list(CELLS) + ([RESET] if (B / RESET / "e2e_result.json").exists() else [])
    inputs = [str(B / t / f) for t in cells_present for f in ("summary.json", "answer_trace.json",
                                                              "per_question.csv")]
    inputs += [str(B / REGRESS / f) for f in ("summary.json", "per_question.csv")]
    json.dump(stamp(rep, "e2e_analyze.py", inputs=inputs), open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}")
    if not ok4:
        sys.exit(1)


if __name__ == "__main__":
    main()

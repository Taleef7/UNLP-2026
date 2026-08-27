#!/usr/bin/env python3
"""Re-derive every quantitative claim in the paper from the raw artifacts, and fail on disagreement.

The rule this enforces is "no number in the manuscript is typed by hand". Each check pairs a literal
string that must appear verbatim in `paper/harness_answers_a.tex` with the figure that string asserts,
recomputed here from `outputs/*.json`. If the paper and the artifacts ever disagree -- because a cell
was re-run, or a number was retyped from memory -- this exits non-zero and names the claim that broke.

    python3 scripts/check_paper_numbers.py

It matches literals rather than scraping numbers out of the prose. An earlier version tried to recover
each figure from its surrounding text and reported mismatches that were all its own doing; a checker
that fails for its own reasons teaches you to ignore it, which is worse than not having one.
"""
import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(REPO, "paper/harness_answers_a.tex")


def load(p):
    return json.load(open(os.path.join(REPO, p)))


def main():
    if not os.path.exists(TEX):
        sys.exit(f"manuscript not found: {TEX}")
    tex = open(TEX, encoding="utf-8").read()
    checks, fails = [], []

    def claim(label, literal, num, expected, tol=5e-4):
        present = literal in tex
        got = float(str(num).replace("−", "-").replace(",", ""))
        ok_val = abs(got - float(expected)) <= tol
        checks.append((label, literal, got, float(expected), present, present and ok_val))
        if not (present and ok_val):
            fails.append((label, literal, got, float(expected), present, ok_val))

    def exact(label, literal, got, expected):
        present = literal in tex
        checks.append((label, literal, float(got), float(expected), present, present and got == expected))
        if not (present and got == expected):
            fails.append((label, literal, float(got), float(expected), present, got == expected))

    # ---- fusion: a default weight, read as "Cyrillic breaks lexical fusion" ----
    fw = load("outputs/miracl2/fusion_weight.json")
    dflt, lolo = fw["default_no_selection"], fw["lolo_selected"]
    swing = abs(dflt["mean_gain"]) + lolo["mean_gain_heldout"]
    claim("swing", "14.6-point swing", 14.6, swing * 100, tol=0.06)
    claim("MIRACL loss", "0.1242 nDCG@10", "0.1242", abs(dflt["mean_gain"]))
    claim("MIRACL n_help", r"\textbf{only 2}", 2, dflt["n_help"], tol=0.5)
    claim("LOLO n_help", r"\textbf{helps in 16}", 16, lolo["n_help"], tol=0.5)
    claim("n languages", "18 languages", 18, fw["n_languages"], tol=0.5)
    tuned_loss = abs(load("outputs/c1_nested_cv.json")
                     ["supplement_fixed_dw_tuned_bm25"]["raw"]["dw0.5"]["delta"]) * 100
    claim("tuned-BM25", "5.42 Doc@1", "5.42", tuned_loss, tol=0.01)

    # ---- visual retrieval: base-model coverage, read as a script barrier ----
    cbz = load("outputs/colsmol2/c2_cluster_bootstrap.json")
    claim("ColSmol page effect", "+0.165", "0.165", cbz["colSmol-500M"]["effects"]["page"]["delta"])
    claim("c2 n queries", "200 queries, 41 documents", 200, cbz["colSmol-500M"]["n_queries"], tol=0.5)
    claim("c2 n clusters", "200 queries, 41 documents", 41, cbz["colSmol-500M"]["n_clusters"], tol=0.5)

    # ---- CoT / quantization: the answer parser, read as a quantization effect ----
    c3 = load("outputs/c3_quant_cot_v2.json")["results"]
    cot, dirr, inter = c3["int8_cot"], c3["int8_direct"], c3["interaction"]
    cot_acc, cot_first = cot["answer_acc"], cot["answer_acc_firstletter_rule"]
    dir_acc = dirr["answer_acc"]
    claim("c3 first-vs-last", "6.67 points", "6.67", 100 * (cot_acc - cot_first), tol=0.01)
    claim("c3 deficit", "5.33-point deficit", "5.33", 100 * (dir_acc - cot_acc), tol=0.01)
    claim("c3 direct CI", r"$\pm$2.7pp", "2.7", 100 * inter["quant_cost_direct_ci"][1], tol=0.05)
    claim("c3 direct 0.0", r"\textbf{0.0} points", "0.0", 100 * inter["quant_cost_direct"], tol=0.01)

    # ---- the two harness bugs in the shipped system ----
    e = load("outputs/e2e_stop_bug.json")
    lad = e["the_ladder"]
    c4_ = e["check4_fix_reaches_mechanism"]["cells"]
    c8_ = e["check8_warm_cache_signature"]
    fb = e["fallback_counterfactual"]["e2e_b3"]
    claim("empties", r"\textbf{36 of 1,383}", 36, c4_["e2e_b3"]["empty"], tol=0.5)
    claim("n calls", r"\textbf{36 of 1,383}", 1383, c4_["e2e_b3"]["n_calls"], tol=0.5)
    claim("blind-A", r"\textbf{13 questions}", 13, fb["questions_decided_by_a_fallback"], tol=0.5)
    claim("right by luck", r"\textbf{2} were correct by chance", 2, lad["e2e_b3"]["right_by_luck"], tol=0.5)
    claim("manufactured", r"\textbf{0.43} points", "0.43",
          100 * (lad["e2e_b3"]["answer_accuracy"] - lad["e2e_b3"]["honest_if_abstained"]), tol=0.01)
    tri = sum(v["n_questions"] for v in c8_["per_cell"].values())
    claim("vote-triples", r"\textbf{0 of 1,844}", 1844, tri, tol=0.5)
    claim("reset identical", r"\textbf{461/461}", 461,
          e["check10_prereg_reset"]["hard_P1_all_identical"]["identical"], tol=0.5)

    # ---- the two falsifiable predictions, as counts that must be exactly zero ----
    exact("calls 2-3 disagree", r"\textbf{0 of 1,844}",
          sum(v["c2_vs_c3_disagree"] for v in c8_["per_cell"].values()), 0)
    exact("homoglyph hits", r"\textbf{0 times in 5,532 calls}",
          sum(e["check7_homoglyph_hypothesis"]["prefix_cyrillic_hits"].values()), 0)
    claim("homoglyph n calls", r"\textbf{0 times in 5,532 calls}", 5532,
          e["check7_homoglyph_hypothesis"]["total_calls_across_cells"], tol=0.5)

    # ---- report ----
    print(f"{'':2}{'check':<34}{'paper says':>13}{'raw JSON':>13}   ok")
    print("-" * 74)
    for label, lit, got, exp, present, ok in checks:
        print(f"  {label:<34}{got:>13.4f}{exp:>13.4f}   "
              f"{'OK' if ok else ('LITERAL NOT IN PAPER' if not present else 'MISMATCH')}")
    print("-" * 74)
    if fails:
        print(f"\n{len(fails)} MISMATCH(ES) — the paper disagrees with the artifacts:\n")
        for label, lit, got, exp, present, okv in fails:
            why = (f"the literal {lit!r} does not appear in the paper" if not present
                   else f"paper asserts {got}, raw JSON says {exp}")
            print(f"  ✗ {label}: {why}")
        sys.exit(1)
    print(f"\n{len(checks)}/{len(checks)} numbers in the paper verified against raw JSON.")


if __name__ == "__main__":
    main()

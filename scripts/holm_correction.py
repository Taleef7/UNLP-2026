#!/usr/bin/env python3
"""Multiplicity control across the paper's contrast families (methodology rigor gate).

REWRITTEN 2026-07-13 to close the audit notes. The previous version claimed families were "declared
before looking at the corrected results" and then, in the same file:

  * sized family F3 as `len(glob(outputs/rusbeir/pq_ndcg_*.json))` -- a family whose size is
    whatever happens to be on disk. The docstring declared 25; the code produced 30 at audit time;
    it produces 48 TODAY, because a later run added two more variants to those files. The family
    silently grew by 92% without anyone editing a line. That is the whole objection, demonstrated.
  * split UNLP into one family per METRIC (Doc@1, Pg@1), halving m and doubling the effective alpha,
    though both metrics test the same variants on the same 461 queries and we report both.
  * excused C2 and C3 from correction BY CITING THEIR OBSERVED EFFECT SIZES ("effects ~6 SE";
    "C3's headline is a NULL") -- i.e. it inspected the results to decide what to correct for.

The fix is not a better guess at m. It is a MEMBERSHIP RULE that is stated in advance and leaves no
freedom at analysis time:

    ONE FAMILY PER (collection, baseline). It contains EVERY variant that was run against that
    baseline, crossed with EVERY metric we report for that collection. No variant is dropped for
    being uninteresting; no metric is split off to shrink m; nothing is excused for its effect size.

Everything below follows mechanically from that rule. The family members are enumerated as literal
constants -- not globbed, not counted at runtime -- and `load_family` RAISES if what is on disk does
not match the declaration exactly, in either direction. A family that gains a variant is a family
that must be re-declared, deliberately, in this file, by a human.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import REPO
from provenance import stamp, check_fresh

NPERM = 20000
SEED = 42

# --------------------------------------------------------------------------------------------------
# THE DECLARATION. Edit this block, and only this block, to change a family.
# --------------------------------------------------------------------------------------------------

UNLP_LEXICAL = [
    "bm25_lemma", "bm25_raw",
    "hybrid_lemma_dw0.30_sw0.70", "hybrid_lemma_dw0.50_sw0.50",
    "hybrid_lemma_dw0.70_sw0.30", "hybrid_lemma_dw0.90_sw0.10",
    "hybrid_raw_dw0.30_sw0.70", "hybrid_raw_dw0.50_sw0.50",
    "hybrid_raw_dw0.70_sw0.30", "hybrid_raw_dw0.90_sw0.10",
]
# BGE-M3's own sparse/ColBERT heads, from bge_native_hybrid.py. Same 461 queries, same dense
# baseline, same question ("does a non-dense signal added to dense help on UNLP?"), so by the rule
# above they are in the SAME family -- they are not a separate experiment. This is where the paper's
# one positive retrieval result lives (B3); it has never before been corrected for anything.
UNLP_NATIVE = [
    "bge_sparse_only",
    "bge_dense+sparse_dw0.5", "bge_dense+sparse_dw0.7", "bge_dense+sparse_dw0.9",
    "bge_dense+sparse+colbert",
]
UNLP_METRICS = [(0, "Doc@1"), (1, "Pg@1")]   # both reported => both in the family

RUSBEIR_DATASETS = ["arguana", "mmarco", "nfcorpus", "scifact", "tydiqa", "xquad"]
RUSBEIR_VARIANTS = [
    "bm25_lemma", "bm25_raw",
    "hybrid_lemma_dw0.5", "hybrid_lemma_dw0.7",
    "hybrid_raw_dw0.5", "hybrid_raw_dw0.7",
    "native_dense+sparse", "native_sparse",
]
# mmarco is INCLUDED. It was silently dropped from A1's headline and denominators (item C7) while
# still sitting in metrics.json and in the old m=30 count -- present for the multiplicity penalty,
# absent from the win column. Under the rule above there is no mechanism to drop it.

BASELINE = "dense"

# Declared family sizes, computed from the declaration itself so they cannot drift from it.
FAM_UNLP = len(UNLP_LEXICAL + UNLP_NATIVE) * len(UNLP_METRICS)   # 15 x 2 = 30
FAM_RUSBEIR = len(RUSBEIR_DATASETS) * len(RUSBEIR_VARIANTS)      # 6 x 8  = 48


def perm_pvalue(diffs, nperm=NPERM, seed=SEED):
    """Two-sided paired randomization (sign-flip) test on the mean of paired differences.

    H0: the paired difference distribution is symmetric about 0, i.e. the variant label carries no
    information. Under H0 each item's sign is exchangeable, so we flip signs at random.
    """
    d = np.asarray(diffs, dtype="float64")
    d = d[~np.isnan(d)]
    n = len(d)
    obs = abs(d.mean())
    if n == 0 or obs == 0:
        return 1.0, 0.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(nperm, n))
    null = np.abs((signs * d).mean(axis=1))
    # +1 correction: the observed assignment is itself one of the permutations
    p = (1.0 + np.sum(null >= obs - 1e-12)) / (nperm + 1.0)
    return float(p), float(d.mean())


def holm(tests):
    """tests: list of (label, p, effect). Returns rows with Holm-adjusted p and reject@.05."""
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * tests[i][1])
        running = max(running, a)          # enforce monotonicity
        adj[i] = running
    return [(tests[i][0], tests[i][2], tests[i][1], adj[i], adj[i] < 0.05) for i in range(m)]


def check_membership(family, declared, found):
    """A family is a DECLARATION, not a directory listing. Any drift is an error, both ways:
    an unexpected variant means the family grew without being re-declared (the F3 bug); a missing
    variant means we are about to correct for fewer tests than we actually ran (anti-conservative).
    Either way the corrected p-values would be wrong, so we refuse to produce them."""
    d, f = set(declared), set(found)
    if d != f:
        raise SystemExit(
            f"\nFAMILY MISMATCH in {family}: the declaration in holm_correction.py does not match "
            f"what is on disk.\n"
            f"  declared but absent : {sorted(d - f) or '(none)'}\n"
            f"  present but undeclared: {sorted(f - d) or '(none)'}\n"
            f"Family size drives every corrected p-value. Re-declare the family DELIBERATELY in the "
            f"DECLARATION block, or restore the missing artifact. Do not let it drift.\n")


def report(name, tests, declared_m):
    if len(tests) != declared_m:
        raise SystemExit(f"{name}: built {len(tests)} contrasts but declared m={declared_m}")
    print(f"\n=== {name}  (declared family size m={declared_m}) ===")
    rows = holm(tests)
    rows.sort(key=lambda r: r[2])
    print(f"{'contrast':<36} {'effect':>9} {'p_raw':>10} {'p_Holm':>10}  sig")
    n_raw = n_holm = 0
    for label, eff, p, pa, rej in rows:
        n_raw += p < 0.05
        n_holm += rej
        print(f"{label:<36} {eff:+9.4f} {p:10.5f} {pa:10.5f}   {'*' if rej else ' '}")
    print(f"-- significant: {n_raw}/{len(tests)} uncorrected -> {n_holm}/{len(tests)} after Holm")
    return rows


def main():
    out, meta = {}, {}
    unevaluable = []

    # ---------- Family 1: UNLP dev retrieval, all variants x both reported metrics ----------
    pqh = os.path.join(REPO, "outputs/retrieval_eval/per_query_hits.json")
    nat = os.path.join(REPO, "outputs/retrieval_eval/c1d_per_query_hits.json")
    missing = [p for p in (pqh, nat) if not os.path.exists(p)]
    if missing:
        # We do NOT quietly fall back to the lexical-only half of the family: shrinking m until the
        # survivors survive is exactly the move this rewrite exists to prevent. The family is simply
        # not evaluable, and we say so instead of reporting a smaller one.
        unevaluable.append(
            "FAM1_UNLP_dev_vs_dense -- missing per-query artifact(s): "
            + ", ".join(os.path.relpath(p, REPO) for p in missing)
            + "\n    The native-hybrid arms (bge_dense+sparse*, +colbert) are DECLARED members, so the "
              "family cannot be corrected without them.\n"
              "    Regenerate: python3 scripts/bge_native_hybrid.py --out-dir outputs/retrieval_eval\n"
              "    (That script computed these per-query hits and discarded them -- item B3.)")
        hits = None
    else:
        hits = json.load(open(pqh))
    if hits is not None:
        nhits = json.load(open(nat))
        check_membership("UNLP/lexical", UNLP_LEXICAL, [t for t in hits if t != BASELINE])
        check_membership("UNLP/native", UNLP_NATIVE,
                         [t for t in nhits if t != "bge_dense_only"])

        qids = sorted(hits[BASELINE])
        tests = []
        for mi, mn in UNLP_METRICS:
            for tag in UNLP_LEXICAL:
                d = [hits[tag][q][mi] - hits[BASELINE][q][mi] for q in qids]
                p, eff = perm_pvalue(d)
                tests.append((f"{mn}/{tag}", p, eff))
            for tag in UNLP_NATIVE:
                d = [nhits[tag][q][mi] - nhits["bge_dense_only"][q][mi] for q in qids]
                p, eff = perm_pvalue(d)
                tests.append((f"{mn}/{tag}", p, eff))
        out["FAM1_UNLP_dev_vs_dense"] = report(
            f"FAM1 — UNLP dev, all variants x {{Doc@1, Pg@1}} vs dense (N={len(qids)})",
            tests, FAM_UNLP)
        meta["FAM1_UNLP_dev_vs_dense"] = {"m": FAM_UNLP, "n_queries": len(qids),
                                          "variants": UNLP_LEXICAL + UNLP_NATIVE,
                                          "metrics": [m for _, m in UNLP_METRICS]}

    # ---------- Family 2: RusBEIR, all datasets x all variants, nDCG@10 ----------
    rd = os.path.join(REPO, "outputs/rusbeir")
    tests = []
    for ds in RUSBEIR_DATASETS:
        f = os.path.join(rd, f"pq_ndcg_{ds}.json")
        if not os.path.exists(f):
            raise SystemExit(f"RusBEIR family declares {ds} but {f} is missing.")
        pq = json.load(open(f))
        check_membership(f"RusBEIR/{ds}", RUSBEIR_VARIANTS, [v for v in pq if v != BASELINE])
        base = pq[BASELINE]
        qs = sorted(base)
        for v in RUSBEIR_VARIANTS:
            d = [pq[v][q] - base[q] for q in qs]
            p, eff = perm_pvalue(d)
            tests.append((f"{ds}/{v}", p, eff))
    out["FAM2_RusBEIR_nDCG10_vs_dense"] = report(
        "FAM2 — RusBEIR, 6 datasets x 8 variants, nDCG@10 vs dense", tests, FAM_RUSBEIR)
    meta["FAM2_RusBEIR_nDCG10_vs_dense"] = {"m": FAM_RUSBEIR, "datasets": RUSBEIR_DATASETS,
                                            "variants": RUSBEIR_VARIANTS, "metrics": ["nDCG@10"]}

    # ---------- A2 / MIRACL ----------
    # MIRACL's 234 contrasts are corrected in scripts/gen_a2_doc.py, where the family is likewise
    # declared statically (18 langs x 13 non-null variants). It is not duplicated here.

    # A partial correction is a wrong correction: if a declared family could not be evaluated, we
    # print what we have for inspection but refuse to SERIALIZE it, so no downstream document can
    # cite a corrected p-value from an incomplete family.
    if unevaluable:
        print("\n" + "=" * 90)
        print("REFUSING TO WRITE outputs/holm_correction.json -- a declared family is not evaluable:")
        for u in unevaluable:
            print(f"  * {u}")
        print("=" * 90)
        sys.exit(1)

    dst = os.path.join(REPO, "outputs/holm_correction.json")
    # Declare every artifact this correction was computed FROM. This file is the one that was
    # corrupted by the original C10 bug -- it read a per_query_hits.json written before a fix to
    # retrieval_eval.py, so FAM1 spanned two code versions. Content-hashing the inputs means that
    # exact state is now detectable by `python3 scripts/provenance.py` instead of invisible.
    _inputs = [os.path.join(REPO, "outputs/retrieval_eval/per_query_hits.json"),
               os.path.join(REPO, "outputs/retrieval_eval/c1d_per_query_hits.json")] + \
              [os.path.join(REPO, "outputs/rusbeir", f"pq_ndcg_{ds}.json")
               for ds in RUSBEIR_DATASETS]
    json.dump(stamp({"families": meta,
                     "results": {k: [{"contrast": r[0], "effect": r[1], "p_raw": r[2],
                                      "p_holm": r[3], "sig_holm": bool(r[4])} for r in v]
                                 for k, v in out.items()}}, "holm_correction.py", inputs=_inputs),
              open(dst, "w"), indent=2)
    print(f"\n[saved] {dst}")
    print(f"[families] FAM1 m={FAM_UNLP} (was 10+10 split by metric); "
          f"FAM2 m={FAM_RUSBEIR} (was len(glob(...)) = 30 at audit, 48 today)")


if __name__ == "__main__":
    main()

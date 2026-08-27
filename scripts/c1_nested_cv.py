#!/usr/bin/env python3
"""C1 REBUILD — a selection-free estimate, and the lemma/stoplist confound separated.

Two defects in the original C1:

  1. STRAWMAN SPARSE ARM. BM25 was `rank_bm25.BM25Okapi` at library defaults; k1/b were never tuned.
     Every "sparse fusion damages dense" conclusion was confounded with "our sparse arm is
     misconfigured."
  2. SELECTION ON THE REPORTING SET. `hybrid_lemma_dw0.90` (+0.9pp Doc@1, the paper's one positive
     retrieval result) won a fusion-weight sweep that was *evaluated on the same 461 questions it was
     reported on*, and Doc@1 is monotone in dw -- so it is a grid-endpoint artifact. No held-out split
     for fusion-weight selection existed anywhere in the repo.
  3. The "+8.7pp lemma gain" confounded LEMMATIZATION with STOPWORD REMOVAL (the lemma arm applied a
     stoplist; the raw arm applied none).

This script fixes all three:

  * NESTED 5-FOLD CROSS-VALIDATION. For each held-out fold, (k1, b, dw) are selected on the OTHER four
    folds and then applied, untouched, to the held-out fold. Concatenating the held-out predictions
    gives a selection-free estimate at the FULL N=461 -- no leakage, no shrunken eval set.
  * TUNED Okapi BM25 (`bm25_sparse.SparseBM25`).
  * A 2x2 TOKENIZER GRID that separates the two changes:
        raw            (no lemma, no stoplist)
        stop           (no lemma, + stoplist)      -> isolates the STOPLIST effect
        lemma_nostop   (+ lemma,  no stoplist)     -> isolates the LEMMATIZATION effect
        lemma          (+ lemma,  + stoplist)      -> the original arm
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp   # item C10: an artifact must remember the code that wrote it
from retrieval_eval import (load_unlp_corpus, load_unlp_queries, bge_encode, run_dense,
                            doc_order_from_units, paired_bootstrap, tok_raw, tok_stop,
                            tok_lemma_nostop, tok_lemma, REPO)
from bm25_sparse import SparseBM25

TOKS = {"raw": tok_raw, "stop": tok_stop, "lemma_nostop": tok_lemma_nostop, "lemma": tok_lemma}
K1_GRID = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
B_GRID = (0.3, 0.5, 0.75, 0.9, 1.0)
DW_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def minmax(M):
    lo = M.min(axis=1, keepdims=True); hi = M.max(axis=1, keepdims=True)
    return (M - lo) / (hi - lo + 1e-9)


def doc1(scores, queries, unit_docs, idxs):
    """Doc@1 == the document of the highest-scoring PAGE (first-appearance ranking makes the top doc
    exactly the argmax unit's doc), so we can skip building the full ranking."""
    top = np.asarray(unit_docs, dtype=object)[np.argmax(scores, axis=1)]
    gold = np.array([queries[i]["gold_doc"] for i in idxs], dtype=object)
    return float(np.mean(top == gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/c1_nested_cv.json"))
    args = ap.parse_args()

    units, doc_domain, _ = load_unlp_corpus(os.path.join(REPO, "data/extracted_text"))
    queries = load_unlp_queries(os.path.join(REPO, "data/dev_questions.csv"))
    unit_docs = [u[0] for u in units]
    n = len(queries)

    cache = os.path.join(REPO, "outputs/retrieval_eval/unlp_embs.npz")
    z = np.load(cache)
    page_embs, q_embs = z["page"], z["query"]
    _, dense_scores = run_dense(queries, units, page_embs, q_embs)   # (Q, U) matrix
    D = np.asarray(dense_scores, dtype="float32")
    Dn = minmax(D)
    print(f"[data] {len(units)} pages, {n} queries", file=sys.stderr)

    # dense baseline, per query
    _topd = np.asarray(unit_docs, dtype=object)[np.argmax(D, axis=1)]
    base_hits = np.array([1.0 if _topd[i] == queries[i]["gold_doc"] else 0.0 for i in range(n)])
    print(f"[baseline] dense Doc@1 = {base_hits.mean():.4f}", file=sys.stderr)

    rng = np.random.default_rng(42)
    fold_of = rng.permutation(n) % args.folds

    report = {"N": n, "dense_doc1": float(base_hits.mean()), "arms": {}}
    for tname, tfn in TOKS.items():
        t0 = time.time()
        bm = SparseBM25().fit([tfn(u[2]) for u in units])
        qtok = [tfn(q["text"]) for q in queries]

        # cache BM25 score matrices for every (k1,b) once -- the grid is reused by every fold
        S = {}
        for k1 in K1_GRID:
            for b in B_GRID:
                bm.set_params(k1, b)
                S[(k1, b)] = minmax(bm.scores(qtok))

        cv_hits = np.zeros(n)
        chosen = []
        for f in range(args.folds):
            te = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            best, bacc = None, -1.0
            for (k1, b), Sn in S.items():
                for dw in DW_GRID:
                    F = dw * Dn[tr] + (1 - dw) * Sn[tr]
                    acc = doc1(F, queries, unit_docs, tr)
                    if acc > bacc:
                        bacc, best = acc, (k1, b, dw)
            k1, b, dw = best
            chosen.append({"fold": f, "k1": k1, "b": b, "dw": dw, "train_doc1": bacc})
            Fte = dw * Dn[te] + (1 - dw) * S[(k1, b)][te]
            topd = np.asarray(unit_docs, dtype=object)[np.argmax(Fte, axis=1)]
            for r, i in enumerate(te):
                cv_hits[i] = 1.0 if topd[r] == queries[i]["gold_doc"] else 0.0

        m, lo, hi = paired_bootstrap(list(cv_hits), list(base_hits))
        sig = "*" if (lo > 0 or hi < 0) else " "
        report["arms"][tname] = {
            "cv_doc1": float(cv_hits.mean()),
            "delta_vs_dense": m, "ci": [lo, hi], "significant": bool(lo > 0 or hi < 0),
            "selected_per_fold": chosen}
        print(f"{tname:<14} CV Doc@1={cv_hits.mean():.4f}   Δ vs dense = {m:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]{sig}   ({time.time()-t0:.0f}s)", flush=True)
        print(f"               folds chose: " +
              ", ".join(f"k1={c['k1']},b={c['b']},dw={c['dw']}" for c in chosen), flush=True)

    # --- the confound decomposition (all vs the SAME dense baseline, nested-CV numbers) ---
    a = report["arms"]
    print("\n=== separating LEMMATIZATION from STOPWORD REMOVAL (nested-CV Doc@1) ===")
    print(f"  raw                       {a['raw']['cv_doc1']:.4f}")
    print(f"  + stoplist only           {a['stop']['cv_doc1']:.4f}   "
          f"(stoplist effect: {a['stop']['cv_doc1']-a['raw']['cv_doc1']:+.4f})")
    print(f"  + lemmatization only      {a['lemma_nostop']['cv_doc1']:.4f}   "
          f"(lemma effect:    {a['lemma_nostop']['cv_doc1']-a['raw']['cv_doc1']:+.4f})")
    print(f"  + both (the original arm) {a['lemma']['cv_doc1']:.4f}   "
          f"(combined:        {a['lemma']['cv_doc1']-a['raw']['cv_doc1']:+.4f})")
    report["decomposition"] = {
        "stoplist_only": a['stop']['cv_doc1'] - a['raw']['cv_doc1'],
        "lemma_only": a['lemma_nostop']['cv_doc1'] - a['raw']['cv_doc1'],
        "both": a['lemma']['cv_doc1'] - a['raw']['cv_doc1']}

    json.dump(stamp(report, "c1_nested_cv.py", inputs=[cache]), open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


def supplement():
    """Isolate WHICH miscalibration caused the damage: the untuned sparse arm, or the equal weight?

    The nested CV above always SELECTS dw (and picks 0.8-0.95), so it never evaluates the equal-weight
    configuration we actually shipped. This supplement holds dw FIXED at a grid and reports the damage
    with a TUNED BM25 -- separating "our tokenizer was wrong" from "our fusion weight was wrong".
    """
    units, _, _ = load_unlp_corpus(os.path.join(REPO, "data/extracted_text"))
    queries = load_unlp_queries(os.path.join(REPO, "data/dev_questions.csv"))
    unit_docs = np.asarray([u[0] for u in units], dtype=object)
    gold = np.array([q["gold_doc"] for q in queries], dtype=object)
    z = np.load(os.path.join(REPO, "outputs/retrieval_eval/unlp_embs.npz"))
    _, D = run_dense(queries, units, z["page"], z["query"])
    D = np.asarray(D, dtype="float32"); Dn = minmax(D)
    base = (unit_docs[np.argmax(D, axis=1)] == gold).astype(float)
    cv = json.load(open(os.path.join(REPO, "outputs/c1_nested_cv.json")))

    print(f"\n=== SUPPLEMENT: damage at FIXED fusion weight, with a TUNED BM25 (N={len(queries)}) ===")
    print(f"dense Doc@1 = {base.mean():.4f}")
    print(f"{'arm':<14}{'k1,b':<10}" + "".join(f"{'dw='+str(w):>22}" for w in (0.5, 0.7, 0.9)))
    out = {}
    for tname, tfn in TOKS.items():
        # use the modal (k1,b) the CV folds selected for this arm
        sel = cv["arms"][tname]["selected_per_fold"]
        k1 = max(set(c["k1"] for c in sel), key=[c["k1"] for c in sel].count)
        b = max(set(c["b"] for c in sel), key=[c["b"] for c in sel].count)
        bm = SparseBM25(k1, b).fit([tfn(u[2]) for u in units])
        Sn = minmax(bm.scores([tfn(q["text"]) for q in queries]))
        cells, row = [], {}
        for w in (0.5, 0.7, 0.9):
            F = w * Dn + (1 - w) * Sn
            hits = (unit_docs[np.argmax(F, axis=1)] == gold).astype(float)
            m, lo, hi = paired_bootstrap(list(hits), list(base))
            sig = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{m:+.4f}[{lo:+.3f},{hi:+.3f}]{sig}")
            row[f"dw{w}"] = {"delta": m, "ci": [lo, hi], "sig": bool(lo > 0 or hi < 0)}
        out[tname] = {"k1": k1, "b": b, **row}
        print(f"{tname:<14}{f'{k1},{b}':<10}" + "".join(f"{c:>22}" for c in cells))
    print("* = 95% CI excludes 0")
    cv["supplement_fixed_dw_tuned_bm25"] = out
    json.dump(stamp(cv, "c1_nested_cv.py",
                    inputs=[os.path.join(REPO, "outputs/retrieval_eval/unlp_embs.npz")]),
              open(os.path.join(REPO, "outputs/c1_nested_cv.json"), "w"), indent=2)


if __name__ == "__main__":
    if os.environ.get("SUPP"):
        supplement()
    else:
        main()

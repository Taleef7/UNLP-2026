#!/usr/bin/env python3
"""C1d — BGE-M3 native unified hybrid (dense + sparse/lexical + ColBERT), all from ONE forward pass.

Answers the reviewer's "why did you discard 2/3 of BGE-M3?" by comparing dense-only against
BGE-M3's own recommended dense+sparse+ColBERT fusion. Sparse scored via sparse vocab matmul;
ColBERT applied as a rerank over top-K dense candidates (its intended efficient use).

Reuses corpus/query loaders + metrics from retrieval_eval.py.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import write_sidecar
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import (load_unlp_corpus, load_unlp_queries, doc_order_from_units,
                            evaluate, fmt, REPO, per_query_hits, paired_bootstrap)


def encode_all(model, texts, batch, max_length):
    return model.encode(texts, batch_size=batch, max_length=max_length,
                        return_dense=True, return_sparse=True, return_colbert_vecs=True)


def lex_to_csr(lexical_weights, vocab=250002):
    """List of {token_id(str): weight} -> CSR (N x vocab)."""
    rows, cols, data = [], [], []
    for i, lw in enumerate(lexical_weights):
        for tid, w in lw.items():
            rows.append(i); cols.append(int(tid)); data.append(float(w))
    return sp.csr_matrix((data, (rows, cols)), shape=(len(lexical_weights), vocab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", default=os.path.join(REPO, "data/extracted_text"))
    ap.add_argument("--queries", default=os.path.join(REPO, "data/dev_questions.csv"))
    ap.add_argument("--rerank-k", type=int, default=20, help="ColBERT rerank depth over dense top-k")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs/retrieval_eval"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    units, doc_domain, _ = load_unlp_corpus(args.text_dir)
    queries = load_unlp_queries(args.queries)
    unit_docs = [u[0] for u in units]
    unit_dp = [(u[0], u[1]) for u in units]
    print(f"[data] {len(units)} units, {len(queries)} queries", file=sys.stderr)

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("[bge-m3] encoding pages...", file=sys.stderr)
    P = encode_all(model, [u[2] for u in units], batch=16, max_length=1536)
    print("[bge-m3] encoding queries...", file=sys.stderr)
    Q = encode_all(model, [q["text"] for q in queries], batch=32, max_length=256)

    # --- dense (normalized vecs -> cosine via matmul) ---
    dense = Q["dense_vecs"] @ P["dense_vecs"].T                     # (nQ, nU)
    # --- sparse / lexical via vocab-space matmul ---
    Pl = lex_to_csr(P["lexical_weights"]); Ql = lex_to_csr(Q["lexical_weights"])
    sparse = (Ql @ Pl.T).toarray().astype("float32")               # (nQ, nU)
    print(f"[scores] dense {dense.shape} sparse {sparse.shape}", file=sys.stderr)

    def per_query_ranked(scores):
        out = {}
        for i, q in enumerate(queries):
            do, order = doc_order_from_units(scores[i], unit_docs)
            out[q["qid"]] = (do, [unit_dp[j] for j in order])
        return out

    def minmax(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-9)

    results, top1, hits = {}, {}, {}

    def record(tag, ranked):
        m, dm, t1 = evaluate(queries, ranked, doc_domain)
        results[tag] = {"overall": m, "per_domain": dm}
        top1[tag] = t1
        hits[tag] = per_query_hits(queries, ranked)
        print(f"{tag:<34} {fmt(m)}")

    record("bge_dense_only", per_query_ranked(dense))
    record("bge_sparse_only", per_query_ranked(sparse))

    # dense+sparse weighted (per-query min-max normalized), sweep
    for dw in (0.5, 0.7, 0.9):
        sw = 1 - dw
        combo = np.vstack([dw * minmax(dense[i]) + sw * minmax(sparse[i])
                           for i in range(len(queries))])
        record(f"bge_dense+sparse_dw{dw:.1f}", per_query_ranked(combo))

    # + ColBERT rerank over top-K dense candidates for the best dense+sparse setting
    dw, sw, cw = 0.4, 0.2, 0.4
    base = np.vstack([dw * minmax(dense[i]) + sw * minmax(sparse[i]) for i in range(len(queries))])
    colbert_scores = np.zeros_like(base)
    for i in range(len(queries)):
        cand = np.argsort(-dense[i])[:args.rerank_k]
        qv = Q["colbert_vecs"][i]
        cs = np.array([model.colbert_score(qv, P["colbert_vecs"][j]) for j in cand], dtype="float32")
        cs = minmax(cs)
        for r, j in enumerate(cand):
            colbert_scores[i, j] = cs[r]
    full = base + cw * colbert_scores
    record("bge_dense+sparse+colbert", per_query_ranked(full))

    # --- paired bootstrap vs dense-only baseline (methodology gate) ---
    base = "bge_dense_only"
    qids = sorted(hits[base])
    print("\nPaired bootstrap 95% CI vs dense-only (10k resamples):")
    print(f"{'variant':<28} {'ΔDoc@1 [95% CI]':<28} {'ΔPg@1 [95% CI]'}")
    boot = {}
    for tag in results:
        if tag == base:
            continue
        for mi, mname in ((0, "Doc@1"), (1, "Pg@1")):
            a = [hits[tag][q][mi] for q in qids]
            b = [hits[base][q][mi] for q in qids]
            boot.setdefault(tag, {})[mname] = paired_bootstrap(a, b)
        d = boot[tag]["Doc@1"]; p = boot[tag]["Pg@1"]
        star = lambda t: "*" if (t[1] > 0 or t[2] < 0) else " "
        print(f"{tag:<28} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]{star(d)}  "
              f"{p[0]:+.4f} [{p[1]:+.4f},{p[2]:+.4f}]{star(p)}")
    print("* = 95% CI excludes 0")

    results_out = {"metrics": results, "bootstrap_vs_dense": boot}
    json.dump(results_out, open(os.path.join(args.out_dir, "c1d_native_hybrid.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(top1, open(os.path.join(args.out_dir, "c1d_top1_docs.json"), "w"), ensure_ascii=False)
    # these per-query hits were COMPUTED above and then thrown away, which is why the
    # paper's single positive retrieval result (`bge_dense+sparse_dw0.9`) sat in no Holm family -- there was
    # no per-query artifact for a paired permutation test to run on. Serializing them puts it under
    # multiplicity control with every other UNLP contrast.
    _c1d = os.path.join(args.out_dir, "c1d_per_query_hits.json")
    json.dump(hits, open(_c1d, "w"), ensure_ascii=False)
    write_sidecar(_c1d, "bge_native_hybrid.py")   # item C10
    print(f"[done] {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

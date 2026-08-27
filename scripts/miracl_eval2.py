#!/usr/bin/env python3
"""A2 REBUILD — an experiment that CAN fail.

An earlier version of this evaluation reported ROC AUC = 1.000 for "document-flip rate predicts transfer
failure." That result is VOID (see the methodology audit). Three defects, all structural:

  (i)   THE BENCHMARK WAS SATURATED. An 8k-doc subsample of a 200k+ corpus put dense at
        nDCG@10 = 0.917 (published BGE-M3 on real MIRACL: ~0.70). With gold already at rank 1 for
        ~90% of queries, a sparse arm has no headroom to help, so the experiment was structurally
        incapable of observing anything but monotone damage.
  (ii)  THE POOL WAS A MONOTONE SWEEP OF ONE DIAL. Flip and damage are both monotone in the fusion
        weight, so AUC = 1.000 is arithmetic, not evidence. There were ZERO high-flip-BENEFICIAL
        variants (max flip among beneficial 0.124; min flip among harmful 0.332 -- no overlap).
  (iii) `flip_rate` IS A DISTANCE, NOT A RISK. It is literally `1 - agreement` with the baseline's
        top-1 doc, with no reference to relevance. A wrong->wrong swap counts as "risk"; swapping in
        an equally relevant near-duplicate counts as "risk".

This rebuild fixes the setup so the hypothesis is falsifiable:

  * FULL hard-negative corpus per language (no subsample) -- the official MTEB setting. This
    de-saturates the benchmark and removes the corpus-draw seed dependence at the same time.
  * A TUNED BM25 (scipy-vectorized, k1/b grid-searched on a query split DISJOINT from the eval
    split). The old arm was rank_bm25 at library defaults, never tuned -- a strawman that confounded
    "sparse fusion hurts" with "our sparse arm is broken".
  * Per-language tokenization folded into ONE lexical arm (char bigrams for zh/ja/th/KO -- Korean was
    missing from the old NO_SEG list), which also removes the duplicate `_seg`/`_raw` variants that
    inflated the old effective n.
  * THREE VARIANT CLASSES, not one:
      A. fusion sweep         -> expected: increasing flip, increasing damage   (the old evidence)
      B. rerankers            -> HIGH FLIP *and* BENEFICIAL                     (the missing class)
      C. pure-churn nulls     -> HIGH FLIP and ~ZERO relevance change           (the killer control)

    Class C is the decisive test. A churn null is dense scores perturbed by noise calibrated to flip
    a TARGET fraction of top-1 docs while leaving nDCG essentially unchanged. If `flip` flags it as
    risky -- and it must, since flip cannot see relevance -- then flip-alone is a distance proxy and
    not a risk signal, and we report that as the finding. Class B tests whether the CONJUNCTION rule
    ("flip AND no commensurate offline gain") has content beyond flip alone: a reranker flips top-1
    aggressively but buys a large offline gain, so the conjunction should clear it.

We expect flip-alone AUC to DROP. That is the honest negative, and it is the point of the rebuild.
"""
import argparse, gc, json, os, re, sys, time, unicodedata
import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import REPO
from rusbeir_eval import ndcg_at_k, recall_at_k, lex_to_csr, minmax_rows
from bm25_sparse import SparseBM25, tune_k1_b
from provenance import write_sidecar

HF_REPO = "mteb/MIRACLRetrievalHardNegatives"

LANGS = {
    "ru": ("Indo-European/Slavic",     "Cyrillic"),    # <- CALIBRATION language
    "en": ("Indo-European/Germanic",   "Latin"),
    "de": ("Indo-European/Germanic",   "Latin"),
    "es": ("Indo-European/Romance",    "Latin"),
    "fr": ("Indo-European/Romance",    "Latin"),
    "fa": ("Indo-European/Iranian",    "Arabic"),
    "hi": ("Indo-European/Indo-Aryan", "Devanagari"),
    "bn": ("Indo-European/Indo-Aryan", "Bengali"),
    "fi": ("Uralic",                   "Latin"),
    "ar": ("Afro-Asiatic/Semitic",     "Arabic"),
    "te": ("Dravidian",                "Telugu"),
    "ja": ("Japonic",                  "Japanese"),
    "ko": ("Koreanic",                 "Hangul"),
    "zh": ("Sino-Tibetan/Sinitic",     "Han"),
    "sw": ("Niger-Congo/Bantu",        "Latin"),
    "yo": ("Niger-Congo/Volta-Niger",  "Latin"),
    "th": ("Kra-Dai",                  "Thai"),
    "id": ("Austronesian",             "Latin"),
}

# languages with no whitespace word boundaries -> \w+ is not a valid tokenization.
# ko was MISSING from this list in the first round.
NO_SEG = {"zh", "ja", "th", "ko"}
TOK = re.compile(r"\w+", re.UNICODE)


def tok(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    if lang not in NO_SEG:
        return TOK.findall(s)
    chars = [c for c in s if not c.isspace()]
    return ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)] if len(chars) > 1 else chars


def load_miracl_full(lang):
    """FULL hard-negative corpus. No subsampling."""
    from datasets import load_dataset

    def first(cfg):
        ds = load_dataset(HF_REPO, cfg)
        return ds[list(ds.keys())[0]]

    qrels = {}
    for r in first(f"{lang}-qrels"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(float(r["score"]))
    qs = [{"id": str(r["id"]), "text": r["text"] or ""} for r in first(f"{lang}-queries")]
    qs = [q for q in qs if qrels.get(q["id"])]
    docs = []
    for r in first(f"{lang}-corpus"):
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        docs.append({"id": str(r["id"]), "text": f"{title}\n{text}" if title else text})
    return docs, qs, qrels


def evaluate(qs, qrels, doc_ids, scores, topk=50):
    """-> metrics, per-query top-1 doc, per-query nDCG@10."""
    idx = np.argpartition(-scores, kth=min(topk, scores.shape[1] - 1), axis=1)[:, :topk]
    top1, pq, rec, t1a = {}, {}, 0.0, 0.0
    for i, q in enumerate(qs):
        cand = idx[i][np.argsort(-scores[i, idx[i]])]
        ranked = [doc_ids[j] for j in cand]
        rel = qrels.get(q["id"], {})
        top1[q["id"]] = ranked[0] if ranked else None
        pq[q["id"]] = ndcg_at_k(ranked, rel, 10)
        rec += recall_at_k(ranked, rel, 10)
        t1a += 1.0 if rel.get(ranked[0], 0) > 0 else 0.0
    n = len(qs)
    return ({"nDCG@10": sum(pq.values()) / n, "Recall@10": rec / n, "Top1Acc": t1a / n, "N": n},
            top1, pq)


def flip_vs(top1_a, top1_b):
    ks = set(top1_a) & set(top1_b)
    return sum(top1_a[k] != top1_b[k] for k in ks) / len(ks) if ks else 0.0


def make_churn_null(dense, base_top1, qs, doc_ids, target_flip, seed=0, iters=28):
    """PURE-CHURN CONTROL (the decisive test).

    Perturb the dense scores with Gaussian noise scaled to flip ~`target_flip` of the top-1 docs
    while changing relevance as little as possible. Because the noise mostly reshuffles near-ties,
    nDCG should move very little -- but `flip_rate`, which cannot see relevance, must register a
    LARGE change. If the risk signal flags this as risky, it is measuring distance, not risk.

    Bisect the noise scale to hit the target flip rate.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(dense.shape).astype(np.float32)
    row_sd = dense.std(axis=1, keepdims=True) + 1e-9
    lo, hi = 0.0, 4.0
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        S = dense + mid * row_sd * noise
        t1 = {q["id"]: doc_ids[int(np.argmax(S[i]))] for i, q in enumerate(qs)}
        f = flip_vs(t1, base_top1)
        best = (mid, S, f)
        if f < target_flip:
            lo = mid
        else:
            hi = mid
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs/miracl2"))
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--tune-frac", type=float, default=0.4,
                    help="fraction of queries reserved for BM25 k1/b tuning (DISJOINT from eval)")
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--no-ce", action="store_true", help="skip the cross-encoder reranker")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mpath = os.path.join(args.out_dir, "metrics.json")
    tpath = os.path.join(args.out_dir, "top1_docs.json")
    all_m = json.load(open(mpath)) if os.path.exists(mpath) else {}
    all_t = json.load(open(tpath)) if os.path.exists(tpath) else {}

    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    ce = None
    if not args.no_ce:
        from FlagEmbedding import FlagReranker
        ce = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    for lang in args.langs.split(","):
        if lang in all_m:
            print(f"[{lang}] done, skip", file=sys.stderr); continue
        t0 = time.time()
        docs, qs, qrels = load_miracl_full(lang)
        if args.max_queries and len(qs) > args.max_queries:
            qs = qs[: args.max_queries]
        doc_ids = [d["id"] for d in docs]
        fam, script = LANGS[lang]
        print(f"[{lang}] {len(docs):,} docs (FULL), {len(qs)} queries  ({fam}, {script})",
              file=sys.stderr, flush=True)

        # ---- disjoint tune / eval query split (BM25 k1,b must NOT be tuned on the reporting set) ----
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(qs))
        n_tune = int(args.tune_frac * len(qs))
        tune_qs = [qs[i] for i in perm[:n_tune]]
        eval_qs = [qs[i] for i in perm[n_tune:]]

        # ---- BGE-M3 dense + sparse over the FULL corpus (cached) ----
        emb_cache = os.path.join(args.out_dir, f"emb_{lang}.npz")
        if os.path.exists(emb_cache):
            z = np.load(emb_cache, allow_pickle=True)
            D = z["dense"].astype(np.float32)
            Pl = sp.csr_matrix((z["lw_data"], z["lw_idx"], z["lw_ptr"]), shape=tuple(z["lw_shape"]))
        else:
            P = model.encode([d["text"] for d in docs], batch_size=32, max_length=512,
                             return_dense=True, return_sparse=True)
            D = P["dense_vecs"].astype(np.float32)
            Pl = lex_to_csr(P["lexical_weights"]).astype(np.float32)
            np.savez(emb_cache, dense=D.astype(np.float16), lw_data=Pl.data,
                     lw_idx=Pl.indices, lw_ptr=Pl.indptr, lw_shape=np.array(Pl.shape))
            D = D.astype(np.float32)
            # BGE-M3 returns lexical_weights as ONE PYTHON DICT PER DOCUMENT. On the large corpora
            # (en 179k, ja, fi, ar) that list outweighs every array here, and it stayed resident
            # through the score matmuls below -- which is what OOM-killed those four languages under
            # the 24G cgroup. Everything we need from P is already in D and Pl. Dropping it is pure
            # memory hygiene: it changes no arithmetic, so these languages remain bit-identical to
            # the fourteen scored before this line existed.
            del P
            gc.collect()
            print(f"  [enc] corpus encoded ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

        Q = model.encode([q["text"] for q in eval_qs], batch_size=32, max_length=256,
                         return_dense=True, return_sparse=True)
        dense = (Q["dense_vecs"].astype(np.float32) @ D.T).astype(np.float32)
        Ql = lex_to_csr(Q["lexical_weights"]).astype(np.float32)
        nsparse = np.asarray((Ql @ Pl.T).todense(), dtype=np.float32)

        # ---- TUNED BM25 (k1,b grid-searched on the tune split only) ----
        bm = SparseBM25().fit([tok(d["text"], lang) for d in docs])
        tq = [tok(q["text"], lang) for q in tune_qs]
        k1, b, tuned_ndcg, _grid = tune_k1_b(
            bm, tq, qrels, [q["id"] for q in tune_qs], doc_ids, ndcg_at_k)
        print(f"  [bm25] tuned k1={k1} b={b}  (tune-split nDCG@10={tuned_ndcg:.4f}, "
              f"n_tune={len(tune_qs)})", file=sys.stderr, flush=True)
        bm25 = bm.scores([tok(q["text"], lang) for q in eval_qs])

        dn, sn, bn = (minmax_rows(x) for x in (dense, nsparse, bm25))

        # baseline first (needed for flip + churn calibration)
        m0, t1_0, pq0 = evaluate(eval_qs, qrels, doc_ids, dense)
        variants = {"dense": (m0, t1_0, pq0)}

        def add(name, S):
            variants[name] = evaluate(eval_qs, qrels, doc_ids, S)

        # ---------- CLASS A: fusion sweep (expect monotone damage) ----------
        for w in (0.98, 0.95, 0.90, 0.80):
            add(f"native_ds_w{w:.2f}", w * dn + (1 - w) * sn)
        add("native_sparse", nsparse)
        for w in (0.95, 0.90, 0.70, 0.50, 0.30):
            add(f"hybrid_dw{w:.2f}", w * dn + (1 - w) * bn)
        add("bm25_tuned", bm25)

        # ---------- CLASS B: rerankers -- HIGH FLIP *and* BENEFICIAL ----------
        depth = args.rerank_depth
        topd = np.argsort(-dense, axis=1)[:, :depth]
        # B1: BGE-M3 ColBERT multi-vector rerank of the dense top-k
        cand_ids = sorted({int(j) for row in topd for j in row})
        pos = {j: i for i, j in enumerate(cand_ids)}
        CB = model.encode([docs[j]["text"] for j in cand_ids], batch_size=8, max_length=512,
                          return_dense=False, return_sparse=False, return_colbert_vecs=True)
        QB = model.encode([q["text"] for q in eval_qs], batch_size=16, max_length=256,
                          return_dense=False, return_sparse=False, return_colbert_vecs=True)
        S_cb = np.full_like(dense, -1e9)
        for i in range(len(eval_qs)):
            qv = QB["colbert_vecs"][i]
            for j in topd[i]:
                S_cb[i, j] = float(model.colbert_score(qv, CB["colbert_vecs"][pos[int(j)]]))
        add("colbert_rerank", S_cb)

        # B2: cross-encoder rerank of the dense top-k
        if ce is not None:
            S_ce = np.full_like(dense, -1e9)
            pairs, where = [], []
            for i in range(len(eval_qs)):
                for j in topd[i]:
                    pairs.append([eval_qs[i]["text"], docs[int(j)]["text"][:2000]])
                    where.append((i, int(j)))
            sc = ce.compute_score(pairs, batch_size=64, normalize=True)
            for (i, j), s in zip(where, sc):
                S_ce[i, j] = float(s)
            add("ce_rerank", S_ce)

        # ---------- CLASS C: pure-churn nulls -- HIGH FLIP, ~ZERO relevance change ----------
        for tgt in (0.25, 0.50):
            S_n, got, alpha = make_churn_null(dense, t1_0, eval_qs, doc_ids, tgt)
            add(f"churn_null_f{int(tgt*100)}", S_n)
            print(f"  [churn] target flip {tgt:.2f} -> got {got:.3f} (alpha={alpha:.3f})",
                  file=sys.stderr, flush=True)

        # ---------- record ----------
        rec = {"_family": fam, "_script": script, "_bm25_k1": k1, "_bm25_b": b,
               "_n_docs": len(docs), "_n_eval_q": len(eval_qs), "_n_tune_q": len(tune_qs)}
        t1s, pqs = {}, {}
        for name, (m, t1, pq) in variants.items():
            rec[name] = m
            t1s[name] = t1
            pqs[name] = pq
        for name in variants:
            if name != "dense":
                rec[name]["flip_vs_dense"] = flip_vs(t1s[name], t1s["dense"])
        all_m[lang] = rec
        all_t[lang] = t1s
        json.dump(pqs, open(os.path.join(args.out_dir, f"pq_ndcg_{lang}.json"), "w"))
        json.dump(all_m, open(mpath, "w"), ensure_ascii=False, indent=2)
        write_sidecar(mpath, "miracl_eval2.py")   # item C10: keys are languages -> sidecar, not inline
        json.dump(all_t, open(tpath, "w"), ensure_ascii=False)

        print(f"  {'variant':<22}{'nDCG@10':>9}{'flip':>7}", file=sys.stderr)
        for name in variants:
            f = rec[name].get("flip_vs_dense", 0.0)
            print(f"  {name:<22}{rec[name]['nDCG@10']:>9.4f}{f:>7.2f}", file=sys.stderr)
        print(f"[{lang}] done {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    print(f"[saved] {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()

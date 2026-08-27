#!/usr/bin/env python3
"""Standalone retrieval-evaluation harness (Option A + C1).

Faithfully reproduces the paper's Table 1 retrieval numbers on UNLP dev (BGE-M3 dense,
BM25, hybrid RRF) and extends them with: hybrid dense/sparse weight sweeps, per-query
oracle fusion, lemmatized-BM25 (pymorphy3), and per-query top-1-doc dumps for transfer-
risk flip-rate. The corpus/query loaders are pluggable so the same engine drives the
external-validation datasets (RusBEIR / MIRACL) later.

Faithful to notebooks/pipeline_shared.py: CLS pooling + L2 normalize, page max_length=1536,
query max_length=256, cosine via normalized inner product; doc ranking = order of first
appearance of each doc in the score-sorted page list (== best-page-per-doc).

No pandas dependency (numpy + torch + transformers + rank_bm25 only).
"""
import argparse, csv, glob, json, os, re, sys, time
from collections import defaultdict
import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from provenance import write_sidecar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# HF model id; resolves from HF_HOME cache or downloads (the local snapshot's blobs are pruned).
BGE_DIR = os.environ.get("BGE_MODEL", "BAAI/bge-m3")

# ----------------------------- loaders (pluggable) -----------------------------

def load_unlp_corpus(text_dir):
    """Return (units, doc_domain, doc_npages). units = list of (doc_id, page_num, text, domain)."""
    units, doc_domain, doc_npages = [], {}, {}
    for f in sorted(glob.glob(os.path.join(text_dir, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if not (isinstance(d, dict) and "doc_id" in d and "pages" in d):
            continue  # skip manifest.json / non-doc files
        doc_id, domain = d["doc_id"], d.get("domain", "")
        doc_domain[doc_id] = domain
        pages = d["pages"]
        doc_npages[doc_id] = len(pages)
        for pg in pages:
            units.append((doc_id, int(pg["page_num"]), pg.get("text") or "", domain))
    return units, doc_domain, doc_npages


def load_unlp_queries(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {"qid": r["Question_ID"], "text": r["Question"],
                   "gold_doc": r["Doc_ID"], "gold_page": int(r["Page_Num"]),
                   "domain": r["Domain"], "n_pages": int(r["N_Pages"]),
                   "gold_answer": r.get("Correct_Answer")}
            for o in ("A", "B", "C", "D", "E", "F"):   # MCQ options (needed by C3)
                row[o] = r.get(o, "")
            rows.append(row)
    return rows

# ----------------------------- tokenizers -----------------------------

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tok_raw(text):
    return TOKEN_RE.findall(text.lower())

_MORPH = None
_UK_STOP = set("і й та в на з до що як за від це не а по для о у б бути це той якщо або але".split())

def _lemmatize(words):
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer(lang="uk")
    return [_MORPH.parse(w)[0].normal_form for w in words]


# The original `tok_lemma` applied lemmatization AND stopword removal, while `tok_raw` applied
# NEITHER — so the reported "+8.7pp lemma gain" confounded two independent changes
#. These four arms form the 2x2 that separates them.
def tok_stop(text):
    """raw tokens, stopwords removed. Isolates the STOPLIST effect."""
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in _UK_STOP]


def tok_lemma_nostop(text):
    """lemmatized, stopwords KEPT. Isolates the LEMMATIZATION effect."""
    return _lemmatize(TOKEN_RE.findall(text.lower()))


def tok_lemma(text):
    """pymorphy3 Ukrainian lemmatization + stopword filter (both — the original C1b arm)."""
    return [w for w in _lemmatize(TOKEN_RE.findall(text.lower())) if w not in _UK_STOP]

# ----------------------------- BGE-M3 dense -----------------------------

_BGE = {}

def bge_encode(texts, max_length, batch=32):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    if not _BGE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _BGE["tok"] = AutoTokenizer.from_pretrained(BGE_DIR)
        _BGE["model"] = AutoModel.from_pretrained(
            BGE_DIR, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        _BGE["dev"] = dev
        print(f"[bge] loaded on {dev}", file=sys.stderr)
    tok, model, dev = _BGE["tok"], _BGE["model"], _BGE["dev"]
    embs = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**enc)
            e = F.normalize(out.last_hidden_state[:, 0], dim=-1)
        embs.append(e.cpu().float().numpy())
    return np.vstack(embs).astype("float32")

# ----------------------------- ranking + metrics -----------------------------

def doc_order_from_units(unit_scores, unit_docs):
    """Rank units by score desc; return doc ids in order of first appearance, + ranked unit idx."""
    order = np.argsort(-unit_scores)
    docs, seen = [], set()
    for idx in order:
        d = unit_docs[idx]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs, order


def per_query_hits(queries, per_query_ranked):
    """{qid: (doc@1 hit, pg@1 hit)} — paired inputs for bootstrap significance tests."""
    out = {}
    for q in queries:
        doc_order, ranked_units = per_query_ranked[q["qid"]]
        d1 = 1.0 if doc_order[:1] == [q["gold_doc"]] else 0.0
        p1 = 1.0 if ranked_units[:1] == [(q["gold_doc"], q["gold_page"])] else 0.0
        out[q["qid"]] = (d1, p1)
    return out


def paired_bootstrap(a, b, n=10000, seed=42):
    """95% CI on mean(a-b) for paired per-query indicators."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(queries, per_query_ranked, doc_domain, k_list=(1, 10)):
    """per_query_ranked[qid] = (doc_order:list[str], ranked_units:list[(doc,page)])."""
    agg = defaultdict(float)
    per_domain = defaultdict(lambda: defaultdict(float))
    dcount = defaultdict(int)
    top1 = {}
    for q in queries:
        qid, gd, gp, dom = q["qid"], q["gold_doc"], q["gold_page"], q["domain"]
        doc_order, ranked_units = per_query_ranked[qid]
        top1[qid] = doc_order[0] if doc_order else None
        dcount[dom] += 1
        # Doc@k
        for k in k_list:
            hit = 1.0 if gd in doc_order[:k] else 0.0
            agg[f"Doc@{k}"] += hit
            per_domain[dom][f"Doc@{k}"] += hit
        # MRR over docs
        rank = next((i + 1 for i, d in enumerate(doc_order) if d == gd), None)
        agg["MRR_doc"] += (1.0 / rank) if rank else 0.0
        # Pg@k over (doc,page) units
        for k in k_list:
            hit = 1.0 if (gd, gp) in ranked_units[:k] else 0.0
            agg[f"Pg@{k}"] += hit
    n = len(queries)
    metrics = {m: agg[m] / n for m in agg}
    metrics["N"] = n
    dm = {dom: {m: per_domain[dom][m] / dcount[dom] for m in per_domain[dom]}
          for dom in per_domain}
    return metrics, dm, top1

# ----------------------------- retrieval methods -----------------------------

def run_dense(queries, units, page_embs, q_embs):
    unit_docs = [u[0] for u in units]
    unit_dp = [(u[0], u[1]) for u in units]
    scores = q_embs @ page_embs.T  # (Q, U)
    out = {}
    for i, q in enumerate(queries):
        doc_order, order = doc_order_from_units(scores[i], unit_docs)
        out[q["qid"]] = (doc_order, [unit_dp[j] for j in order])
    return out, scores


def build_bm25(units, tok, tune_queries=None, gold=None):
    """Tuned Okapi BM25. The original used rank_bm25 at library defaults (k1=1.5, b=0.75) and never
    tuned them -- a strawman that confounded "sparse fusion hurts" with "our sparse arm is
    misconfigured". k1/b are grid-searched on a DISJOINT query split when one is
    supplied."""
    from bm25_sparse import SparseBM25
    bm = SparseBM25().fit([tok(u[2]) for u in units])
    if tune_queries and gold:
        unit_docs = [u[0] for u in units]
        best = (bm.k1, bm.b, -1.0)
        for k1 in (0.6, 0.9, 1.2, 1.5, 1.8, 2.1):
            for b in (0.3, 0.5, 0.75, 0.9, 1.0):
                bm.set_params(k1, b)
                S = bm.scores([tok(q["text"]) for q in tune_queries])
                hit = 0
                for i, q in enumerate(tune_queries):
                    order = np.argsort(-S[i])
                    seen = []
                    for j in order:
                        if unit_docs[j] not in seen:
                            seen.append(unit_docs[j])
                        if len(seen) >= 1:
                            break
                    if seen and seen[0] == q["gold_doc"]:
                        hit += 1
                acc = hit / max(1, len(tune_queries))
                if acc > best[2]:
                    best = (k1, b, acc)
        bm.set_params(best[0], best[1])
        print(f"[bm25] tuned k1={best[0]} b={best[1]} (tune-split Doc@1={best[2]:.4f}, "
              f"n_tune={len(tune_queries)})", file=sys.stderr)
    return bm


def run_bm25(queries, units, bm25, tok):
    unit_docs = [u[0] for u in units]
    unit_dp = [(u[0], u[1]) for u in units]
    out, all_scores = {}, {}
    for q in queries:
        s = bm25.scores([tok(q["text"])])[0]
        doc_order, order = doc_order_from_units(s, unit_docs)
        out[q["qid"]] = (doc_order, [unit_dp[j] for j in order])
        all_scores[q["qid"]] = s
    return out, all_scores


def run_hybrid(queries, units, dense_scores, sparse_scores, dense_w, sparse_w, rrf_k, sparse_top_k):
    """RRF over units, matching pipeline: weight/(k+rank), sparse limited to sparse_top_k."""
    unit_docs = [u[0] for u in units]
    unit_dp = [(u[0], u[1]) for u in units]
    out = {}
    for i, q in enumerate(queries):
        ds = dense_scores[i]
        ss = sparse_scores[q["qid"]]
        drank = {idx: r for r, idx in enumerate(np.argsort(-ds))}
        srank = {idx: r for r, idx in enumerate(np.argsort(-ss)[:sparse_top_k])}
        fused = np.full(len(units), 0.0, dtype="float32")
        for idx in range(len(units)):
            v = dense_w / (rrf_k + drank[idx])
            if idx in srank:
                v += sparse_w / (rrf_k + srank[idx])
            fused[idx] = v
        doc_order, order = doc_order_from_units(fused, unit_docs)
        out[q["qid"]] = (doc_order, [unit_dp[j] for j in order])
    return out


def oracle_fusion(queries, dense_top1, sparse_top1):
    """Per-query: correct if EITHER dense or sparse top-1 doc is gold. Upper bound on fusion."""
    hit = 0
    for q in queries:
        gd = q["gold_doc"]
        if dense_top1.get(q["qid"]) == gd or sparse_top1.get(q["qid"]) == gd:
            hit += 1
    return hit / len(queries)

# ----------------------------- main -----------------------------

def fmt(m):
    return (f"Doc@1={m.get('Doc@1',0):.4f} Doc@10={m.get('Doc@10',0):.4f} "
            f"Pg@1={m.get('Pg@1',0):.4f} Pg@10={m.get('Pg@10',0):.4f} MRR={m.get('MRR_doc',0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dir", default=os.path.join(REPO, "data/extracted_text"))
    ap.add_argument("--queries", default=os.path.join(REPO, "data/dev_questions.csv"))
    ap.add_argument("--methods", default="dense,bm25_raw,bm25_lemma,hybrid_raw,hybrid_lemma")
    ap.add_argument("--dense-weights", default="0.5",
                    help="comma list of dense weights for hybrid sweep (sparse=1-w unless --equal)")
    ap.add_argument("--equal-weights", action="store_true", help="use dense=sparse=1.0 (pipeline default)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--sparse-top-k", type=int, default=40)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs/retrieval_eval"))
    ap.add_argument("--cache-embs", default=None, help="npy path to cache/load page+query embs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    methods = args.methods.split(",")
    t0 = time.time()

    units, doc_domain, doc_npages = load_unlp_corpus(args.text_dir)
    queries = load_unlp_queries(args.queries)
    print(f"[data] {len(units)} page-units, {len(doc_domain)} docs, {len(queries)} queries",
          file=sys.stderr)

    cache = args.cache_embs or os.path.join(args.out_dir, "unlp_embs.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        page_embs, q_embs = z["page"], z["query"]
        print(f"[emb] loaded cache {cache}", file=sys.stderr)
    else:
        page_embs = bge_encode([u[2] for u in units], max_length=1536)
        q_embs = bge_encode([q["text"] for q in queries], max_length=256)
        np.savez(cache, page=page_embs, query=q_embs)
        print(f"[emb] encoded + cached -> {cache}", file=sys.stderr)

    results = {}       # method -> metrics
    top1_dumps = {}    # method -> {qid: top1_doc}
    hits = {}          # method -> {qid: (doc@1, pg@1)} for paired bootstrap

    dense_ranked, dense_scores = run_dense(queries, units, page_embs, q_embs)
    # always compute the dense baseline's hits (bootstrap reference)
    hits["dense"] = per_query_hits(queries, dense_ranked)
    if "dense" in methods:
        m, dm, t1 = evaluate(queries, dense_ranked, doc_domain)
        results["dense"] = {"overall": m, "per_domain": dm}
        top1_dumps["dense"] = t1
        print(f"dense        {fmt(m)}")

    bm25_cache = {}
    for tokname, tokfn in (("raw", tok_raw), ("lemma", tok_lemma)):
        need = (f"bm25_{tokname}" in methods) or (f"hybrid_{tokname}" in methods)
        if not need:
            continue
        bm25 = build_bm25(units, tokfn)
        ranked, scores = run_bm25(queries, units, bm25, tokfn)
        bm25_cache[tokname] = (ranked, scores)
        if f"bm25_{tokname}" in methods:
            m, dm, t1 = evaluate(queries, ranked, doc_domain)
            results[f"bm25_{tokname}"] = {"overall": m, "per_domain": dm}
            top1_dumps[f"bm25_{tokname}"] = t1
            hits[f"bm25_{tokname}"] = per_query_hits(queries, ranked)
            print(f"bm25_{tokname:<6}  {fmt(m)}")

    dense_top1 = {q["qid"]: dense_ranked[q["qid"]][0][0] for q in queries}
    weights = [1.0] if args.equal_weights else [float(x) for x in args.dense_weights.split(",")]
    for tokname in ("raw", "lemma"):
        if f"hybrid_{tokname}" not in methods or tokname not in bm25_cache:
            continue
        _, sparse_scores = bm25_cache[tokname]
        sparse_top1 = {q["qid"]: bm25_cache[tokname][0][q["qid"]][0][0] for q in queries}
        # oracle upper bound
        orc = oracle_fusion(queries, dense_top1, sparse_top1)
        print(f"oracle(dense|bm25_{tokname}) Doc@1={orc:.4f}")
        for dw in weights:
            sw = 1.0 if args.equal_weights else (1.0 - dw)
            ranked = run_hybrid(queries, units, dense_scores, sparse_scores,
                                dw, sw, args.rrf_k, args.sparse_top_k)
            m, dm, t1 = evaluate(queries, ranked, doc_domain)
            tag = f"hybrid_{tokname}_dw{dw:.2f}_sw{sw:.2f}"
            results[tag] = {"overall": m, "per_domain": dm, "oracle_doc1": orc}
            top1_dumps[tag] = t1
            hits[tag] = per_query_hits(queries, ranked)
            print(f"{tag:<28} {fmt(m)}")

    # ---- paired bootstrap vs dense baseline (methodology gate) ----
    qids = sorted(hits["dense"])
    print("\nPaired bootstrap 95% CI vs dense (10k resamples):")
    print(f"{'variant':<28} {'ΔDoc@1 [95% CI]':<30} {'ΔPg@1 [95% CI]'}")
    boot = {}
    for tag in hits:
        if tag == "dense":
            continue
        for mi, mn in ((0, "Doc@1"), (1, "Pg@1")):
            a = [hits[tag][q][mi] for q in qids]
            b = [hits["dense"][q][mi] for q in qids]
            boot.setdefault(tag, {})[mn] = paired_bootstrap(a, b)
        d, p = boot[tag]["Doc@1"], boot[tag]["Pg@1"]
        st = lambda t: "*" if (t[1] > 0 or t[2] < 0) else " "
        print(f"{tag:<28} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]{st(d)}   "
              f"{p[0]:+.4f} [{p[1]:+.4f},{p[2]:+.4f}]{st(p)}")
    print("* = 95% CI excludes 0")

    json.dump({"metrics": results, "bootstrap_vs_dense": boot},
              open(os.path.join(args.out_dir, "metrics.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(top1_dumps, open(os.path.join(args.out_dir, "top1_docs.json"), "w"),
              ensure_ascii=False)
    # per-query hits, needed downstream for exact permutation p-values + Holm-Bonferroni
    _pqh = os.path.join(args.out_dir, "per_query_hits.json")
    json.dump({tag: {q: list(v) for q, v in d.items()} for tag, d in hits.items()},
              open(_pqh, "w"))
    # Item C10: this file outlived a fix to this very script once, and Holm silently mixed two code
    # versions. Its top-level keys ARE the variant names, so the stamp goes in a sidecar.
    write_sidecar(_pqh, "retrieval_eval.py")
    print(f"[done] {time.time()-t0:.1f}s -> {args.out_dir}/metrics.json", file=sys.stderr)


if __name__ == "__main__":
    main()

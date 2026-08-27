#!/usr/bin/env python3
"""C2 §5.2 cluster-robust bootstrap.

The four 2x2 visual-retrieval effects in `outputs/colsmol2/*_2x2.json` were
given i.i.d. paired-bootstrap CIs (`retrieval_eval.paired_bootstrap`). But the
200 queries cluster over 41 gold documents, so their per-query hits are
correlated and those CIs are too NARROW -- precisely the class of error this
paper is about. This script recomputes every effect's interval with the
nonparametric cluster bootstrap (clusters = gold documents), reading the
serialized `per_query_hits` (NO model rerun), and stores BOTH the i.i.d. and the
clustered intervals so the widening is auditable. Point estimates are unchanged
(the observed mean); only the intervals move.

Inputs (provenance-pinned): the two `*_2x2.json` and `data/dev_questions.csv`
(the qid -> gold_doc join). Output: `outputs/colsmol2/c2_cluster_bootstrap.json`.

`cluster_bootstrap` lives HERE, not in retrieval_eval.py, on purpose: editing
retrieval_eval.py would mark its expensive downstream artifacts
(per_query_hits.json, a BGE-M3 retrieval output) STALE under provenance's
per-file rule, forcing a GPU rerun for a change that does not affect them. This
analysis is the function's only consumer, so it owns it. `paired_bootstrap` is
imported from retrieval_eval unchanged, so the i.i.d. numbers stay identical.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import load_unlp_queries, paired_bootstrap, REPO
from provenance import stamp


def cluster_bootstrap(a, b, clusters, n=10000, seed=42):
    """95% CI on mean(a-b) for paired per-query indicators, resampling CLUSTERS
    (here: gold documents) with replacement instead of individual queries.

    Nonparametric cluster bootstrap: draw the G distinct clusters with
    replacement, pool every observation belonging to each drawn cluster, take
    the mean over the pooled set. Accounts for the within-cluster correlation
    that paired_bootstrap ignores, so the interval is (correctly) wider. The
    point estimate is identical -- it is the observed mean(a-b). Returns
    (mean, lo, hi, n_clusters).
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [np.where(clusters == u)[0] for u in uniq]
    G = len(groups)
    means = np.empty(n, dtype="float64")
    for i in range(n):
        pick = rng.integers(0, G, size=G)
        idx = np.concatenate([groups[j] for j in pick])
        means[i] = d[idx].mean()
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(G))

MODELS = {"colSmol-500M":    "outputs/colsmol2/colSmol-500M_2x2.json",
          "colqwen2.5-v0.2": "outputs/colsmol2/colqwen2.5-v0.2_2x2.json"}
QUERIES_CSV = "data/dev_questions.csv"
OUT = "outputs/colsmol2/c2_cluster_bootstrap.json"

# effect name -> (minuend cell, subtrahend cell). Doc@1 is index 0 of each hit.
CELLS = {"page":  ("page-lat_query-cyr", "page-cyr_query-cyr"),   # c - a
         "query": ("page-cyr_query-lat", "page-cyr_query-cyr"),   # b - a
         "both":  ("page-lat_query-lat", "page-cyr_query-cyr")}   # d - a


def main():
    qid2doc = {r["qid"]: r["gold_doc"]
               for r in load_unlp_queries(os.path.join(REPO, QUERIES_CSV))}
    out = {}
    for tag, rel in MODELS.items():
        d = json.load(open(os.path.join(REPO, rel)))
        pqh = d["per_query_hits"]
        qids = sorted(pqh["page-cyr_query-cyr"].keys())
        missing = [q for q in qids if q not in qid2doc]
        if missing:
            raise SystemExit(f"[c2_cluster] {tag}: {len(missing)} qids have no "
                             f"gold_doc join: {missing[:5]}")
        clusters = [qid2doc[q] for q in qids]
        n_clusters = len(set(clusters))
        eff = {}
        for name, (cx, cy) in CELLS.items():
            x = [pqh[cx][q][0] for q in qids]
            y = [pqh[cy][q][0] for q in qids]
            m_i, lo_i, hi_i = paired_bootstrap(x, y)
            m_c, lo_c, hi_c, G = cluster_bootstrap(x, y, clusters)
            assert G == n_clusters, (G, n_clusters)
            eff[name] = {"delta": m_c,
                         "iid_ci": [lo_i, hi_i],
                         "cluster_ci": [lo_c, hi_c],
                         "iid_sig": bool(lo_i > 0 or hi_i < 0),
                         "cluster_sig": bool(lo_c > 0 or hi_c < 0)}
        out[tag] = {"n_queries": len(qids), "n_clusters": n_clusters, "effects": eff}

    inputs = [os.path.join(REPO, rel) for rel in MODELS.values()] + \
             [os.path.join(REPO, QUERIES_CSV)]
    dst = os.path.join(REPO, OUT)
    json.dump(stamp(out, "c2_cluster_bootstrap.py", inputs=inputs),
              open(dst, "w"), indent=2)
    print(f"[saved] {dst}")
    for tag, o in out.items():
        print(f"\n{tag}: {o['n_queries']} queries / {o['n_clusters']} clusters")
        for name, e in o["effects"].items():
            si = "*" if e["iid_sig"] else " "
            sc = "*" if e["cluster_sig"] else " "
            print(f"  {name:6} d={e['delta']:+.4f}  "
                  f"iid[{e['iid_ci'][0]:+.4f},{e['iid_ci'][1]:+.4f}]{si}  "
                  f"cluster[{e['cluster_ci'][0]:+.4f},{e['cluster_ci'][1]:+.4f}]{sc}")


if __name__ == "__main__":
    main()

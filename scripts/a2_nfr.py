#!/usr/bin/env python3
"""Decompose flip rate into its labelled parts. This is the experiment that decides contribution A.

`flip_rate` is literally `1 - agreement` with the baseline's top-1 document. It never looks at a
relevance label. So a change from a WRONG document to a DIFFERENT WRONG document counts as "risk",
and swapping in an EQUALLY RELEVANT near-duplicate counts as "risk". We sold a distance as a risk.

The literature named this a decade before we did, and named the fix:

  * Milani Fard et al., NeurIPS 2016, "Launch and Iterate: Reducing Prediction Churn" -- defines
    churn as the fraction of predictions that differ from the incumbent, and states in the abstract
    that much of it is "unnecessary neutral changes", net-zero wins and losses.
  * Xie et al., ACL 2021, "Regression Bugs Are In Your Model!" -- defines the NEGATIVE FLIP RATE:
    flips from CORRECT to INCORRECT. That is flip rate with the relevance label attached.
  * Marx et al., ICML 2020, "Predictive Multiplicity in Classification" -- two equally good models
    can disagree on many individual instances. Flip cannot tell that apart from degradation, by
    construction, because it never sees a label.

So we decompose every flip, using the qrels we already have:

    flip = NFR + PFR + NEUTRAL

    NFR      baseline top-1 was RELEVANT, variant top-1 is NOT     -- genuine harm
    PFR      baseline top-1 was NOT relevant, variant top-1 IS     -- genuine help
    NEUTRAL  both relevant, or both irrelevant, but different docs -- pure churn, ZERO information

If NEUTRAL dominates flip, then flip rate is mostly noise about relevance and the signal is dead as
framed -- not because our AUC was bad, but because the quantity was never about correctness. And the
decomposition says so in a way no threshold-tuning can rescue.

Reads ids only (streaming qrels), so it is safe on a login node. No GPU, no re-run: `top1_docs.json`
already stores the per-query top-1 document of every variant.
"""
import argparse, json, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_REPO = "mteb/MIRACLRetrievalHardNegatives"


def load_qrels(lang):
    """Mirror miracl_eval2.load_miracl_full()'s accessor exactly: first split, whatever its name."""
    from datasets import load_dataset
    ds = load_dataset(HF_REPO, f"{lang}-qrels", streaming=True)
    split = ds[list(ds.keys())[0]]
    rel = defaultdict(set)
    for r in split:
        if int(float(r["score"])) > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(REPO, "outputs/miracl2"))
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/miracl2/nfr_decomposition.json"))
    args = ap.parse_args()

    top1 = json.load(open(os.path.join(args.in_dir, "top1_docs.json")))
    metrics = json.load(open(os.path.join(args.in_dir, "metrics.json")))
    langs = sorted(top1)

    cache = args.out + ".qrels_cache"
    qc = json.load(open(cache)) if os.path.exists(cache) else {}

    per_lang = {}
    for L in langs:
        if L not in qc:
            print(f"[{L}] loading qrels ...", file=sys.stderr, flush=True)
            qc[L] = {q: sorted(d) for q, d in load_qrels(L).items()}
            json.dump(qc, open(cache, "w"))
        rel = {q: set(d) for q, d in qc[L].items()}
        base = top1[L]["dense"]
        out = {}
        for v, t1 in top1[L].items():
            if v == "dense":
                continue
            nfr = pfr = neutral = flip = 0
            n = 0
            for q, dv in t1.items():
                db = base.get(q)
                if db is None:
                    continue
                n += 1
                if dv == db:
                    continue
                flip += 1
                rb, rv = (db in rel.get(q, ())), (dv in rel.get(q, ()))
                if rb and not rv:
                    nfr += 1
                elif rv and not rb:
                    pfr += 1
                else:
                    neutral += 1
            out[v] = {"n": n, "flip": flip / n, "nfr": nfr / n, "pfr": pfr / n,
                      "neutral": neutral / n,
                      "neutral_share_of_flip": (neutral / flip) if flip else float("nan"),
                      "net_flip": (pfr - nfr) / n,
                      "dndcg": metrics[L][v]["nDCG@10"] - metrics[L]["dense"]["nDCG@10"]}
        per_lang[L] = out

    variants = sorted({v for L in per_lang for v in per_lang[L]})
    agg = {}
    for v in variants:
        rows = [per_lang[L][v] for L in langs if v in per_lang[L]]
        agg[v] = {k: st.mean([r[k] for r in rows if r[k] == r[k]])
                  for k in ("flip", "nfr", "pfr", "neutral", "neutral_share_of_flip",
                            "net_flip", "dndcg")}

    json.dump(stamp({"languages": langs, "per_language": per_lang, "aggregate": agg}, "a2_nfr.py"),
              open(args.out, "w"), indent=2)

    print(f"\n=== flip decomposed, {len(langs)} languages "
          f"(flip = NFR + PFR + NEUTRAL) ===\n")
    print(f"{'variant':20s} {'flip':>6s} {'NFR':>6s} {'PFR':>6s} {'NEUTRAL':>8s} "
          f"{'%flip neutral':>13s} {'net':>7s} {'dNDCG':>8s}")
    for v in sorted(variants, key=lambda x: -agg[x]["flip"]):
        a = agg[v]
        print(f"{v:20s} {a['flip']:6.3f} {a['nfr']:6.3f} {a['pfr']:6.3f} {a['neutral']:8.3f} "
              f"{a['neutral_share_of_flip']*100:12.1f}% {a['net_flip']:+7.3f} {a['dndcg']:+8.4f}")

    ns = st.mean([agg[v]["neutral_share_of_flip"] for v in variants])
    print(f"\nMean share of flips that carry NO relevance information: {ns*100:.1f}%")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

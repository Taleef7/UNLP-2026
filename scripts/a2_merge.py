#!/usr/bin/env python3
"""Merge the per-worker MIRACL out-dirs into the canonical outputs/miracl2.

A2 was sharded across parallel workers, each with a private out-dir, because miracl_eval2.py
rewrites metrics.json wholesale after every language and concurrent workers would clobber one
another. The language keys are disjoint BY INTENTION -- but we do not assume that, we check it.

What "collision" must mean, precisely:
  * The same language scored twice with the SAME numbers is not a conflict. It happens whenever a
    merge is re-run (this script is called from a watcher that fires on worker exit), or when a
    language is retried on a bigger-memory node after an OOM. Re-merging must be idempotent, or the
    watcher can never run twice.
  * The same language scored twice with DIFFERENT numbers is a real conflict: one of the two results
    is about to be silently dropped, and we would have no idea which one the paper reported. That is
    exactly the class of error this paper is about, so it is a hard failure -- never a warning.

So: dedupe on equality, raise on disagreement.
"""
import glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(REPO, "outputs/miracl2")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    workers = sorted(glob.glob(os.path.join(REPO, "outputs/miracl2_w*")))
    if not workers:
        sys.exit("no worker dirs found")

    metrics = load(os.path.join(CANON, "metrics.json"))
    top1 = load(os.path.join(CANON, "top1_docs.json"))
    provenance = {L: "canonical" for L in metrics}
    added, redundant = [], []

    for w in workers:
        tag = os.path.basename(w)
        wm, wt = load(os.path.join(w, "metrics.json")), load(os.path.join(w, "top1_docs.json"))
        for L in wm:
            if L in metrics:
                if metrics[L] == wm[L]:
                    # Same language, identical numbers -- a re-merge or a redundant retry. Harmless.
                    redundant.append(f"{L} ({tag} == {provenance[L]})")
                    continue
                raise SystemExit(
                    f"\nCONFLICT: language {L!r} was scored in BOTH {provenance[L]} and {tag}, and the "
                    f"two results DISAGREE.\n"
                    f"Merging would silently drop one of them and we would not know which one the paper "
                    f"reported. Refusing.\n"
                    f"Inspect both metrics.json, decide which run is authoritative, and delete the other.\n")
            metrics[L], provenance[L] = wm[L], tag
            added.append(L)
            if L in wt:
                top1[L] = wt[L]
        # per-query nDCG files are already per-language; copy them across
        for src in glob.glob(os.path.join(w, "pq_ndcg_*.json")):
            dst = os.path.join(CANON, os.path.basename(src))
            if not os.path.exists(dst):
                open(dst, "w").write(open(src).read())

    os.makedirs(CANON, exist_ok=True)
    json.dump(metrics, open(os.path.join(CANON, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(top1, open(os.path.join(CANON, "top1_docs.json"), "w"), ensure_ascii=False)

    langs = sorted(metrics)
    print(f"[merged] {len(langs)} languages -> {CANON}   (+{len(added)} new this run)")
    for L in langs:
        nv = len([k for k in metrics[L] if not k.startswith("_")])
        print(f"   {L:3s}  {nv:2d} variants   (from {provenance[L]})")
    if redundant:
        print(f"[idempotent] {len(redundant)} already-present, identical: {', '.join(redundant)}")


if __name__ == "__main__":
    main()

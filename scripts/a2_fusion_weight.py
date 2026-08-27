#!/usr/bin/env python3
"""Is "sparse fusion damages dense retrieval" a finding, or is it a bad default?

The de-saturated MIRACL run (18 languages, FULL corpora) says the fusion weight decides:

    dw=0.95  +0.0144   helps in 17/18 languages
    dw=0.90  +0.0214   helps in 16/18
    dw=0.70  -0.0153   helps in  5/18
    dw=0.50  -0.1242   helps in  2/18      <-- EQUAL WEIGHT
    dw=0.30  -0.2438   helps in  0/18

So the damage we reported is real AT AN EQUAL WEIGHT and disappears -- reverses, in fact -- at a
dense-dominant weight. That is a claim about a hyperparameter default, not about fusion.

CAREFUL (item C5): dw=0.50 here is NOT literally the config we shipped. The shipped UNLP pipeline used
weighted RRF with the sparse arm truncated at depth 40; this is min-max score fusion at full depth.
Both are "equal weight" in spirit and both are harmful, but they are DIFFERENT OPERATORS and their
magnitudes must never be averaged or quoted as one number.

BUT: reading "dw=0.90 is best" off that table is exactly the grid-endpoint / test-set-selection error
we filed against ourselves as item B3. The weight would be chosen on the same split it is scored on.

So we select the weight WITHOUT ever looking at the language we score it on: leave-one-language-out.
For each held-out language, pick the dw that maximises MEAN gain over the OTHER languages, then apply
that dw to the held-out language and report only that. No language contributes to the choice of the
weight it is evaluated under. The reported gain is an honest out-of-sample estimate of what you get
from "choose a fusion weight sensibly" versus "use the equal-weight default".

We also report the pre-specified equal weight (dw=0.50) with no selection at all -- it needs none: it
is a fixed, declared-in-advance setting, not a fitted one.
"""
import argparse, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Declared in source, not discovered: the fusion-weight grid and the equal-weight reference point.
GRID = ["hybrid_dw0.95", "hybrid_dw0.90", "hybrid_dw0.70", "hybrid_dw0.50", "hybrid_dw0.30"]
DEFAULT = "hybrid_dw0.50"          # equal weight (dense 0.5 / sparse 0.5), pre-specified
BASE = "dense"
DW = {v: float(v.split("dw")[1]) for v in GRID}


def gain(m, L, v):
    return m[L][v]["nDCG@10"] - m[L][BASE]["nDCG@10"]


def boot_ci(xs, n=10000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(xs, dtype=float)
    bs = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(REPO, "outputs/miracl2"))
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/miracl2/fusion_weight.json"))
    args = ap.parse_args()

    m = json.load(open(os.path.join(args.in_dir, "metrics.json")))
    langs = sorted(m)
    for L in langs:
        missing = [v for v in GRID + [BASE] if v not in m[L]]
        if missing:
            sys.exit(f"{L} is missing declared variants {missing}")

    # ---- 1. the full sweep, for the record (selection-free: every cell reported) ----
    sweep = {}
    for v in GRID:
        g = [gain(m, L, v) for L in langs]
        sweep[v] = {"dw": DW[v], "mean_gain": st.mean(g),
                    "n_help": sum(1 for x in g if x > 0), "n_lang": len(langs),
                    "ci95": boot_ci(g), "per_lang": {L: gain(m, L, v) for L in langs}}

    # ---- 2. the pre-specified default: no selection, nothing to correct for ----
    dflt = sweep[DEFAULT]

    # ---- 3. LOLO-selected weight: the language never sees the weight chosen for it ----
    lolo = {}
    for L in langs:
        others = [x for x in langs if x != L]
        pick = max(GRID, key=lambda v: st.mean([gain(m, o, v) for o in others]))
        lolo[L] = {"selected": pick, "dw": DW[pick], "gain_on_heldout": gain(m, L, pick)}
    lg = [lolo[L]["gain_on_heldout"] for L in langs]
    picks = [lolo[L]["selected"] for L in langs]

    out = {
        "n_languages": len(langs), "languages": langs, "grid": GRID, "default": DEFAULT,
        "sweep": sweep,
        "default_no_selection": {"variant": DEFAULT, "mean_gain": dflt["mean_gain"],
                                 "ci95": dflt["ci95"], "n_help": dflt["n_help"]},
        "lolo_selected": {
            "mean_gain_heldout": st.mean(lg), "ci95": boot_ci(lg),
            "n_help": sum(1 for x in lg if x > 0), "n_lang": len(langs),
            "weights_chosen": {v: picks.count(v) for v in GRID if picks.count(v)},
            "per_lang": lolo},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(stamp(out, "a2_fusion_weight.py"), open(args.out, "w"), indent=2)

    print(f"=== fusion weight, {len(langs)} languages ===\n")
    print(f"{'variant':16s} {'dw':>5s} {'mean gain':>10s} {'95% CI':>20s}  helps")
    for v in GRID:
        s = sweep[v]
        print(f"{v:16s} {s['dw']:5.2f} {s['mean_gain']:+10.4f} "
              f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]  {s['n_help']:2d}/{s['n_lang']}")
    print(f"\n-- PRE-SPECIFIED DEFAULT (no selection): {DEFAULT}")
    print(f"   mean gain {dflt['mean_gain']:+.4f}  CI [{dflt['ci95'][0]:+.4f},{dflt['ci95'][1]:+.4f}]  "
          f"helps in {dflt['n_help']}/{len(langs)}")
    print(f"\n-- LOLO-SELECTED WEIGHT (weight chosen on the OTHER languages, scored on the held-out one)")
    print(f"   mean gain {st.mean(lg):+.4f}  CI [{out['lolo_selected']['ci95'][0]:+.4f},"
          f"{out['lolo_selected']['ci95'][1]:+.4f}]  helps in {out['lolo_selected']['n_help']}/{len(langs)}")
    print(f"   weights chosen: {out['lolo_selected']['weights_chosen']}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

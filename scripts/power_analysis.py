#!/usr/bin/env python3
"""Design-stage MDE + equivalence bounds for the UNLP dev set (N=461).

TWO THINGS THIS SCRIPT FIXES ABOUT ITS OWN PREVIOUS VERSION:

B1 — THE MDE WAS COMPUTED WRONG, ANTI-CONSERVATIVELY.
  The old simulator set the discordant-pair counts deterministically
  (`k01 = int(round((churn+delta)*n))`, `k10 = int(round(churn*n))`). With those counts fixed, every
  replicate realises EXACTLY the true effect: the estimator is pinned to `delta` and the only
  randomness left is *which* items flip. The routine therefore measured `delta > 1.96*SE` -- the
  ~50% power point -- and reported it as 80% power. It produced a step function (0.010 -> 0.893
  across one 0.5pp grid step) and an MDE of 3.0pp where the true 80% MDE is ~3.7pp. The rigor script
  was wrong by ~25%, in the direction that flattered us.
  FIX: draw n01, n10 ~ Binomial per replicate, so delta_hat has its real sampling variance.
  Cross-checked against the closed-form McNemar MDE:  delta_MDE ≈ (z_{α/2} + z_β) * sqrt(p_d / n).

B2 — POST-HOC POWER IS A FALLACY (Hoenig & Heisey 2001, "The Abuse of Power").
  The old C1 write-up used the MDE to explain away a NON-SIGNIFICANT OBSERVED EFFECT ("our +0.9pp is
  far below the MDE, so the CIs spanning zero are exactly what the design predicts"). Observed power
  is a monotone function of the p-value; it carries no information and cannot license "the null is
  due to low power." Card et al. (2020) advocate a DESIGN-stage MDE, which is legitimate and is what
  we report here. To say anything about the null itself we use the correct instruments:
    * the CI PRECISION BOUND  -- "the data exclude improvements larger than X"
    * a TOST EQUIVALENCE TEST -- against a pre-specified smallest effect of interest (SESOI)
  The CEILING/HEADROOM argument is untouched by any of this: it is a construct argument (only ~36 of
  461 questions are even improvable on Doc@1), not a power argument, and it survives. Lead with it.

Churn is ESTIMATED FROM THE DATA (per_query_hits.json), not invented as 0.03 as before.
"""
import argparse, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp   # item C10: an artifact must remember the code that wrote it

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def observed_discordance(hits, metric_idx):
    """Real discordance rates (p01, p10) of each variant vs the dense baseline."""
    qids = sorted(hits["dense"])
    out = {}
    for tag in hits:
        if tag == "dense":
            continue
        b = np.array([hits["dense"][q][metric_idx] for q in qids])
        v = np.array([hits[tag][q][metric_idx] for q in qids])
        p01 = float(np.mean((b == 0) & (v == 1)))     # baseline wrong, variant right
        p10 = float(np.mean((b == 1) & (v == 0)))     # baseline right, variant wrong
        out[tag] = (p01, p10)
    return out


def power_sim(n, p_base, delta, churn, n_sim=4000, alpha=0.05, seed=0):
    """Power of a two-sided paired test to detect a true `delta`, with BINOMIAL discordance.

    p10 = churn             (baseline right -> variant wrong)
    p01 = churn + delta     (baseline wrong -> variant right)     => E[delta_hat] = delta
    Ceiling: p01 cannot exceed the improvable mass (1 - p_base).
    """
    p10, p01 = churn, churn + delta
    if p01 > (1 - p_base):
        return 0.0, False                      # not enough improvable items to realise this delta
    rng = np.random.default_rng(seed)
    n01 = rng.binomial(n, p01, size=n_sim)
    n10 = rng.binomial(n, p10, size=n_sim)
    dhat = (n01 - n10) / n
    disc = n01 + n10
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.maximum(disc - n * dhat ** 2, 1e-9)) / n    # McNemar-style paired SE
        z = np.abs(dhat) / np.maximum(se, 1e-12)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.mean(z > zc)), True


def mde_closed_form(n, p_d, alpha=0.05, power=0.80):
    """delta_MDE ≈ (z_{α/2} + z_β) * sqrt(p_d / n), p_d = discordance rate."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(p_d / n))


def tost(a, b, sesoi, n_boot=10000, seed=42):
    """Two one-sided tests for equivalence on paired binary outcomes.

    H0 (non-equivalence): |true delta| >= sesoi.  Rejecting it means the effect is SMALLER than the
    smallest effect we said we would care about -- which is a positive claim about the null, unlike
    post-hoc power. Bootstrap form, to match the paired bootstrap used elsewhere in the paper.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
    p_upper = float(np.mean(means >= sesoi))     # against "delta >= +sesoi"
    p_lower = float(np.mean(means <= -sesoi))    # against "delta <= -sesoi"
    lo, hi = np.percentile(means, [2.5, 97.5])
    return obs, float(lo), float(hi), max(p_upper, p_lower)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default=os.path.join(REPO, "outputs/retrieval_eval/per_query_hits.json"))
    ap.add_argument("--sesoi", type=float, default=0.03,
                    help="smallest effect of interest for TOST (pre-specify)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/power_analysis.json"))
    args = ap.parse_args()

    if not os.path.exists(args.hits):
        sys.exit(f"missing {args.hits} — run retrieval_eval.py first")
    hits = json.load(open(args.hits))
    qids = sorted(hits["dense"])
    n = len(qids)
    report = {"N": n, "sesoi": args.sesoi, "metrics": {}}

    for mi, mn in ((0, "Doc@1"), (1, "Pg@1")):
        base = np.array([hits["dense"][q][mi] for q in qids])
        p_base = float(base.mean())
        n_wrong = int((base == 0).sum())
        disc = observed_discordance(hits, mi)
        churn = float(np.median([p10 for _, p10 in disc.values()]))
        p_d = float(np.median([p01 + p10 for p01, p10 in disc.values()]))

        print(f"\n=== {mn} ===")
        print(f"  baseline = {p_base:.4f} | improvable headroom = {1-p_base:.4f} "
              f"({n_wrong} of {n} questions are wrong)   <- the CONSTRUCT argument, and it survives")
        print(f"  OBSERVED discordance (median over the variant pool): churn p10 = {churn:.4f}, "
              f"p_d = {p_d:.4f}    [previously hard-coded as 0.03]")

        grid = np.arange(0.005, 0.121, 0.0025)
        mde_sim = None
        for d in grid:
            pw, feas = power_sim(n, p_base, float(d), churn)
            if feas and pw >= 0.80:
                mde_sim = float(d)
                break
        mde_cf = mde_closed_form(n, p_d)
        ms = f"{mde_sim:.4f}" if mde_sim else "infeasible"
        print(f"  DESIGN-stage MDE @80% power — binomial sim: {ms} | closed-form McNemar: {mde_cf:.4f}")
        print(f"     (the previous script reported 0.0300 for both metrics; it was measuring the "
              f"~50% power point)")

        print(f"\n  {'variant':<28}{'delta':>9}{'95% CI':>21}{'TOST p':>9}  verdict")
        rows = {}
        for tag in sorted(hits):
            if tag == "dense":
                continue
            v = [hits[tag][q][mi] for q in qids]
            b = [hits["dense"][q][mi] for q in qids]
            obs, lo, hi, p = tost(v, b, args.sesoi)
            if lo > 0 or hi < 0:
                verdict = "DIFFERENT"
            elif p < 0.05:
                verdict = f"EQUIVALENT (|d| < {args.sesoi})"
            else:
                verdict = "inconclusive"
            rows[tag] = {"delta": obs, "ci": [lo, hi], "tost_p": p, "verdict": verdict}
            print(f"  {tag:<28}{obs:>+9.4f} [{lo:>+7.4f},{hi:>+7.4f}]{p:>9.4f}  {verdict}")

        report["metrics"][mn] = {
            "baseline": p_base, "n_wrong": n_wrong, "headroom": 1 - p_base,
            "observed_churn_p10": churn, "observed_discordance_pd": p_d,
            "mde80_sim": mde_sim, "mde80_closed_form": mde_cf, "variants": rows}

    # Declare the input. This file's numbers are a pure function of per_query_hits.json, and on
    # 2026-07-13 that file was regenerated (C10 fix) while this one was not -- so C1's MDE, churn and
    # every TOST verdict were computed from hits that no longer existed. Recording the input's content
    # hash makes that state detectable instead of invisible.
    json.dump(stamp(report, "power_analysis.py", inputs=[args.hits]), open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")
    print("NOTE: no post-hoc power appears anywhere in this file, by design (Hoenig & Heisey 2001). "
          "The MDE is a DESIGN-stage quantity; claims about the observed nulls come from the CI "
          "bounds and the TOST column.")


if __name__ == "__main__":
    main()

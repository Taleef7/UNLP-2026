#!/usr/bin/env python3
"""
Compute all paper-facing analyses from existing benchmark artifacts.
Run from repo root: python3 scripts/paper_analysis/compute_analyses.py
Outputs a paper_analysis_report.md with all findings ready to drop into the paper.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

out_lines: list[str] = []

def h(title: str) -> None:
    out_lines.append(f"\n## {title}\n")

def p(*args) -> None:
    out_lines.append(" ".join(str(a) for a in args))

def load_summary(preset: str) -> dict | None:
    path = REPO / "outputs" / "benchmarks" / preset / "summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_pq(preset: str) -> pd.DataFrame | None:
    path = REPO / "outputs" / "benchmarks" / preset / "per_question.csv"
    if path.exists():
        df = pd.read_csv(path).sort_values("Question_ID").reset_index(drop=True)
        df["composite"] = 0.5*df["answer_correct"] + 0.25*df["doc_correct"] + 0.25*df["page_proximity"]
        return df
    return None

def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = (a.values - b.values)
    d = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi, diff.mean()

# ── 1. Verify key numbers ────────────────────────────────────────────────────
h("1. Key System Numbers (v7 full-dev)")
s_v7 = load_summary("v7_full_dev")
if s_v7:
    p(f"answer_accuracy : {s_v7['answer_accuracy']:.4f}")
    p(f"doc_accuracy    : {s_v7['doc_accuracy']:.4f}")
    p(f"page_proximity  : {s_v7['page_proximity']:.4f}")
    p(f"composite_score : {s_v7['composite_score']:.4f}")
    p(f"n_questions     : {s_v7['n_questions']}")
    p("\nPer-domain:")
    for dom, dm in s_v7.get("per_domain", {}).items():
        p(f"  {dom}: composite={dm['composite']:.4f}  answer={dm['answer_acc']:.4f}  doc={dm['doc_acc']:.4f}  page={dm['page_prox']:.4f}")

# ── 2. Closed-book baseline ──────────────────────────────────────────────────
h("2. Closed-Book Baseline")
s_cb = load_summary("paper_closed_book")
if s_cb and s_v7:
    cb_ans = s_cb["answer_accuracy"]
    v7_ans = s_v7["answer_accuracy"]
    p(f"Closed-book answer_accuracy : {cb_ans:.4f}")
    p(f"v7 answer_accuracy          : {v7_ans:.4f}")
    p(f"Retrieval contribution      : +{v7_ans - cb_ans:.4f} pp ({v7_ans - cb_ans:.1%} gain)")
    p(f"Closed-book composite       : {s_cb['composite_score']:.4f}")
    p(f"Note: composite for closed-book not directly comparable (no doc/page signal)")
else:
    p("paper_closed_book NOT FOUND — run scripts/paper_analysis/run_gpu_experiments.sh first")

# ── 3. Context-size sensitivity ──────────────────────────────────────────────
h("3. Context-Size Sensitivity (k pages, 100 questions, seed=42)")
p(f"{'k':>3} | {'answer':>8} | {'composite':>10} | {'delta_vs_2':>12}")
p("-" * 45)
ans_k2 = None
for k in [0, 1, 2, 3, 5]:
    preset = "paper_closed_book" if k == 0 else f"paper_ctx_k{k}_n100"
    s = load_summary(preset)
    if s:
        ans = s["answer_accuracy"]
        comp = s["composite_score"]
        if k == 2:
            ans_k2 = ans
        delta = f"{ans - ans_k2:+.4f}" if ans_k2 is not None and k != 2 else ("baseline" if k == 2 else "n/a")
        label = "0 (closed-book)" if k == 0 else str(k)
        p(f"{label:>16} | {ans:8.4f} | {comp:10.4f} | {delta:>12}")
    else:
        p(f"{k:>16} | MISSING")

# ── 4. Page distance distribution ───────────────────────────────────────────
h("4. Page Distance Distribution (wrong-page cases)")
df_v7 = load_pq("v7_full_dev")
gt    = pd.read_csv(REPO / "data" / "dev_questions.csv")
if df_v7 is not None:
    df = df_v7.merge(gt[["Question_ID","N_Pages"]], on="Question_ID", how="left")
    wrong_page = df[(df["doc_correct"] == 1.0) & (df["page_proximity"] < 1.0)].copy()
    wrong_page["page_dist"] = (wrong_page["pred_page_num"] - wrong_page["true_page_num"]).abs()
    n = len(wrong_page)
    p(f"Total wrong-page cases (correct doc, wrong page): {n}")
    for thresh in [1, 2, 3, 5, 10]:
        cnt = (wrong_page["page_dist"] <= thresh).sum()
        p(f"  ≤ {thresh:2d} pages : {cnt:3d} / {n} = {cnt/n:.1%}")
    p(f"  > 10 pages : {(wrong_page['page_dist'] > 10).sum():3d} / {n} = {(wrong_page['page_dist'] > 10).mean():.1%}")
    p(f"\n  median dist : {wrong_page['page_dist'].median():.1f}")
    p(f"  mean dist   : {wrong_page['page_dist'].mean():.1f}")
    p("\nPer-domain:")
    for dom in ["domain_1","domain_2"]:
        d = wrong_page[wrong_page["Domain"]==dom]
        w3 = (d["page_dist"] <= 3).mean()
        p(f"  {dom} (n={len(d)}): median={d['page_dist'].median():.1f}  within-3={w3:.1%}")

# ── 5. Voting consistency ────────────────────────────────────────────────────
h("5. Voting Consistency (3-pass MamayLM)")
if df_v7 is not None:
    df = df_v7.copy()
    df["votes_list"] = df["votes"].apply(lambda x: x.split("|") if pd.notna(x) and str(x).strip() else [])
    df["n_unique"] = df["votes_list"].apply(lambda x: len(set(x)) if x else 0)
    for label, mask in [
        ("unanimous 3/3", df["n_unique"] == 1),
        ("majority   2/1", df["n_unique"] == 2),
        ("all-diff      ", df["n_unique"] == 3),
    ]:
        n_m = mask.sum()
        acc = df[mask]["answer_correct"].mean() if n_m > 0 else float("nan")
        p(f"  {label}: {n_m:4d} ({n_m/len(df):.1%})  answer_acc={acc:.4f}")
    un_acc  = df[df["n_unique"]==1]["answer_correct"].mean()
    maj_acc = df[df["n_unique"]==2]["answer_correct"].mean()
    p(f"\n  Delta (unanimous − majority): {un_acc - maj_acc:+.4f} pp")
    p(f"  Interpretation: unanimous 3/3 questions are {un_acc-maj_acc:.1%} more likely to be correct")

# ── 6. Reranking lift ────────────────────────────────────────────────────────
h("6. Pipeline Reranking Lift (dense → reranked)")
if s_v7:
    ir = s_v7.get("ir_metrics", {})
    stages = ir.get("stages", {})
    rows = []
    for stage_name in ["dense_doc","final_doc","dense_page","final_page"]:
        item = stages.get(stage_name, {})
        ov = item.get("overall", item)
        r1  = ov.get("recall_at_1", float("nan"))
        mrr = ov.get("mrr", float("nan"))
        rows.append((stage_name, r1, mrr))
    p(f"{'Stage':<14} {'recall@1':>10} {'MRR':>8}")
    p("-" * 36)
    for name, r1, mrr in rows:
        p(f"{name:<14} {r1:10.4f} {mrr:8.4f}")
    dense_p = stages.get("dense_page", {}).get("overall", stages.get("dense_page",{})).get("recall_at_1", float("nan"))
    final_p = stages.get("final_page", {}).get("overall", stages.get("final_page",{})).get("recall_at_1", float("nan"))
    p(f"\n  Page reranking lift: {dense_p:.4f} → {final_p:.4f}  (+{final_p-dense_p:.4f} pp recall@1)")

# ── 7. Fair 4B per-domain breakdown ─────────────────────────────────────────
h("7. Fair 4B Reranker — Per-Domain Breakdown")
s_4b = load_summary("candidate_stronger_reranker_v2_fair_full_dev")
if s_4b and s_v7:
    p(f"{'System':<12} {'Domain':<10} {'composite':>10} {'answer':>8} {'doc':>8} {'page':>8}")
    p("-" * 60)
    for preset_name, s in [("v7", s_v7), ("fair_4B", s_4b)]:
        for dom, dm in s.get("per_domain", {}).items():
            p(f"{preset_name:<12} {dom:<10} {dm['composite']:10.4f} {dm['answer_acc']:8.4f} {dm['doc_acc']:8.4f} {dm['page_prox']:8.4f}")
    p("\n  Domain deltas (fair_4B − v7):")
    for dom in s_v7.get("per_domain", {}).keys():
        v7_c = s_v7["per_domain"][dom]["composite"]
        fb_c = s_4b["per_domain"][dom]["composite"]
        p(f"  {dom}: {fb_c - v7_c:+.4f}")

# ── 8. Paired bootstrap CIs ──────────────────────────────────────────────────
h("8. Paired Bootstrap CIs (10k iterations, seed=42)")
df_v7 = load_pq("v7_full_dev")
df_4b = load_pq("candidate_stronger_reranker_v2_fair_full_dev")
df_cb = load_pq("paper_closed_book")

if df_v7 is not None and df_4b is not None:
    lo, hi, mu = paired_bootstrap(df_4b["composite"], df_v7["composite"])
    p(f"Fair 4B vs v7 composite: delta={mu:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
    for col in ["answer_correct","doc_correct","page_proximity"]:
        lo2,hi2,mu2 = paired_bootstrap(df_4b[col], df_v7[col])
        sig = "✓" if lo2 > 0 else ("✗" if hi2 < 0 else "~")
        p(f"  {col:<20}: delta={mu2:+.4f}  CI=[{lo2:+.4f}, {hi2:+.4f}]  {sig}")

if df_v7 is not None and df_cb is not None:
    lo, hi, mu = paired_bootstrap(df_v7["composite"], df_cb["composite"])
    p(f"\nv7 vs closed-book composite: delta={mu:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
    for col in ["answer_correct"]:
        lo2,hi2,mu2 = paired_bootstrap(df_v7[col], df_cb[col])
        p(f"  {col}: delta={mu2:+.4f}  CI=[{lo2:+.4f}, {hi2:+.4f}]")

# ── Write report ─────────────────────────────────────────────────────────────
report_path = REPO / "docs" / "paper" / "paper_analysis_report.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    f.write("# Paper Analysis Report\n")
    f.write("_Generated by scripts/paper_analysis/compute_analyses.py_\n")
    f.write("\n".join(out_lines))

print("\n".join(out_lines))
print(f"\n\nReport written to: {report_path}")

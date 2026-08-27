#!/usr/bin/env python3
"""Generate the paper's evidence figure ENTIRELY from raw artifacts (item F1 for figures).

Every value plotted is read from `outputs/e2e_stop_bug.json`; nothing is typed by hand. Output:
`paper/fig_bugs.pdf`, a compact two-panel figure sized for one ACL column.
- Panel (a): the stop-sequence bug's footprint (empty generations, shipped vs fixed).
- Panel (b): the voting bug's falsifiable signature (which of the three passes disagree), plus the
  true-reset control.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper/fig_bugs.pdf")

e = json.load(open(os.path.join(REPO, "outputs/e2e_stop_bug.json")))
cells = e["check4_fix_reaches_mechanism"]["cells"]
fb = e["fallback_counterfactual"]["e2e_b3"]
c8 = e["check8_warm_cache_signature"]["per_cell"]
reset = e["check10_prereg_reset"]["hard_P1_all_identical"]

# --- panel (a): stop-sequence bug ---
empty_shipped = cells["e2e_b3"]["empty"]          # 36
empty_fixed = cells["e2e_fix"]["empty"]           # 0
blind = fb["questions_decided_by_a_fallback"]     # 13
lucky = fb["of_those_right_by_luck"]              # 2

# --- panel (b): voting bug ---
c2v3 = sum(v["c2_vs_c3_disagree"] for v in c8.values())        # 0 of 1844
c1diff = sum(v["c1_differs_from_c2"] for v in c8.values())     # 16 of 1844
n_trip = sum(v["n_questions"] for v in c8.values())            # 1844
reset_differ = reset["n"] - reset["identical"]                 # 0 of 461
reset_n = reset["n"]                                           # 461

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.3, 1.7))

# Panel (a)
axa.bar([0, 1], [empty_shipped, empty_fixed], color=["#c0392b", "#7f8c8d"], width=0.6)
axa.set_xticks([0, 1])
axa.set_xticklabels(["shipped", "fixed"])
axa.set_ylabel("empty generations")
axa.set_ylim(0, empty_shipped * 1.25 + 1)
axa.text(0, empty_shipped + 1, str(empty_shipped), ha="center", va="bottom", fontsize=7)
axa.text(1, empty_fixed + 1, str(empty_fixed), ha="center", va="bottom", fontsize=7)
axa.annotate(f"{blind} blind guesses\n({lucky} banked)", xy=(0, empty_shipped), xytext=(0.5, empty_shipped * 0.72),
             fontsize=6, ha="left", va="center")
axa.set_title("(a) stop bug", fontsize=7)

# Panel (b)
labels = ["calls\n2 vs 3", "call 1\nvs 2--3", "any\n(reset)"]
vals = [c2v3, c1diff, reset_differ]
denom = [n_trip, n_trip, reset_n]
axb.bar(range(3), vals, color=["#7f8c8d", "#c0392b", "#7f8c8d"], width=0.6)
axb.set_xticks(range(3))
axb.set_xticklabels(labels, fontsize=6)
axb.set_ylabel("passes that disagree")
axb.set_ylim(0, max(vals) * 1.3 + 1)
for i, (v, d) in enumerate(zip(vals, denom)):
    axb.text(i, v + max(vals) * 0.04 + 0.15, f"{v}/{d}", ha="center", va="bottom", fontsize=6)
axb.set_title("(b) voting bug", fontsize=7)

fig.tight_layout(pad=0.4)
# Strip the creation timestamp so the PDF is reproducible run-to-run.
fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
print(f"[gen_paper_figures] wrote {OUT}  "
      f"(a: {empty_shipped}->{empty_fixed} empties, {blind}/{lucky}; "
      f"b: c2v3={c2v3}/{n_trip}, c1diff={c1diff}, reset_differ={reset_differ}/{reset_n})")

# Job Outcomes on March 30, 2026

## Overview

This note records the completed outcome of the queued paper-ablation Slurm chain:

- `10491629` `paper-dense-margin`
- `10491657` `paper-fair-combo`
- `10491658` `paper-8b-feas`
- `10491663` `paper-audit-refresh`

## 10491629 `paper-dense-margin`

Status: completed successfully in `00:46:58`.

Result:

- `candidate_dense_margin_lock_v1` matched `v7_baseline` exactly on `fold0`
- `candidate_dense_margin_lock_v1` matched `v7_baseline` exactly on `fold1`
- `candidate_dense_margin_lock_v1` matched `v7_baseline` exactly on `lockbox`
- `candidate_dense_margin_lock_v1` regressed slightly on `full_dev`

Key full-dev diff versus `v7_baseline`:

- `avg_score_delta = -0.0023069568907596203`
- `top_1_doc_flip_rate = 0.06290672451193059`
- `page_change_rate = 0.15835140997830802`
- `answer_change_rate = 0.049891540130151846`

Interpretation:

- The margin-lock idea is not a paper promotion candidate.
- It is better reported as a neutral-to-slightly-negative ablation than as an incomplete promising lane.
- The split pattern suggests the dense-margin lock is mostly redundant with the baseline behavior on the public validation ladder, then mildly harmful when aggregated over the broader dev set.

## 10491657 `paper-fair-combo`

Status: timed out at `00:30:17`.

Completed artifacts before timeout:

- `candidate_v5refocus_reranker4b_fair_fold0`
- `candidate_v5refocus_reranker4b_fair_fold1`

Missing because of timeout:

- `candidate_v5refocus_reranker4b_fair_lockbox`
- all `candidate_dense_doc_lock_v3_reranker4b_fair_*`
- all `candidate_v5refocus_ddl_v3_reranker4b_fair_*`

Observed results:

- On `fold0`, `candidate_v5refocus_reranker4b_fair` improved over `v7_baseline` by `+0.017515475606954593`
- On `fold1`, `candidate_v5refocus_reranker4b_fair` improved over `v7_baseline` by `+0.058716882246294004`
- On `fold0`, it was identical to the fair 4B anchor `candidate_stronger_reranker_v2_fair`
- On `fold1`, it improved over the fair 4B anchor by `+0.01640271493212697`

Interpretation:

- The combined `v5_refocus + fair 4B reranker` lane remains promising.
- The evidence is incomplete because the lockbox gate did not finish.
- No conclusion should be drawn yet about `dense_doc_lock_v3 + fair 4B` or the triple combo because those stages never started.

## 10491658 `paper-8b-feas`

Status: completed in `00:09:43`.

Attempts:

1. Strict `candidate_stronger_reranker_v3_8b_feasibility`
2. Batch-size-2 retry of the same feasibility preset

Observed behavior:

- In both attempts, `qwen3_8b` loaded successfully
- In both attempts, page-level reranking completed successfully
- In both attempts, the pipeline failed immediately after reranking when loading MamayLM GGUF
- The final error was `RuntimeError: Could not load MamayLM GGUF at any configured n_ctx`

Interpretation:

- `qwen3_8b` is `rerank-feasible` on the A30 single-GPU lane
- `qwen3_8b` is not `pipeline-feasible` in the current architecture because MamayLM cannot then be loaded end to end
- This should be reported as a feasibility-only finding, not as a fair ablation result

## 10491663 `paper-audit-refresh`

Status: completed successfully in `00:00:28`.

Actions completed:

- backfilled `paper_notes.md` for finished paper-ablation run directories
- rebuilt:
  - `docs/paper/ablations/audit_summary.md`
  - `docs/paper/ablations/methodology_audit.md`
  - `docs/paper/ablations/system_lineage.md`
  - `docs/paper/tables/master_ablation_table.md`
  - `docs/paper/tables/public_vs_offline_dual_anchor.md`
  - `docs/paper/tables/negative_results_table.md`
  - `docs/paper/tables/master_ablation_ledger.csv`
  - `docs/paper/tables/master_ablation_ledger.json`

## Paper-Facing Takeaway

The strongest trustworthy story after this run chain is:

- public anchor remains `v7`
- strongest fair offline anchor remains `candidate_stronger_reranker_v2_fair`
- `candidate_dense_margin_lock_v1` is a negative or at best neutral ablation
- `v5_refocus + fair 4B reranker` is promising but still incomplete because the gate timed out before lockbox
- `qwen3_8b` should be written up as `rerank-feasible but not pipeline-feasible`

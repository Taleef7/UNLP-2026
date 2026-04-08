# Job Outcomes on March 31, 2026

## Overview

This note records the March 31 paper-completion work after the competition ended and the repo moved from deadline-mode to paper-mode.

## 1. Full-Dev Split Correction

The round-2 paper runner briefly pointed `full_dev` at `data/splits/val.csv`, which contains only the small validation subset.

The correction is:

- `full_dev` must use `data/dev_questions.csv`
- the trusted comparison baseline remains `outputs/benchmarks/v7_full_dev`
- paper-facing local-vs-Kaggle analysis must use the 461-question full-dev source

Why this matters:

- `v7_full_dev` is a 461-question artifact
- comparing new paper reruns against it with an 85-question subset would have been methodologically invalid

The runner was patched before accepting any new round-2 result.

## 2. Corrected `v13` Full-Dev Result

The exact `candidate_v5_refocus_v1` submission family was rerun on `data/dev_questions.csv` and diffed against `v7_full_dev`.

Result:

- full-dev composite: `0.8637336984188643`
- delta vs `v7_full_dev`: `+0.00032675513017636147`
- answer accuracy: `0.89587852494577`
- doc accuracy: `0.8828633405639913`
- page proximity: `0.780314403219924`

Diff versus `v7_full_dev`:

- `avg_score_delta = +0.00032675513017636147`
- `top_1_doc_flip_rate = 0.06290672451193059`
- `page_change_rate = 0.1648590021691974`
- `answer_change_rate = 0.0824295010845987`
- per-domain delta:
  - `domain_1 = -0.015546285954214594`
  - `domain_2 = +0.012276801269755867`

Interpretation:

- `v13` is not a strong full-dev local winner.
- It is effectively a local tie with `v7`, with a bootstrap interval that crosses zero.
- That makes the final Kaggle regression easier to explain honestly:
  - the prompt change produced small, distribution-sensitive movement
  - it did not establish a robust offline win before submission

## 3. Round-2 Fair-4B Matrix

The clean round-2 fair-4B matrix was launched under:

- hardware lane: interactive `a30` standby allocation
- output root: `outputs/paper_ablation_runs_r2/2026-03-31_a30/`
- command family:
  - `candidate_v5refocus_reranker4b_fair`
  - `candidate_dense_doc_lock_v3_reranker4b_fair`
  - `candidate_v5refocus_ddl_v3_reranker4b_fair`

Execution policy:

- one-question smoke before each family
- `--skip-existing` so interrupted work can resume without overwriting finished artifacts
- same A30 lane for all paper-facing reruns

Status at the time of writing:

- the smoke run completed successfully
- `candidate_v5refocus_reranker4b_fair_fold0` completed and reproduced the earlier March 30 delta exactly:
  - `avg_score_delta = +0.017515475606954593`
- `candidate_v5refocus_reranker4b_fair_fold1` completed and reproduced the earlier March 30 delta exactly:
  - `avg_score_delta = +0.058716882246294004`
- `candidate_v5refocus_reranker4b_fair_lockbox` is now running
- the remaining pairwise and triple families will continue under the same resume-safe round-2 root

## Paper-Facing Takeaway

The most important March 31 conclusion so far is methodological, not just numerical:

- the corrected `v13` full-dev evidence shows that `v13` was never a decisive offline win over `v7`
- therefore the paper should frame `v13` as a local prompt-variation signal that failed to generalize, not as a clearly superior system that was unexpectedly punished by Kaggle

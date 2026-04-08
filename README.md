# UNLP 2026 — Multi-Domain Ukrainian Document Understanding

**Task:** Retrieval-augmented QA over Ukrainian PDFs (MCQ, 6 options A-F)  
**Metric:** `0.5 * answer_acc + 0.25 * doc_acc + 0.25 * page_prox`  
**Deadline:** March 30, 2026 (Kaggle) | April 8, 2026 (paper)

For a new agent joining the repo, read [AGENT_HANDOFF.md](/scratch/gilbreth/tamst01/unlp2026/AGENT_HANDOFF.md) immediately after this README.

## Current Reality

| Version | Local / Historical Signal | Kaggle Score | Status |
|---------|---------------------------|--------------|--------|
| v6 | 0.8618-era baseline | 0.8618 | Historical |
| v7 | best scored recovery baseline | **0.8688** | Trusted rollback |
| v8 | quality branch, timed out | — | Historical |
| v9 | regression case | **0.6716** on March 26, 2026 | Do not promote |
| candidate_structure_chunks_v1 | full-dev local improvement over `v7` | — | Historical local winner |
| candidate_structure_chunks_v2_doc_guard | public T4x2 run underperformed | **0.8415** on March 28, 2026 | Not promotable |
| candidate_no_bleed_v1 | fold-0 win, but failed fold-1 and lockbox | — | Rejected by robustness gate |
| `candidate_locked_top_doc_v1` | tiny but stable local page-only gains | — | Near-miss, not promotable |

The repo is now in **v7-first robustness mode**.

- `v7` is the trusted submission baseline until a new candidate proves better.
- `v9` is treated as a reproduced regression, not as the active source of truth.
- No new Kaggle submission should be pushed until a candidate:
  - beats the reproduced `v7` local composite on full dev
  - survives grouped CV and lockbox checks
  - projects safely under Kaggle's 9-hour T4x2 limit

## First Local Benchmark Evidence

The shared benchmark path has now been validated on an interactive A30 node, and the first controlled sample already matches the Kaggle direction:

| Preset | Sample | Composite | Answer | Doc | Page | Rerank Time |
|--------|--------|-----------|--------|-----|------|-------------|
| `v7_baseline` | 10 dev questions | **0.976991** | 1.0000 | 1.0000 | 0.9080 | 14.9s |
| `v9_regression` | 10 dev questions | **0.792424** | 0.9000 | 0.7000 | 0.6697 | 84.5s |

Question-level diff on that sample:

- average score delta: `-0.184567`
- regressed questions: `4`
- improved questions: `1`
- document changes: `3`
- page changes: `5`
- answer changes: `1`

Current inference: `v9` is losing mainly through retrieval / routing instability and extra rerank overhead, not because the answer model suddenly got better or worse in isolation.

## Recovery Workflow

The current workflow is built around one shared benchmark path:

- Shared pipeline core: `notebooks/pipeline_shared.py`
- Named presets: `notebooks/pipeline_presets.json`
- Local benchmark CLI: `scripts/benchmark_candidate.py`
- Benchmark diff CLI: `scripts/diff_benchmark_runs.py`
- Exact metric harness: `scripts/00_eval_harness.py`

Use that path instead of reasoning from historical notebook copies.

## Current Local Leaderboard

The first full-dev `v7`-preserving win is now on record:

| Preset | Questions | Composite | Answer | Doc | Page | Status |
|--------|-----------|-----------|--------|-----|------|--------|
| `v7_baseline` | 461 | **0.863407** | 0.8850 | 0.8894 | 0.7942 | Full-dev rollback baseline |
| `v9_regression` | 461 | **0.743363** | 0.8134 | 0.7115 | 0.6351 | Confirmed regression |
| `candidate_structure_chunks_v1` | 50 | **0.886044** | 0.9400 | 0.8800 | 0.7842 | Passed medium gate |
| `candidate_structure_chunks_v1` | 461 | **0.867503** | 0.8915 | 0.8915 | 0.7954 | Historical local winner |
| `candidate_structure_chunks_v2_doc_guard` | 50 | **0.916044** | 0.9400 | 0.9400 | 0.8442 | Historical medium-gate win |
| `candidate_structure_chunks_v2_doc_guard` | 461 | **0.881726** | 0.8915 | 0.9219 | 0.8219 | Historical local winner only |
| `candidate_avir_threshold_v1` | 50 | **0.870726** | 0.9000 | 0.8800 | 0.8029 | Passed medium gate |
| `candidate_avir_threshold_v1` | 461 | **0.866343** | 0.8850 | 0.8894 | 0.8059 | Secondary full-dev improvement |
| `candidate_no_bleed_v1` | fold 0 | **0.857678** | 0.8767 | 0.8904 | 0.7869 | False-positive grouped-CV win |
| `candidate_no_bleed_v1` | fold 1 | **0.862212** | 0.9103 | 0.8590 | 0.7694 | Lost to `v7_baseline` |
| `candidate_no_bleed_v1` | lockbox | **0.857546** | 0.8632 | 0.9053 | 0.7986 | Failed promotion gate |
| `candidate_locked_top_doc_v1` | fold 0 | **0.832011** | 0.8493 | 0.8767 | 0.7527 | Slight stable page-only win |
| `candidate_locked_top_doc_v1` | fold 1 | **0.872335** | 0.9359 | 0.8333 | 0.7842 | Slight stable page-only win |
| `candidate_locked_top_doc_v1` | lockbox | **0.887354** | 0.8947 | 0.9368 | 0.8231 | Near-tie, not enough to promote |

What changed after the public `0.8415` result:

- grouped CV and lockbox now outrank historical full-dev-only wins
- smoke parity is now enforced between the shared runtime and the rendered Kaggle bundle
- `candidate_structure_chunks_v2_doc_guard` did not transfer robustly
- `candidate_no_bleed_v1` produced a tempting fold-0 gain, then failed fold 1 and lockbox
- `candidate_locked_top_doc_v1` is the closest near-miss: very stable, but too small to justify promotion
- `v7_baseline` is still the strongest promotion control under the new ladder

Current interpretation:

- local full-dev alone is not a sufficient promotion gate
- the main failure mode is robustness under split / domain shift, not bundle drift
- `v7_baseline` remains the reference candidate to beat

See [logs/2026-03-28_robustness_recovery.md](/scratch/gilbreth/tamst01/unlp2026/logs/2026-03-28_robustness_recovery.md) for the current robustness ledger.

## Default Presets

- `v7_baseline`
  - scored rollback baseline
  - page-level rerank
  - answer-aware page selection within the chosen doc
- `v8_quality_branch`
  - quality-first historical branch
  - precomputed segments
  - broader context and always-on page stage 2
- `v9_regression`
  - regression case for diagnosis
  - doc-level rerank
  - lazy segments
  - conditional voting
  - runtime governor
- candidate presets
  - `candidate_lazy_segments_v1`
  - `candidate_conditional_voting_v1`
  - `candidate_runtime_guard_v1`
  - `candidate_structure_chunks_v1`
  - `candidate_structure_chunks_v2_doc_guard`
  - `candidate_page_summary_selector_v1`
  - `candidate_avir_threshold_v1`
  - `candidate_confidence_gated_v1`

## Commands

Reproduce the scored baseline locally:

```bash
python scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --questions data/dev_questions.csv \
  --output-dir outputs/benchmarks/v7_baseline
```

Reproduce the regression locally:

```bash
python scripts/benchmark_candidate.py \
  --preset v9_regression \
  --questions data/dev_questions.csv \
  --output-dir outputs/benchmarks/v9_regression
```

Compare them question-by-question:

```bash
python scripts/diff_benchmark_runs.py \
  --base-dir outputs/benchmarks/v7_baseline \
  --candidate-dir outputs/benchmarks/v9_regression
```

Compatibility wrapper for the rollback baseline:

```bash
python notebooks/v7_local_eval.py --n-questions 10
```

## Submission Policy

`notebooks/kernel-metadata.json` should track the current promotion default.

Current default:

- kernel: `taleeftamsal/unlp-2026-v7-kaggle-wrapper`
- wrapper: `notebooks/v7_kaggle_submission.py`

Historical note:

- `notebooks/v10_kaggle_submission.py` is preserved as the standalone Kaggle wrapper for the historical doc-guard candidate
- it is **not** the current promotion default

A new candidate should only replace the `v7` default after all of the following are true:

1. Full-dev local composite is greater than reproduced `v7_baseline`
2. Runtime projection stays below the 9-hour Kaggle budget with margin
3. Submission CSV validates cleanly
4. Docs are updated to describe the promoted candidate accurately

## Environment Notes

Current competition target:

- Kaggle submission runtime: T4 x 2
- Hard wall-clock limit: 9 hours

Historical P100 timing notes still exist in older logs and scripts, but they are **stale when they conflict with the current T4x2 submission environment**.

## Directory Layout

```text
notebooks/
  pipeline_shared.py        <- shared runtime for local benchmark and Kaggle wrappers
  pipeline_presets.json     <- named benchmark / submission presets
  v7_local_eval.py          <- thin local wrapper for the v7 baseline
  v7_kaggle_submission.py   <- thin Kaggle wrapper for the rollback baseline
  v8_kaggle_submission.py   <- thin Kaggle wrapper for the historical quality branch
  v9_kaggle_submission.py   <- thin Kaggle wrapper for the regression case
  v10_kaggle_submission.py  <- thin Kaggle wrapper for the historical doc-guard candidate
  kernel-metadata.json      <- should track the current promotion default

scripts/
  benchmark_candidate.py    <- main local benchmark runner
  diff_benchmark_runs.py    <- compare two benchmark runs
  00_eval_harness.py        <- exact metric implementation

outputs/benchmarks/
  <preset>/
    predictions.csv
    per_question.csv
    summary.json
    timings.json
    submission.csv
```

## Bottom Line

The repo is no longer optimizing around "make v9 finish." It is optimizing around:

1. reproduce `v7`
2. explain `v9`
3. promote only isolated changes that beat `v7` locally and stay inside the Kaggle budget

Current state:

- `v7_baseline` is still the trusted promotion leader
- the doc-guard and no-bleed branches have been demoted by the robustness gate
- `candidate_locked_top_doc_v1` is the cleanest near-miss, but still not enough to justify a new submission

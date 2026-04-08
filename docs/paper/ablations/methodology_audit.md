# Methodology Audit

## Critical Fairness and Interpretation Flags

- `candidate_stronger_reranker_v1` is historical and confounded. It used `qwen3_4b` with rerank `max_length=1024`, while `v7_baseline` used `2048`, so it must not be cited as evidence that `0.6B` beats `4B`.
- `candidate_stronger_reranker_v2_fair` is the correct fair reranker comparison because it keeps the `v7_baseline` rerank budget and context length.
- `v5refocus_*`, `ctx5_*`, and `dual_ctx_*` were originally recorded under `v7_baseline` with `override_json`. They are reproducible and traceable, but they are override experiments rather than original named presets.
- `candidate_dense_margin_lock_v1` had only fold-1 evidence before the paper audit and is now reclassified as neutral-to-negative after the completed ladder.

## Stale Documentation Findings

- `UNLP 2026 Shared Task: A Complete Technical Research Guide for Multi-Domain Ukrainian Document Understanding.md` describes the score as a 50/50 answer-vs-reference split in prose. The official metric and the repo's evaluation code use `0.5 * answer + 0.25 * doc + 0.25 * page_proximity`.
- The same guide refers to a train/test domain structure that is stale against the official task repo. The official README says `dev` and `test_public` have two domains, while `test_private` has three domains, with an unseen extra domain in private evaluation.
- The old `plan.md` describes the 4B reranker as a catastrophic failure. That statement is stale once the fair 4B reranker retest is treated as the authoritative comparison.

## Reporting Rule

Only manifest-backed or public-Kaggle-backed evidence should be treated as trusted in the paper. Historical notes without a fair comparison or a run manifest should be labeled as historical context only.

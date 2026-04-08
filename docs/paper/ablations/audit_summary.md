# Ablation Audit Summary

## Trusted Anchors

- `v7` is the competition anchor because it is the strongest real submission, not just the strongest local run.
- `candidate_stronger_reranker_v2_fair` is the offline anchor because it improves on `v7` under a fair reranker budget and context length.
- `candidate_dense_margin_lock_v1` is now a completed negative or neutral lane rather than an unfinished hypothesis.
- `qwen3_8b` remains feasibility-only until the explicit teardown diagnostic proves end-to-end compatibility.

## Retrieval and Doc-Stability

| System | Split | Evidence | Composite | Delta vs v7 | Trust |
| --- | --- | --- | --- | --- | --- |
| _smoke_a30_dense_margin_fold0 | fold0 | offline_named_preset | 0.9722 | 0.0000 | trusted_offline |
| candidate_ddl_v1_fold0 | fold0 | offline_named_preset | 0.8183 | -0.0126 | trusted_offline |
| candidate_ddl_v2_fold0 | fold0 | offline_named_preset | 0.8502 | 0.0193 | trusted_offline |
| candidate_ddl_v2_fold1 | fold1 | offline_named_preset | 0.8983 | 0.0265 | trusted_offline |
| candidate_ddl_v2_fulldev | full_dev | offline_named_preset | 0.8811 | 0.0177 | trusted_offline |
| candidate_ddl_v2_lockbox | lockbox | offline_named_preset | 0.8880 | 0.0003 | trusted_offline |
| candidate_ddl_v3_fold0 | fold0 | offline_named_preset | 0.8502 | 0.0193 | trusted_offline |
| candidate_ddl_v3_fold1 | fold1 | offline_named_preset | 0.9025 | 0.0307 | trusted_offline |
| candidate_ddl_v3_fulldev | full_dev | offline_named_preset | 0.8812 | 0.0178 | trusted_offline |
| candidate_ddl_v3_lockbox | lockbox | offline_named_preset | 0.8929 | 0.0053 | trusted_offline |
| candidate_dense_margin_lock_v1_fold0 | fold0 | offline_named_preset | 0.8309 | 0.0000 | trusted_offline |
| candidate_dense_margin_lock_v1_fold1 | fold1 | offline_named_preset | 0.8718 | 0.0000 | trusted_offline |
| candidate_dense_margin_lock_v1_fold1 | fold1 | offline_named_preset | 0.8718 | 0.0000 | trusted_offline |
| candidate_dense_margin_lock_v1_full_dev | full_dev | offline_named_preset | 0.8611 | -0.0023 | trusted_offline |
| candidate_dense_margin_lock_v1_lockbox | lockbox | offline_named_preset | 0.8877 | 0.0000 | trusted_offline |
| candidate_dense_margin_lock_v1_thresh020_fold1 | fold1 | offline_override | 0.8718 | 0.0000 | traceable_override |
| candidate_dense_margin_lock_v1_thresh030_fold1 | fold1 | offline_override | 0.8718 | 0.0000 | traceable_override |
| candidate_extraction_blocks_v1_fold0 | fold0 | offline_named_preset | 0.8175 | -0.0134 | trusted_offline |
| candidate_hybrid_rrf_v1_fold0 | fold0 | offline_named_preset | 0.8210 | -0.0098 | trusted_offline |
| candidate_locked_top_doc_v1_50 | 50q | offline_named_preset | 0.8673 |  | trusted_offline |
| candidate_locked_top_doc_v1_fold0 | fold0 | offline_named_preset | 0.8320 | 0.0011 | trusted_offline |
| candidate_locked_top_doc_v1_fold1 | fold1 | offline_named_preset | 0.8723 | 0.0005 | trusted_offline |
| candidate_locked_top_doc_v1_lockbox | lockbox | offline_named_preset | 0.8874 | -0.0003 | trusted_offline |
| candidate_sc_v3_fold0 | fold0 | offline_named_preset | 0.8012 | -0.0297 | trusted_offline |
| candidate_sc_v3_fold0_r2 | fold0 | offline_named_preset | 0.8195 | -0.0114 | trusted_offline |
| candidate_sc_v3_fold0_r3 | fold0 | offline_named_preset | 0.8309 | 0.0000 | trusted_offline |
| candidate_sc_v3_fold1 | fold1 | offline_named_preset | 0.8718 | 0.0000 | trusted_offline |
| candidate_sc_v3_lockbox | lockbox | offline_named_preset | 0.8827 | -0.0049 | trusted_offline |
| candidate_sc_v3_smoke | fold0 | offline_named_preset | 0.8258 | -0.0417 | trusted_offline |
| candidate_structure_chunks_v1_10 | 10q | offline_named_preset | 0.9770 |  | trusted_offline |
| candidate_structure_chunks_v1_50 | 50q | offline_named_preset | 0.8860 |  | trusted_offline |
| candidate_structure_chunks_v1_fold0 | fold0 | offline_named_preset | 0.8149 | -0.0160 | trusted_offline |
| candidate_structure_chunks_v1_full | full_dev | offline_named_preset | 0.8675 | 0.0041 | trusted_offline |
| candidate_structure_chunks_v2_doc_guard_10 | 10q | offline_named_preset | 0.9770 |  | trusted_offline |
| candidate_structure_chunks_v2_doc_guard_50 | 50q | offline_named_preset | 0.9160 |  | trusted_offline |
| candidate_structure_chunks_v2_doc_guard_fold0 | fold0 | offline_named_preset | 0.8210 | -0.0099 | trusted_offline |
| candidate_structure_chunks_v2_doc_guard_full | full_dev | offline_named_preset | 0.8817 | 0.0183 | trusted_offline |

## Prompt and Answering

| System | Split | Evidence | Composite | Delta vs v7 | Trust |
| --- | --- | --- | --- | --- | --- |
| 1pass_fold0 | fold0 | offline_override | 0.8309 | 0.0000 | traceable_override |
| 1pass_fold1 | fold1 | offline_override | 0.8690 | -0.0028 | traceable_override |
| 5pass_fold0 | fold0 | offline_override | 0.8309 | 0.0000 | traceable_override |
| 7pass_fold0 | fold0 | offline_override | 0.8309 | 0.0000 | traceable_override |
| candidate_chat_tmpl_v1_fold0 | fold0 | offline_named_preset | 0.8502 | 0.0193 | trusted_offline |
| candidate_chat_tmpl_v1_fold1 | fold1 | offline_named_preset | 0.8741 | 0.0023 | trusted_offline |
| candidate_chat_tmpl_v2_fold0 | fold0 | offline_named_preset | 0.8518 | 0.0209 | trusted_offline |
| candidate_chat_tmpl_v2_fold1 | fold1 | offline_named_preset | 0.8933 | 0.0215 | trusted_offline |
| candidate_v5_refocus_v1_full_dev | full_dev | offline_named_preset | 0.8637 | 0.0003 | trusted_offline |
| candidate_v5refocus_reranker4b_fair_fold0 | fold0 | offline_named_preset | 0.8484 | 0.0175 | trusted_offline |
| candidate_v5refocus_reranker4b_fair_fold0 | fold0 | offline_named_preset | 0.8484 | 0.0175 | trusted_offline |
| candidate_v5refocus_reranker4b_fair_fold0_smoke | fold0 | offline_named_preset | 0.9722 | 0.0000 | trusted_offline |
| candidate_v5refocus_reranker4b_fair_fold1 | fold1 | offline_named_preset | 0.9305 | 0.0587 | trusted_offline |
| candidate_v5refocus_reranker4b_fair_fold1 | fold1 | offline_named_preset | 0.9305 | 0.0587 | trusted_offline |
| ctx5_fold0 | fold0 | offline_override | 0.8325 | 0.0016 | traceable_override |
| ctx5_fold1 | fold1 | offline_override | 0.8590 | -0.0128 | traceable_override |
| dual_ctx_fold0 | fold0 | offline_override | 0.8394 | 0.0085 | traceable_override |
| dual_ctx_fold1 | fold1 | offline_override | 0.8626 | -0.0092 | traceable_override |
| v5refocus_fold0 | fold0 | offline_override | 0.8325 | 0.0016 | traceable_override |
| v5refocus_fold1 | fold1 | offline_override | 0.8754 | 0.0036 | traceable_override |
| v5refocus_lockbox | lockbox | offline_override | 0.9033 | 0.0156 | traceable_override |

## Model Choice

| System | Split | Evidence | Composite | Delta vs v7 | Trust |
| --- | --- | --- | --- | --- | --- |
| candidate_lapalm_v1_fold0 | fold0 | offline_named_preset | 0.8430 | 0.0121 | trusted_offline |
| candidate_stronger_reranker_v1_50 | 50q | historical_confounded | 0.6553 |  | historical_only |
| candidate_stronger_reranker_v2_fair_50 | 50q | offline_named_preset | 0.8751 |  | trusted_offline |
| candidate_stronger_reranker_v2_fair_fold0 | fold0 | offline_named_preset | 0.8484 | 0.0175 | trusted_offline |
| candidate_stronger_reranker_v2_fair_fold1 | fold1 | offline_named_preset | 0.9141 | 0.0423 | trusted_offline |
| candidate_stronger_reranker_v2_fair_full_dev | full_dev | offline_named_preset | 0.8866 | 0.0232 | trusted_offline |
| candidate_stronger_reranker_v2_fair_lockbox | lockbox | offline_named_preset | 0.9254 | 0.0377 | trusted_offline |

## Failure Analysis

| System | Split | Evidence | Composite | Delta vs v7 | Trust |
| --- | --- | --- | --- | --- | --- |
| smoke_v9_10 | 10q | offline_named_preset | 0.7924 |  | trusted_offline |
| v9_full_dev | full_dev | offline_named_preset | 0.7434 | -0.1200 | trusted_offline |

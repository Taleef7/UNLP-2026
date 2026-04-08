# Master Ablation Table

| System | Block | Split | Evidence | Fairness | Trust | Composite | Delta vs v7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_v9_10 | failure_analysis | 10q | offline_named_preset | fair | trusted_offline | 0.7924 |  |
| v9_full_dev | failure_analysis | full_dev | offline_named_preset | fair | trusted_offline | 0.7434 | -0.1200 |
| candidate_lapalm_v1_fold0 | model_choice | fold0 | offline_named_preset | fair | trusted_offline | 0.8430 | 0.0121 |
| candidate_stronger_reranker_v1_50 | model_choice | 50q | historical_confounded | not_fair | historical_only | 0.6553 |  |
| candidate_stronger_reranker_v2_fair_50 | model_choice | 50q | offline_named_preset | fair | trusted_offline | 0.8751 |  |
| candidate_stronger_reranker_v2_fair_fold0 | model_choice | fold0 | offline_named_preset | fair | trusted_offline | 0.8484 | 0.0175 |
| candidate_stronger_reranker_v2_fair_fold1 | model_choice | fold1 | offline_named_preset | fair | trusted_offline | 0.9141 | 0.0423 |
| candidate_stronger_reranker_v2_fair_full_dev | model_choice | full_dev | offline_named_preset | fair | trusted_offline | 0.8866 | 0.0232 |
| candidate_stronger_reranker_v2_fair_lockbox | model_choice | lockbox | offline_named_preset | fair | trusted_offline | 0.9254 | 0.0377 |
| 1pass_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8309 | 0.0000 |
| 1pass_fold1 | prompt_answering | fold1 | offline_override | override | traceable_override | 0.8690 | -0.0028 |
| 5pass_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8309 | 0.0000 |
| 7pass_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8309 | 0.0000 |
| candidate_chat_tmpl_v1_fold0 | prompt_answering | fold0 | offline_named_preset | fair | trusted_offline | 0.8502 | 0.0193 |
| candidate_chat_tmpl_v1_fold1 | prompt_answering | fold1 | offline_named_preset | fair | trusted_offline | 0.8741 | 0.0023 |
| candidate_chat_tmpl_v2_fold0 | prompt_answering | fold0 | offline_named_preset | fair | trusted_offline | 0.8518 | 0.0209 |
| candidate_chat_tmpl_v2_fold1 | prompt_answering | fold1 | offline_named_preset | fair | trusted_offline | 0.8933 | 0.0215 |
| candidate_v5_refocus_v1_full_dev | prompt_answering | full_dev | offline_named_preset | fair | trusted_offline | 0.8637 | 0.0003 |
| candidate_v5refocus_reranker4b_fair_fold0 | prompt_answering | fold0 | offline_named_preset | fair | trusted_offline | 0.8484 | 0.0175 |
| candidate_v5refocus_reranker4b_fair_fold0 | prompt_answering | fold0 | offline_named_preset | fair | trusted_offline | 0.8484 | 0.0175 |
| candidate_v5refocus_reranker4b_fair_fold0_smoke | prompt_answering | fold0 | offline_named_preset | fair | trusted_offline | 0.9722 | 0.0000 |
| candidate_v5refocus_reranker4b_fair_fold1 | prompt_answering | fold1 | offline_named_preset | fair | trusted_offline | 0.9305 | 0.0587 |
| candidate_v5refocus_reranker4b_fair_fold1 | prompt_answering | fold1 | offline_named_preset | fair | trusted_offline | 0.9305 | 0.0587 |
| ctx5_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8325 | 0.0016 |
| ctx5_fold1 | prompt_answering | fold1 | offline_override | override | traceable_override | 0.8590 | -0.0128 |
| dual_ctx_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8394 | 0.0085 |
| dual_ctx_fold1 | prompt_answering | fold1 | offline_override | override | traceable_override | 0.8626 | -0.0092 |
| v5refocus_fold0 | prompt_answering | fold0 | offline_override | override | traceable_override | 0.8325 | 0.0016 |
| v5refocus_fold1 | prompt_answering | fold1 | offline_override | override | traceable_override | 0.8754 | 0.0036 |
| v5refocus_lockbox | prompt_answering | lockbox | offline_override | override | traceable_override | 0.9033 | 0.0156 |
| _smoke_a30_dense_margin_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.9722 | 0.0000 |
| candidate_ddl_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8183 | -0.0126 |
| candidate_ddl_v2_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8502 | 0.0193 |
| candidate_ddl_v2_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.8983 | 0.0265 |
| candidate_ddl_v2_fulldev | retrieval_doc_stability | full_dev | offline_named_preset | fair | trusted_offline | 0.8811 | 0.0177 |
| candidate_ddl_v2_lockbox | retrieval_doc_stability | lockbox | offline_named_preset | fair | trusted_offline | 0.8880 | 0.0003 |
| candidate_ddl_v3_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8502 | 0.0193 |
| candidate_ddl_v3_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.9025 | 0.0307 |
| candidate_ddl_v3_fulldev | retrieval_doc_stability | full_dev | offline_named_preset | fair | trusted_offline | 0.8812 | 0.0178 |
| candidate_ddl_v3_lockbox | retrieval_doc_stability | lockbox | offline_named_preset | fair | trusted_offline | 0.8929 | 0.0053 |
| candidate_dense_margin_lock_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8309 | 0.0000 |
| candidate_dense_margin_lock_v1_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.8718 | 0.0000 |
| candidate_dense_margin_lock_v1_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.8718 | 0.0000 |
| candidate_dense_margin_lock_v1_full_dev | retrieval_doc_stability | full_dev | offline_named_preset | fair | trusted_offline | 0.8611 | -0.0023 |
| candidate_dense_margin_lock_v1_lockbox | retrieval_doc_stability | lockbox | offline_named_preset | fair | trusted_offline | 0.8877 | 0.0000 |
| candidate_dense_margin_lock_v1_thresh020_fold1 | retrieval_doc_stability | fold1 | offline_override | override | traceable_override | 0.8718 | 0.0000 |
| candidate_dense_margin_lock_v1_thresh030_fold1 | retrieval_doc_stability | fold1 | offline_override | override | traceable_override | 0.8718 | 0.0000 |
| candidate_extraction_blocks_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8175 | -0.0134 |
| candidate_hybrid_rrf_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8210 | -0.0098 |
| candidate_locked_top_doc_v1_50 | retrieval_doc_stability | 50q | offline_named_preset | fair | trusted_offline | 0.8673 |  |
| candidate_locked_top_doc_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8320 | 0.0011 |
| candidate_locked_top_doc_v1_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.8723 | 0.0005 |
| candidate_locked_top_doc_v1_lockbox | retrieval_doc_stability | lockbox | offline_named_preset | fair | trusted_offline | 0.8874 | -0.0003 |
| candidate_sc_v3_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8012 | -0.0297 |
| candidate_sc_v3_fold0_r2 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8195 | -0.0114 |
| candidate_sc_v3_fold0_r3 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8309 | 0.0000 |
| candidate_sc_v3_fold1 | retrieval_doc_stability | fold1 | offline_named_preset | fair | trusted_offline | 0.8718 | 0.0000 |
| candidate_sc_v3_lockbox | retrieval_doc_stability | lockbox | offline_named_preset | fair | trusted_offline | 0.8827 | -0.0049 |
| candidate_sc_v3_smoke | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8258 | -0.0417 |
| candidate_structure_chunks_v1_10 | retrieval_doc_stability | 10q | offline_named_preset | fair | trusted_offline | 0.9770 |  |
| candidate_structure_chunks_v1_50 | retrieval_doc_stability | 50q | offline_named_preset | fair | trusted_offline | 0.8860 |  |
| candidate_structure_chunks_v1_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8149 | -0.0160 |
| candidate_structure_chunks_v1_full | retrieval_doc_stability | full_dev | offline_named_preset | fair | trusted_offline | 0.8675 | 0.0041 |
| candidate_structure_chunks_v2_doc_guard_10 | retrieval_doc_stability | 10q | offline_named_preset | fair | trusted_offline | 0.9770 |  |
| candidate_structure_chunks_v2_doc_guard_50 | retrieval_doc_stability | 50q | offline_named_preset | fair | trusted_offline | 0.9160 |  |
| candidate_structure_chunks_v2_doc_guard_fold0 | retrieval_doc_stability | fold0 | offline_named_preset | fair | trusted_offline | 0.8210 | -0.0099 |
| candidate_structure_chunks_v2_doc_guard_full | retrieval_doc_stability | full_dev | offline_named_preset | fair | trusted_offline | 0.8817 | 0.0183 |
| smoke_v7 | submission_lineage | 10q | offline_named_preset | fair | trusted_offline | 1.0000 |  |
| smoke_v7_10 | submission_lineage | 10q | offline_named_preset | fair | trusted_offline | 0.9770 |  |
| structure_doc_guard | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8415 |  |
| v11 | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8320 |  |
| v13 | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8231 |  |
| v14 | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8320 |  |
| v5_offline | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8043 |  |
| v6_reranker | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8618 |  |
| v7 | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.8688 |  |
| v7_50 | submission_lineage | 50q | offline_named_preset | fair | trusted_offline | 0.8673 |  |
| v7_baseline_fold0 | submission_lineage | fold0 | offline_named_preset | fair | trusted_offline | 0.8309 | 0.0000 |
| v7_baseline_fold0_smoke | submission_lineage | fold0 | offline_named_preset | fair | trusted_offline | 0.9861 | 0.0000 |
| v7_baseline_fold1 | submission_lineage | fold1 | offline_named_preset | fair | trusted_offline | 0.8718 | 0.0000 |
| v7_baseline_lockbox | submission_lineage | lockbox | offline_named_preset | fair | trusted_offline | 0.8877 | 0.0000 |
| v7_full_dev | submission_lineage | full_dev | offline_named_preset | fair | trusted_offline | 0.8634 | 0.0000 |
| v9 | submission_lineage | public_kaggle | public_kaggle | public_final | trusted_public | 0.6716 |  |
| candidate_avir_threshold_v1_10 | supplementary | 10q | offline_named_preset | fair | trusted_offline | 0.9953 |  |
| candidate_avir_threshold_v1_50 | supplementary | 50q | offline_named_preset | fair | trusted_offline | 0.8707 |  |
| candidate_avir_threshold_v1_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8308 | -0.0001 |
| candidate_avir_threshold_v1_full | supplementary | full_dev | offline_named_preset | fair | trusted_offline | 0.8663 | 0.0029 |
| candidate_confidence_gated_v1_10 | supplementary | 10q | offline_named_preset | fair | trusted_offline | 0.9770 |  |
| candidate_confidence_gated_v1_50 | supplementary | 50q | offline_named_preset | fair | trusted_offline | 0.8873 |  |
| candidate_confidence_gated_v1_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8164 | -0.0145 |
| candidate_confidence_gated_v1_full | supplementary | full_dev | offline_named_preset | fair | trusted_offline | 0.8617 | -0.0017 |
| candidate_ddl_v3_exhaust_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8451 | 0.0142 |
| candidate_evidence_first_v1_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8308 | -0.0001 |
| candidate_no_bleed_v1_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8577 | 0.0268 |
| candidate_no_bleed_v1_fold1 | supplementary | fold1 | offline_named_preset | fair | trusted_offline | 0.8622 | -0.0096 |
| candidate_no_bleed_v1_lockbox | supplementary | lockbox | offline_named_preset | fair | trusted_offline | 0.8575 | -0.0301 |
| candidate_page_summary_selector_v1_10 | supplementary | 10q | offline_named_preset | fair | trusted_offline | 0.9709 |  |
| candidate_page_summary_selector_v1_50 | supplementary | 50q | offline_named_preset | fair | trusted_offline | 0.8640 |  |
| candidate_page_summary_selector_v1_fold0 | supplementary | fold0 | offline_named_preset | fair | trusted_offline | 0.8301 | -0.0008 |
| reranker_ablation_2026-03-30 | supplementary | unknown | offline_named_preset | fair | trusted_offline |  |  |

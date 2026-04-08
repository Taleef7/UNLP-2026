# Negative Results

These rows are useful in the paper because they are clearly negative, unstable, or methodologically limited.

| System | Split | Composite | Delta vs v7 | Fairness | Trust | Flags |
| --- | --- | --- | --- | --- | --- | --- |
| candidate_chat_tmpl_v1_fold0 | fold0 | 0.8502 | 0.0193 | fair | trusted_offline |  |
| candidate_chat_tmpl_v1_fold1 | fold1 | 0.8741 | 0.0023 | fair | trusted_offline |  |
| candidate_chat_tmpl_v2_fold0 | fold0 | 0.8518 | 0.0209 | fair | trusted_offline |  |
| candidate_chat_tmpl_v2_fold1 | fold1 | 0.8933 | 0.0215 | fair | trusted_offline |  |
| _smoke_a30_dense_margin_fold0 | fold0 | 0.9722 | 0.0000 | fair | trusted_offline |  |
| candidate_dense_margin_lock_v1_fold0 | fold0 | 0.8309 | 0.0000 | fair | trusted_offline |  |
| candidate_dense_margin_lock_v1_fold1 | fold1 | 0.8718 | 0.0000 | fair | trusted_offline | Historical pre-audit evidence covered only fold1; later paper reruns completed the full ladder. |
| candidate_dense_margin_lock_v1_fold1 | fold1 | 0.8718 | 0.0000 | fair | trusted_offline |  |
| candidate_dense_margin_lock_v1_thresh020_fold1 | fold1 | 0.8718 | 0.0000 | override | traceable_override | Traceable override run: benchmark_manifest.json preserves override_json.; Historical pre-audit evidence covered only fold1; later paper reruns completed the full ladder. |
| candidate_dense_margin_lock_v1_thresh030_fold1 | fold1 | 0.8718 | 0.0000 | override | traceable_override | Traceable override run: benchmark_manifest.json preserves override_json.; Historical pre-audit evidence covered only fold1; later paper reruns completed the full ladder. |
| candidate_dense_margin_lock_v1_full_dev | full_dev | 0.8611 | -0.0023 | fair | trusted_offline |  |
| candidate_dense_margin_lock_v1_lockbox | lockbox | 0.8877 | 0.0000 | fair | trusted_offline |  |
| candidate_extraction_blocks_v1_fold0 | fold0 | 0.8175 | -0.0134 | fair | trusted_offline |  |
| candidate_hybrid_rrf_v1_fold0 | fold0 | 0.8210 | -0.0098 | fair | trusted_offline |  |
| candidate_lapalm_v1_fold0 | fold0 | 0.8430 | 0.0121 | fair | trusted_offline |  |
| candidate_stronger_reranker_v1_50 | 50q | 0.6553 |  | not_fair | historical_only | Unfair reranker comparison: candidate_stronger_reranker_v1 used qwen3_4b with max_length=1024 while v7_baseline used 2048. |
| smoke_v9_10 | 10q | 0.7924 |  | fair | trusted_offline |  |
| v9_full_dev | full_dev | 0.7434 | -0.1200 | fair | trusted_offline |  |

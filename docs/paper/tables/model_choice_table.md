# Model Choice Table

| Lane | Artifact | Composite | Delta vs v7 | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| qwen3_0_6b_baseline | v7_full_dev | 0.8634 |  | mainline | Competition anchor. |
| qwen3_4b_unfair_history |  |  |  | historical_confounded | Do not cite as a fair 0.6B vs 4B comparison. |
| qwen3_4b_fair | candidate_stronger_reranker_v2_fair_full_dev | 0.8866 | 0.0232 | trusted_offline | Main offline reranker-size result. |
| qwen3_8b_feasibility |  |  |  | feasibility_only | Feasibility-only lane pending explicit teardown diagnosis. |

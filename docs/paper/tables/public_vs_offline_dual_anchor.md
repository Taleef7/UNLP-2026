# Public vs Offline Dual Anchor

| System | Evidence | Public | Private | Offline Composite | Trust | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| v7 | public_kaggle | 0.8688 | 0.8722 | 0.8688 | trusted_public | Best actual competition submission and final paper anchor. |
| v13 | public_kaggle | 0.8231 | 0.8441 | 0.8231 | trusted_public | Prompt-only submission that looked locally promising but regressed on the final leaderboard. |
| v14 | public_kaggle | 0.8320 | 0.8396 | 0.8320 | trusted_public | Dense doc lock v3 stayed negative on Kaggle despite strong local offline evidence. |
| v7_full_dev | offline_named_preset |  |  | 0.8634 | trusted_offline |  |
| candidate_stronger_reranker_v2_fair_full_dev | offline_named_preset |  |  | 0.8866 | trusted_offline |  |

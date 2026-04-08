# System Lineage

## Dual-Anchor Narrative

- Competition anchor: `v7` is the best actual competition submission with public `0.8688`, private `0.8722`, and final rank `7`.
- Analysis anchor: `candidate_stronger_reranker_v2_fair` remains the strongest trustworthy fair offline system unless a clean round-2 combo beats it on both `lockbox` and `full_dev`.
- Feasibility lane: `qwen3_8b` stays outside the main model-choice matrix unless it can coexist with MamayLM end to end on a single A30.

## Submission Lineage

| System | Public | Private | Rank | Local Counterpart | Interpretation |
| --- | --- | --- | --- | --- | --- |
| v7 | 0.8688 | 0.8722 | 7 | v7_full_dev | Best actual competition submission and final paper anchor. |
| v13 | 0.8231 | 0.8441 |  | candidate_v5_refocus_v1_full_dev | Prompt-only submission that looked locally promising but regressed on the final leaderboard. |
| v14 | 0.8320 | 0.8396 |  | candidate_ddl_v3_fulldev | Dense doc lock v3 stayed negative on Kaggle despite strong local offline evidence. |
| v11 | 0.8320 | 0.8410 |  | candidate_ddl_v2_fulldev | Dense doc lock v2 also failed to generalize despite the long-document heuristic. |
| structure_doc_guard | 0.8415 | 0.8600 |  | candidate_structure_chunks_v2_doc_guard_full | Strong local winner that still lost decisively to the simpler v7 system in competition. |
| v9 | 0.6716 | 0.7265 |  | v9_full_dev | Clear negative control for the paper's failure-analysis section. |
| v6_reranker | 0.8618 | 0.8517 |  | candidate_stronger_reranker_v1 | Historically strong public score, but the linked local 4B comparison remains confounded. |
| v5_offline | 0.8043 | 0.8124 |  |  | Early public baseline retained for leaderboard chronology. |

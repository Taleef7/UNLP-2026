# Execution Protocol

## Scope

This document records the paper-facing execution protocol after the competition ended and the repo shifted from deadline-era experimentation to paper-quality evidence gathering.

## Trusted Hardware Lane

- Use the `a30` standby lane for MamayLM end-to-end ablations.
- Keep all paper-completion reruns on the same A30 lane unless a separate compatibility note says otherwise.
- Reason:
  - A30 completes the full single-GPU pipeline end to end.
  - A100 retrieval and reranking succeeded, but MamayLM GGUF load failed with `Illegal instruction` in this environment.

See [hardware_compatibility.md](/scratch/gilbreth/tamst01/unlp2026/docs/paper/ablations/hardware_compatibility.md) for the host-level compatibility evidence.

## Interactive Session Workflow

1. Start `tmux`.
2. Launch:
   `sinteractive -A pfw-cs -p a30 --qos=standby --gres=gpu:1 --mem=60G -c 6 -t 04:00:00`
3. Start a terminal capture:
   `script -f logs/interactive_a30_paper_2026-03-31.log`
4. Keep new paper runs under:
   `outputs/paper_ablation_runs_r2/2026-03-31_a30/`
5. Use one-question smoke runs before each new preset family.
6. Resume only missing splits if the standby session ends.

## Important Corrections

- `full_dev` for paper-facing runs means `data/dev_questions.csv`, not `data/splits/val.csv`.
- The March 31 audit caught an incorrect temporary mapping of `full_dev -> data/splits/val.csv` in the round-2 runner and fixed it before accepting any new evidence.
- The first corrected `v13` full-dev rerun therefore uses the same 461-question source as `v7_full_dev`.

## Stage Order

1. Close the missing `v13` offline evidence on `data/dev_questions.csv`.
2. Run the clean fair-4B matrix for:
   - `candidate_v5refocus_reranker4b_fair`
   - `candidate_dense_doc_lock_v3_reranker4b_fair`
   - `candidate_v5refocus_ddl_v3_reranker4b_fair`
3. Run the final `qwen3_8b` teardown diagnostic with GPU memory snapshots.
4. Backfill `paper_notes.md` and rebuild the paper audit tables.

## Resume Rule

- Never overwrite a finished round-2 artifact.
- Use `--skip-existing` when relaunching the matrix.
- If a session ends in the middle of a split, rerun only that unfinished split directory.

## Why This Protocol Exists

The earlier March 30 queued chain was good enough for triage, but not for a paper-quality matrix:

- the fair-4B chain timed out before completing the full combination ladder
- the 8B lane needed a more explicit teardown diagnosis
- one round-2 runner version briefly pointed `full_dev` at the wrong split

The corrected March 31 protocol fixes those issues and makes every paper-facing claim traceable to a manifest-backed artifact under one clean hardware lane.

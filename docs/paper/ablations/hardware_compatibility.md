# Hardware Compatibility Notes

**Date:** March 30, 2026

## Summary

The current UNLP 2026 pipeline is **A30-compatible** but **A100-incompatible** for end-to-end MamayLM runs in this environment.

## What Was Observed

- The frontend host (`gilbreth-fe03`) could not run the benchmark suite directly because its only A30 GPU was already occupied by unrelated processes from other users.
- Two separate `a100-80gb` runs were attempted on:
  - `gilbreth-k022`
  - `gilbreth-k034`
- Both A100 runs:
  - completed extraction
  - loaded BGE-M3
  - completed retrieval
  - loaded the reranker
  - completed page reranking
  - **crashed with `Illegal instruction (core dumped)` at MamayLM GGUF load**
- An `a30` run launched with `--qos=standby` on `gilbreth-b009`:
  - completed extraction, retrieval, reranking, and MamayLM load successfully
  - completed a 1-question smoke benchmark successfully

## Interpretation

- The `Illegal instruction` issue is not node-specific within the `a100-80gb` partition, because it reproduced on two different nodes.
- The issue is also not a generic pipeline bug, because the same code path succeeds on `a30`.
- For the paper-facing ablation execution pass, **A30 standby** is the trusted cluster lane for MamayLM-based follow-up benchmarks.
- Any A100 timings or failed attempts should be treated as **hardware compatibility findings**, not as quality evidence against the model or preset being evaluated.

## Reporting Guidance

- Report A30-based benchmark results as the trusted offline execution environment for this audit pass.
- Report A100 failure as an environment constraint:
  - compatible for retrieval and reranking stages
  - incompatible for full MamayLM inference in the current setup
- Do not compare A100 aborted runs against A30 completed runs as if they were valid system-level ablations.

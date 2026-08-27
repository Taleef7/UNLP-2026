# When the Harness Answers "A"

Code, artifacts, and paper source for *"When the Harness Answers 'A': Score-Inflating Evaluation Bugs
Camouflaged by Model Bias."*

Taleef Tamsal, Jonathan Rusert — Purdue University Fort Wayne.

Paper: [`paper/harness_answers_a.pdf`](paper/harness_answers_a.pdf) ([source](paper/harness_answers_a.tex)).

## What this is

Evaluation harnesses are usually studied as a source of *variance*. This repository documents a more
dangerous class of fault: bugs that reliably **inflate** accuracy — so a leaderboard cannot flag them, since
the bug improves your standing — and that are **camouflaged**, because their visible symptom coincides with
a documented model behaviour.

We audit a retrieval-augmented multiple-choice pipeline for Ukrainian document QA that placed mid-field in
a shared task, and find two such bugs in the system as submitted. Every failure path in that harness
terminates at the answer **"A"**, and LLM option-A bias is a documented finding — so a parser failure looked
expected. The literature that explains the symptom is what stopped anyone looking for the cause.

## The two bugs

Both live in [`notebooks/pipeline_shared.py`](notebooks/pipeline_shared.py):

1. **A stop sequence that empties the generation.** `stop=["\n", "."]` on a prompt ending
   `"Правильний варіант:"` — the model's leading newline halts generation at **token zero**, returning the
   empty string. `extract_answer("")` falls through every branch to an unconditional `return "A"`.
   `finish_reason` is `"stop"`, never `"length"`, so no truncation check catches it. On **36 of 1,383**
   generation calls the pipeline received nothing and answered "A"; this decided **13 questions** by a blind
   guess, **2** correct by chance.
2. **A prefix cache that corrupts self-consistency voting.** The binding cancels its own reset on a
   prefix-cache hit, so votes 2–3 evaluate a single token against a warm KV cache while vote 1 evaluates a
   divergent suffix — same prompt, different logits. The falsifiable prediction (calls 2 and 3 share one
   code path and must always agree) holds on **1,844/1,844** vote-triples; forcing a true reset makes all
   three passes identical on **461/461** questions.

**Correcting both lowers the score** — they were manufacturing it:

| system | correct | composite | |
|---|---:|---:|---|
| as shipped | 406/461 | 0.880780 | both bugs; 13 answers were blind guesses |
| stop sequence fixed | 408/461 | 0.882471 | cache bug still present |
| **both fixed** | **407/461** | **0.881423** | **the clean system** |

These deltas are below the McNemar floor at these counts and are reported as directionally suggestive only;
the load-bearing evidence is the deterministic forensics above.

Both bugs are exposed as flags (`stop_on_newline`, `reset_kv_cache`) **defaulting to the shipped behaviour
bit-for-bit**, so the buggy and the corrected runs each reproduce.

## The same camouflage misattributes tool defaults

Three controlled experiments in which a default is read as a fact about a language, a modality, or
quantization:

| the claim | the actual cause |
|---|---|
| "hybrid fusion doesn't transfer to Cyrillic" | a fusion weight of `0.5` — on 18 MIRACL languages, a **14.6-point swing** |
| "visual retrieval can't read Cyrillic" | the base VLM's pretraining coverage — the **sign flips** across models |
| "CoT collapses under quantization" | the answer parser — the **artifact (6.67pp) is comparable to the effect (5.33pp)** |

## Verification (run these; they are the point)

```bash
python3 scripts/provenance.py            # every artifact commit-stamped and input-hash-pinned
python3 scripts/check_paper_numbers.py   # every number in the paper re-derived from raw JSON
```

- **`scripts/provenance.py`** stamps each artifact with the commit it was written at, refuses to trust one
  written while `scripts/` was dirty, and **content-hashes its inputs** — so regenerating a dependency
  forces `STALE_INPUT` rather than letting a stale artifact quietly keep looking authoritative.
- **`scripts/check_paper_numbers.py`** pairs a literal string in the manuscript with the figure it asserts,
  recomputed from `outputs/*.json`, and exits non-zero on any disagreement. No number in the paper is typed
  by hand.

## Pipeline

BGE-M3 dense retrieval → Qwen3-0.6B cross-encoder page reranker → MamayLM-Gemma-3-12B-IT (Q4_K_M,
`llama.cpp`) for 6-option MCQ, with three-pass self-consistency voting. Composite =
`0.5·answer + 0.25·doc + 0.25·page-proximity`.

The submitted source is included at
[`notebooks/v7_submission_kaggle_full.py`](notebooks/v7_submission_kaggle_full.py); bug 1 is present in the
literal submitted code. That run voted stochastically (`VOTE_TEMP=0.5`), so the determinism analysis uses a
`vote_temp=0` reconstruction that isolates the cache effect. All analyses are over the 461-question
development set and **are not equated with the private-test leaderboard score**.

## Repository map

```
paper/
  harness_answers_a.tex        # the paper; build with paper/build.sh
  harness_answers_a.bib        # references
  fig_bugs.pdf                 # the evidence figure, from scripts/gen_paper_figures.py

scripts/
  provenance.py                # commit stamping + input hashing
  check_paper_numbers.py       # every reported number re-derived from raw JSON
  e2e_run.py / e2e_analyze.py  # the four-cell harness experiment behind both bugs
  kv_cache_probe.py            # the prefix-cache probe
  c2_cluster_bootstrap.py      # document-clustered bootstrap for the visual-retrieval control
  retrieval_eval.py            # fusion / retrieval evaluation
  c1_nested_cv.py              # nested CV for the fusion weight
  miracl_eval2.py, a2_*.py     # the 18-language MIRACL arm
  colsmol_script_control2.py   # the 2x2 page-script x query-script control
  quant_cot_factorial.py       # the quantization x CoT factorial
  gen_paper_figures.py         # the evidence figure, from raw JSON

notebooks/
  pipeline_shared.py           # all pipeline logic; both bug flags live here
  v7_submission_kaggle_full.py # the submitted source
  pipeline_presets.json        # named presets

outputs/                       # raw per-question artifacts behind every number in the paper
data/dev_questions.csv         # the 461 development questions
```

## Reproducing

```bash
python3 scripts/benchmark_candidate.py --preset v7_baseline \
  --questions data/dev_questions.csv --output-dir outputs/benchmarks/v7_baseline
```

Models required: `BAAI/bge-m3`, `INSAIT/MamayLM-Gemma-3-12B-IT-GGUF` (Q4_K_M), `Qwen/Qwen3-0.6B`.
GPU ≥ 16 GB. `llama.cpp` with Python bindings — **build it for your CPU**: a prebuilt
`libggml-cpu.so` may contain AVX-512 instructions that many Zen 3 nodes do not support, and it dies with
`Illegal instruction`. See [`scripts/build_llama_cpp.sh`](scripts/build_llama_cpp.sh).

The competition PDFs are not redistributed; see the [UNLP 2026 shared task page](https://unlp.org.ua/shared-task/)
for data access. Place them under `data/raw_pdfs/` by domain.

Large embedding caches (`*.npz`) are regenerable and are not committed; the one small cache that a
stamped artifact declares as an input is included, so `provenance.py` can verify it.

## Citation

```bibtex
@article{tamsal2026harness,
  title  = {When the Harness Answers ``A'': Score-Inflating Evaluation Bugs Camouflaged by Model Bias},
  author = {Tamsal, Taleef and Rusert, Jonathan},
  year   = {2026}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The paper text and figures are
CC BY 4.0.

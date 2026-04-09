# PFW at UNLP 2026: Ukrainian Document Question Answering

**Paper:** "When Local Wins Fail to Generalize: PFW at UNLP 2026 Ukrainian Document QA"
**Author:** Taleef Tamsal, Purdue University
**Venue:** UNLP 2026 Workshop at ACL 2026
**Result:** Private score 0.8722, Rank 7/15

## Overview

A retrieval-augmented QA pipeline for Ukrainian document question answering combining BGE-M3 dense retrieval, a Qwen3-0.6B cross-encoder page reranker, and MamayLM-Gemma-3-12B-IT for 6-option MCQ answering. The system achieves rank 7 of 15 teams on the UNLP 2026 shared task. The key finding is that conservative baselines outperform locally-optimized variants under hidden domain shift: all offline improvements regressed on the Kaggle private leaderboard.

## Pipeline Architecture

1. **Text Extraction** -- PyMuPDF extracts page-level plain text from PDF documents
2. **Dense Retrieval** -- BGE-M3 (FP16) encodes pages and queries; FAISS exact inner-product search retrieves top-10 pages (Doc@1 = 0.933)
3. **Page Reranking** -- Qwen3-0.6B cross-encoder rescores pages within top-3 candidate documents (answer-aware, max_length=2048)
4. **MCQ Answering** -- MamayLM-Gemma-3-12B-IT (Q4_K_M, llama.cpp) with 3-pass majority voting and a direct prompt format

## Repository Structure

```
paper/                          # LaTeX source and compiled PDF
  unlp2026_paper.tex            # Paper source
  unlp2026.bib                  # Bibliography
  unlp2026_paper.pdf            # Compiled paper

notebooks/
  pipeline_shared.py            # All pipeline logic (retrieval, reranking, MCQ, metrics)
  pipeline_presets.json         # Named preset configurations (34 presets)
  v7_kaggle_submission.py       # Kaggle submission script (v7 baseline)
  kaggle_bundle.py              # Renders standalone Kaggle scripts from shared runtime

scripts/
  benchmark_candidate.py        # Main local benchmark runner
  diff_benchmark_runs.py        # Question-level diff between two benchmark runs
  00_eval_harness.py            # Standalone metric computation
  00_create_splits.py           # Generate CV folds and lockbox splits
  check_kaggle_parity.py        # Verify shared runtime matches rendered Kaggle bundle

data/
  dev_questions.csv             # Development set (461 questions)
  splits/                       # CV folds and lockbox partition

docs/paper/
  paper_analysis_report.md      # Analysis report with all cited statistics
  tables/                       # Source data for all paper tables
```

## Reproducing Results

### Prerequisites

- Python 3.9+
- GPU with >= 16 GB VRAM (A30 or better for benchmarking)
- `llama.cpp` with Python bindings

### Model Downloads

The following models must be downloaded to `models/`:

- **BGE-M3**: `BAAI/bge-m3` from Hugging Face
- **MamayLM**: `INSAIT/MamayLM-Gemma-3-12B-IT-GGUF` (Q4_K_M quantization) from Hugging Face
- **Qwen3-0.6B**: `Qwen/Qwen3-0.6B` from Hugging Face (used as cross-encoder reranker)

### Running the Pipeline

```bash
# Full development set benchmark (461 questions)
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --questions data/dev_questions.csv \
  --output-dir outputs/benchmarks/v7_baseline

# Smoke test (10 questions)
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --n-questions 10 \
  --output-dir outputs/benchmarks/smoke

# Compare two benchmark runs
python3 scripts/diff_benchmark_runs.py \
  --base-dir outputs/benchmarks/v7_baseline \
  --candidate-dir outputs/benchmarks/<candidate>
```

### Running Evaluation

```bash
python3 scripts/00_eval_harness.py
```

## Key Results

| System | Answer Acc. | Doc Acc. | Page Prox. | Composite |
|--------|------------|----------|------------|-----------|
| v7 (local full-dev) | 0.885 | 0.889 | 0.794 | 0.8634 |
| v7 (Kaggle public) | -- | -- | -- | 0.8688 |
| v7 (Kaggle private) | -- | -- | -- | 0.8722 |

## Citation

```bibtex
@inproceedings{tamsal2026unlp,
  title={When Local Wins Fail to Generalize: {PFW} at {UNLP} 2026 Ukrainian Document {QA}},
  author={Tamsal, Taleef},
  booktitle={Proceedings of the Ukrainian NLP Workshop (UNLP)},
  year={2026}
}
```

## License

This repository is released for academic research purposes. The competition data (PDF documents) is not included; see the [UNLP 2026 shared task page](https://unlp.org.ua/shared-task/) for access.

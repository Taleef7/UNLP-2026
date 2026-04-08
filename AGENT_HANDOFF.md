# Agent Handoff — UNLP 2026 Ukrainian Document QA

**Last updated:** 2026-04-07  
**Repository:** `/scratch/gilbreth/tamst01/unlp2026`  
**Author:** Taleef Tamsal (tamst01@purdue.edu), Purdue University

This document is written for an external auditor, reviewer, or new agent who needs a complete, accurate picture of everything done in this project. Every number cited here has a source; the source file is noted. Numbers that could not be verified from project files are explicitly flagged.

---

## 1. Competition Overview

**Task:** UNLP 2026 Shared Task — Ukrainian Document Question Answering  
**Format:** 6-option multiple-choice questions (MCQ, options A–F) grounded in domain-specific PDF documents. Each submission must answer the question AND identify the correct document AND the correct page.

**Composite metric:**
```
S = 0.5 × answer_accuracy + 0.25 × doc_accuracy + 0.25 × page_proximity
```
where `page_proximity = max(0, 1 - |page_pred - page_true| / N)`, and N is the total pages in the **predicted** document (not the ground truth document). Partial credit is granted for pages near the correct page.

**Infrastructure:** Kaggle competition, T4×2 GPU, 9-hour wall-clock hard limit, no internet access during inference. Models and data must be packaged as Kaggle dataset inputs.

**Competition size:** 15 teams total. Winning team: public 0.9377, private 0.9598 (source: paper introduction; winning team's private score not independently verified from project files — CLAUDE.md only records public leader as 0.9377).

---

## 2. Dataset Description

**Source:** `data/dev_questions.csv` — 461 questions total  
**Source (splits):** `data/splits/` — fold CSVs, lockbox CSV  
**Source (documents):** `data/raw_pdfs/` — 41 unique PDFs

| Domain | Label | Subject | Docs | Questions | Total Pages | Avg Pages/Doc |
|--------|-------|---------|------|-----------|-------------|---------------|
| Domain 1 | visible | Sports competition regulations | 11 | 198 | 865 | 78.6 |
| Domain 2 | visible | Medical drug instructions | 30 | 263 | 256 | 8.5 |
| **Total** | | | **41** | **461** | **1,121** | |
| Domain 3 | **HIDDEN** | Unknown (private test only) | unknown | unknown | unknown | unknown |

**Critical note:** The private test set includes a third hidden domain not present in development. This is the identified primary driver of the local-vs-Kaggle generalization gap. See methodology_audit.md: "The official README says dev and test_public have two domains, while test_private has three domains."

**Answer label distribution:** Approximately uniform across A–F (confirmed in task description).

**Scanned content:** 16 pages in Domain 2 have <50 characters — likely scanned images with no usable OCR text. These are not recoverable without an OCR fallback (which is not implemented).

---

## 3. Pipeline Architecture — v7 (Final Submission)

The v7 pipeline is the competition anchor. It is the best-performing submission on both public and private Kaggle. All other candidates are evaluated relative to v7.

**Source files:**  
- `notebooks/pipeline_shared.py` — all logic  
- `notebooks/pipeline_presets.json` — preset `v7_baseline` (the canonical config)  
- `notebooks/v7_kaggle_submission.py` — Kaggle bundle for v7

### Stage 1: Text Extraction
- Tool: PyMuPDF
- Output: page-level plain text
- No OCR fallback
- Extraction time for all 1,121 pages: ~4 seconds
- 16 pages in Domain 2 are effectively blank (<50 chars)

### Stage 2: Dense Retrieval
- Model: **BGE-M3** (BAAI/bge-m3, 568M parameters, FP16)
  - Citation: Chen et al. (2024), Findings of ACL
- Storage: FAISS `IndexFlatIP` (exact inner product search)
- Query: **question text only** (adding answer options to the query hurts Doc@1 by 7.6 pp — see memory note and confirmed in ablations)
- Embedding dimension: 1024 (dense-only, not sparse/multi-vector)
- Retrieval: top-10 pages; then derive top-3 highest-scoring **documents** from those pages for reranking
- Key result: Doc@1 = 0.933, Doc@10 = 0.994 (source: `logs/checkpoint_1_retrieval.md`, confirmed by `docs/paper/paper_analysis_report.md` §6)

### Stage 3: Page-Level Cross-Encoder Reranking
- Model: **Qwen3-0.6B** cross-encoder (reranker mode)
  - Citation: Qwen Team (2025)
- Input: Each page in the top-3 candidate documents is reranked
- Query to reranker: question text + all six answer options (answer-aware reranking)
- `max_length = 2048` tokens
- Output: top-1 page per document determines final document selection; the page with highest reranker score becomes the predicted page
- Lift from reranking: page recall@1 improves from 0.4707 (dense only) to 0.5380 (+6.73 pp)
  - Source: `docs/paper/paper_analysis_report.md` §6
- Reranking wall time (v7, 461 questions): 624.1 seconds
  - Source: `docs/paper/tables/runtime_table.md`

### Stage 4: MCQ Answer Generation
- Model: **MamayLM-Gemma-3-12B-IT**, quantized to Q4_K_M (6.8 GB VRAM)
  - Citation: INSAIT Institute (2025) — NOTE: model is based on Gemma-3 which was released 2025, not 2024
  - Served via `llama.cpp` (Gerganov, 2023)
- Inference: greedy decoding (temperature = 0)
- Prompt format: top-2 pages from the selected document + question + 6 answer options + directive "Answer (letter only):" in Ukrainian. **No chat template applied.**
- **3-pass majority voting**: run the LLM 3 times on the same input, take the majority vote
  - Unanimous 3/3 agreement: 77.2% of questions → 90.2% answer accuracy
  - Majority 2/1 agreement: 22.6% of questions → 83.7% answer accuracy  
  - All-different 0/3: 0.2% of questions → 0.0% accuracy (random in the tie-break)
  - Delta (unanimous vs. majority): +6.5 pp
  - Source: `docs/paper/paper_analysis_report.md` §5
- Context window: 8192 tokens (but effectively ~4096 usable for MCQ after prompt overhead)

**Total v7 benchmark time (full dev, 461 questions, A30 GPU):**
- Extraction: 6.2s
- Retrieval: 38.0s
- Reranking: 624.1s
- MCQ answering: 1127.2s
- **Total: 1,795.5 seconds ≈ 30 minutes**
- Source: `docs/paper/tables/runtime_table.md`
- Kaggle T4×2 projection: ~60 minutes (within 9-hour limit)

---

## 4. Official Competition Results

All scores verified from `docs/paper/tables/final_submission_results.md` and `docs/paper/tables/final_submission_results.csv`.

| Submission | Key Change | Local Full-Dev | Kaggle Public | Kaggle Private | Rank |
|-----------|-----------|---------------|--------------|----------------|------|
| v5 | Early pipeline | N/A | 0.8043 | 0.8124 | — |
| v6 | Reranker added | N/A (confounded) | 0.8618 | 0.8517 | — |
| **v7** | **Conservative rollback** | **0.8634** | **0.8688** | **0.8722** | **7** |
| v9 | Doc-level rerank | 0.7434 | 0.6716 | 0.7265 | — |
| structure_doc_guard | Structure chunks | 0.8817 | 0.8415 | 0.8600 | — |
| v14 (DDL v3) | Dense-doc-lock | 0.8812 | 0.8320 | 0.8396 | — |
| v11 (DDL v2) | Dense-doc-lock v2 | 0.8811 | 0.8320 | 0.8410 | — |
| v13 (refocus) | Prompt refocus | 0.8637 | 0.8231 | 0.8441 | — |

**Final rank: 7th of 15 teams.** Public leaderboard at time of best submission had v7 at 4th place (per CLAUDE.md); final private ranking moved to 7th.

### v7 Full-Dev Breakdown (461 questions)
Source: `docs/paper/paper_analysis_report.md` §1, `outputs/benchmarks/v7_full_dev/summary.json`

| Metric | Score |
|--------|-------|
| Composite | 0.8634 |
| Answer accuracy | 0.8850 |
| Document accuracy | 0.8894 |
| Page proximity | 0.7942 |

**Per-domain breakdown:**

| Domain | Composite | Answer | Doc | Page |
|--------|-----------|--------|-----|------|
| Domain 1 (sports, 198 q) | 0.8730 | 0.8939 | 0.9040 | 0.8000 |
| Domain 2 (medical, 263 q) | 0.8562 | 0.8783 | 0.8783 | 0.7898 |

---

## 5. Evaluation Framework

The project adopted a **three-gate robustness protocol** after early false positives were identified (see §7).

### Splits
Source: `data/splits/`, generated by `scripts/00_create_splits.py`

| Split | File | N questions | Description |
|-------|------|-------------|-------------|
| fold_0_val | `data/splits/fold_0_val.csv` | 73 | 5-fold CV, grouped by Doc_ID |
| fold_1_val | `data/splits/fold_1_val.csv` | 78 | 5-fold CV, grouped by Doc_ID |
| lockbox | `data/splits/lockbox.csv` | 95 | Fixed 20% held-out document partition |
| full-dev | `data/dev_questions.csv` | 461 | Full development set |

**Critical design choice:** CV folds are **grouped by document identity** (Doc_ID), not by question identity. This prevents document-level memorization from inflating fold scores — if questions from the same document appear in both train and val, the system can memorize document content rather than learning generalizable retrieval.

### Promotion Gate (all three required before Kaggle submission)
1. Full-dev composite > v7_baseline (0.8634)
2. Grouped CV fold 0 > v7_baseline (0.8309)
3. Grouped CV fold 1 > v7_baseline (0.8718)
4. Lockbox > v7_baseline (0.8877)
5. Timing projection < 8.5 hours on T4×2

**Important:** Full-dev alone is insufficient. Early experiments showed candidates that beat v7 on full-dev by up to +0.018 still failed on fold1 and lockbox (see §7 false positives).

### Parity Check
- Tool: `scripts/check_kaggle_parity.py`
- Purpose: Verify that the shared runtime (`pipeline_shared.py`) produces identical predictions to the rendered Kaggle bundle
- Root cause that was fixed: the rendered bundle was executed from a bare temp directory; local repo-relative paths (to `data/`, `models/`, `local_packages/`) broke inside the bundle
- Fix: bundle-side runs now staged in a temp repo-shaped workspace with symlinks
- Status: Parity verified on `candidate_no_bleed_v1` smoke run — exact prediction match

---

## 6. Retrieval Experiments (Phase 1, March 20, 2026)

Source: `logs/checkpoint_1_retrieval.md`

All retrieval experiments evaluated on full dev (461 questions).

| System | Doc@1 | Doc@10 | Page@1 | Page@10 | Notes |
|--------|-------|--------|--------|---------|-------|
| Random | 0.056 | — | — | — | 1/18 docs expected |
| BM25 | 0.458 | 0.781 | 0.219 | 0.547 | Baseline lexical |
| **BGE-M3** | **0.933** | **0.994** | **0.482** | **0.796** | **Selected** |
| BM25 + BGE-M3 (RRF, k=60) | 0.659 | 0.991 | 0.343 | 0.831 | −27.4 pp Doc@1 vs BGE-M3 alone |
| ColSmol-500M | 0.039 | 0.074 | 0.000 | 0.000 | Near-zero |
| M3 + ColSmol (RRF) | 0.928 | — | 0.479 | — | ColSmol drags down top-1 |

**Key finding:** BGE-M3 dense-only is decisively best. BM25's weak Ukrainian representation corrupts RRF fusion — the fusion lowers Doc@1 by 27.4 pp. ColSmol-500M (a vision-language model) fails because FP16 normalization produces NaN query embeddings for Ukrainian Cyrillic; even after correcting the NaN bug, ColSmol achieves only 0.039 Doc@1, consistent with ColPali's training on Latin-script visual layouts rather than Cyrillic text.

**Query ablation (separate experiment):** Adding answer options to the BGE-M3 query hurts Doc@1 by 7.6 pp. Query is question text only.

---

## 7. MCQ and Prompt Experiments (Phase 2, March 20, 2026)

Source: `logs/checkpoint_2_results.md`

**Oracle retrieval MCQ accuracy** (val split, N=91, A30 GPU):
- MamayLM-Gemma-3-12B-IT greedy: **86.8%** — far exceeds random (16.7%)
- Speed: 1.54 questions/second on A30

**Prompt format ablation** (val split, N=91, oracle retrieval):

| Prompt | Accuracy | Domain 1 | Domain 2 | Notes |
|--------|----------|----------|----------|-------|
| V4: Evidence-first | 0.868 | 0.846 | 0.885 | Best in this ablation |
| V1: Direct | 0.846 | 0.821 | 0.865 | Used in v7 (outperforms at pipeline scale) |
| V2: Chain-of-thought | 0.143 | 0.179 | 0.115 | **Critical failure: near-random** |
| V5: Elimination | 0.143 | 0.179 | 0.115 | **Critical failure: near-random** |
| V3: Ukrainian instruction | 0.582 | 0.513 | 0.635 | Significant degradation |

**Critical negative finding:** Chain-of-thought (CoT) and elimination prompts collapse to ~14.3% accuracy — below random (16.7% for 6 options). This is attributed to Q4_K_M quantization degrading multi-step reasoning and insufficient context budget for a reasoning chain. The V1 direct prompt outperforms V4 evidence-first at full pipeline scale (0.867 vs. 0.856) because the longer V4 template competes with the limited context window.

**Context-size sweep** (100 questions, seed=42):
Source: `docs/paper/paper_analysis_report.md` §3

| k (pages) | Answer acc | Composite | Notes |
|-----------|------------|-----------|-------|
| 0 (closed-book) | 0.668 | 0.749 | No retrieval |
| 1 | 0.860 | 0.825 | Sharp gain |
| 2 | 0.920 | 0.856 | **v7 default** |
| 3 | 0.930 | 0.862 | +0.010 over k=2 |
| 5 | 0.890 | 0.841 | Degradation (context dilution) |

k=2 is at the performance knee. v7 uses k=2 pages as context for the LLM.

---

## 8. Robustness Experiments (March 28 — April 5, 2026)

Source: `logs/2026-03-28_robustness_recovery.md`, `logs/session_2026_03_29_part2.md`, `docs/paper/ablations/`, `docs/paper/tables/master_ablation_ledger.csv` (101 rows)

### 8.1 Reference Points (v7_baseline on splits)

| Split | N | Composite | Answer | Doc | Page |
|-------|---|-----------|--------|-----|------|
| fold_0 | 73 | 0.8309 | 0.8493 | 0.8767 | 0.7481 |
| fold_1 | 78 | 0.8718 | — | — | — |
| lockbox | 95 | 0.8877 | — | — | — |
| full_dev | 461 | 0.8634 | 0.8850 | 0.8894 | 0.7942 |

Source: `outputs/benchmarks/v7_baseline_fold0/summary.json`, `v7_baseline_fold1/summary.json`, `v7_baseline_lockbox/summary.json`, `v7_full_dev/summary.json`

### 8.2 Reranker Scaling

**Confounded historical experiment (DO NOT CITE AS EVIDENCE):**
- `candidate_stronger_reranker_v1`: Qwen3-4B at `max_length=1024` vs. v7_baseline's `max_length=2048`
- 50q result: 0.6553 (appears catastrophically worse than v7)
- **This is unfair.** The context budget was halved. This is NOT evidence that 4B rerankers are worse.
- Source: `docs/paper/ablations/methodology_audit.md`

**Fair reranker comparison (TRUSTED):**
- `candidate_stronger_reranker_v2_fair`: Qwen3-4B at same `max_length=2048` as v7
- Source: `outputs/benchmarks/candidate_stronger_reranker_v2_fair_*/summary.json`

| Split | v7 | 4B fair | Delta | Bootstrap CI |
|-------|----|---------|----|---|
| fold_0 | 0.8309 | 0.8484 | +0.0175 | — |
| fold_1 | 0.8718 | 0.9141 | +0.0423 | — |
| lockbox | 0.8877 | 0.9254 | +0.0377 | — |
| full_dev | 0.8634 | 0.8866 | **+0.0232** | **[+0.0058, +0.0407]** |

Bootstrap CI source: `docs/paper/paper_analysis_report.md` §8 (10,000 iterations, seed=42)

Per-component full-dev deltas (fair 4B vs. v7):
- Answer: +0.0043, CI [−0.0195, +0.0282] → **CI crosses zero (not significant)**
- Document: +0.0477, CI [+0.0174, +0.0781] → **Significant positive**
- Page proximity: +0.0366, CI [+0.0088, +0.0651] → **Significant positive**

The 4B reranker improvement is driven by better document selection and page localization, not better MCQ answering. This makes sense: the reranker directly determines document and page; the LLM only sees the selected content.

**Why 4B wasn't submitted:**
- Reranking time: v7 = 624.1s vs. 4B = 2130.9s (3.4× slower)
  - Source: `docs/paper/tables/runtime_table.md`
- Competition deadline prevented submission of the 4B system

**8B reranker:**
- `candidate_stronger_reranker_v3_8b_feasibility`: feasibility-only
- No successful full run exists (`docs/paper/ablations/reranker_8b_diagnosis.md`: "No 8B feasibility artifact was found")
- Status: pipeline-infeasible — GPU memory conflict when co-loading Qwen3-8B reranker + MamayLM-12B on a single A30

### 8.3 Dense Document Lock Variants

**Concept:** When the top-1 document from dense retrieval has a very high score AND the document is long (many pages), lock to that document without allowing the reranker to switch documents. Intended to reduce doc flip rate for long-document questions in Domain 1.

**DDL v1** (threshold unknown): 
- fold_0: 0.8183 (−0.0126 vs v7) → **Clear regression**

**DDL v2** (long-doc threshold: ≥25 pages):
| Split | Composite | Delta |
|-------|-----------|-------|
| fold_0 | 0.8502 | +0.0193 |
| fold_1 | 0.8983 | +0.0265 |
| lockbox | 0.8880 | +0.0003 |
| full_dev | 0.8811 | +0.0177 |

- Kaggle (v11): public = 0.8320 (−0.0368), private = 0.8410 (−0.0312) → **Failed to generalize**

**DDL v3** (threshold: ≥27 pages, refinement of v2):
| Split | Composite | Delta |
|-------|-----------|-------|
| fold_0 | 0.8502 | +0.0193 |
| fold_1 | 0.9025 | +0.0307 |
| lockbox | 0.8929 | +0.0053 |
| full_dev | 0.8812 | +0.0178 |

Bootstrap CI (full_dev delta): [+0.0042, +0.0315] — **nominally significant**
- Kaggle (v14): public = 0.8320 (−0.0368), private = 0.8396 (−0.0326) → **Failed to generalize**

**DDL v3 exhaustive (all pages)**:
- fold_0: 0.8451 (+0.0142) — **weaker than DDL v3** (0.8502)
- Conclusion: Reranker's in-document page ordering is better than exhaustive brute-force search

**Why DDL failed:** The heuristic targets long-document proxies (≥25–27 pages). Domain 1 has 11 documents averaging 78.6 pages — very long. Domain 2 has 30 documents averaging 8.5 pages — very short. The hidden Domain 3's document length distribution is unknown, but the Kaggle regression strongly suggests Domain 3 docs are short/medium length, making the DDL heuristic harmful rather than helpful. Verified in: `docs/paper/ablations/why_v14_failed.md`

**DDL risk signals (v14 vs v7):**
- Doc flip rate: 6.94% (v7: 0%)
- Page change rate: 16.7%
- Answer change rate: 5.0%
- Per-domain delta: Domain 1 +0.035, Domain 2 +0.005 → Domain 1 heavily biased
Source: `docs/paper/ablations/why_v14_failed.md`

### 8.4 Dense Margin Lock

**Concept:** Lock to top document when the retrieval score gap (margin) between top-1 and top-2 document exceeds a threshold (0.25), indicating high-confidence retrieval.

| Split | v7 | DML | Delta |
|-------|----|----|-------|
| fold_0 | 0.8309 | 0.8309 | **+0.0000** |
| fold_1 | 0.8718 | 0.8718 | **+0.0000** |
| lockbox | 0.8877 | 0.8877 | **+0.0000** |
| full_dev | 0.8634 | 0.8611 | −0.0023 |

Threshold sweeps (0.20 and 0.30) on fold_1: both tied exactly with v7 (0.8718).
**Verdict: Near-neutral. Not promotable.**

### 8.5 Structure-Aware Chunking

**Concept:** Instead of using full pages as retrieval units, split pages into 700-character semantic chunks (120-char overlap, min 180 chars). Intended to improve granularity for long-document retrieval.

**Structure chunks v1** (baseline chunking):
- fold_0: 0.8149 (−0.0160 vs v7) → **Negative on fold 0**
- full_dev: 0.8675 (+0.0041) → misleadingly positive on full-dev

**Structure chunks v2 with doc guard** (adds domain-level gating: only activate on Domain 1-like docs):
- fold_0: 0.8210 (−0.0099 vs v7) → **Negative on fold 0**
- full_dev: 0.8817 (+0.0183) → strong-looking full-dev win
- Kaggle (structure_doc_guard): public = 0.8415, private = 0.8600 → **Failed to generalize**
- The fold_0 result (negative) correctly predicted the generalization failure

**Structure chunks v3** (sc_v3, debug iterations):
- sc_v3_fold0: 0.8012 (−0.0297) → very negative on first attempt
- sc_v3_fold0_r2: 0.8195 (−0.0114) → improved with fixes
- sc_v3_fold0_r3: 0.8309 (+0.0000) → tied with v7
- sc_v3_fold1: 0.8718 (+0.0000) → tied
- sc_v3_lockbox: 0.8827 (−0.0049) → slight regression
**Verdict: Not promotable.**

### 8.6 Prompt Refocus (v5 refocus)

**Concept:** Modify the LLM prompt to direct attention more explicitly to evidence in the retrieved text before answering.

**v5_refocus (override experiments):**
- v5refocus_fold0: 0.8325 (+0.0016)
- v5refocus_fold1: 0.8754 (+0.0036)
- v5refocus_lockbox: 0.9033 (+0.0156) ← promising

**v5_refocus_v1 (named preset, full_dev):**
- full_dev: 0.8637 (+0.0003, CI [−0.0141, +0.0150]) → **within noise**
- Kaggle (v13): public = 0.8231 (−0.0457), private = 0.8441 (−0.0281) → **Failed badly**

**v5_refocus + 4B reranker combo:**
- fold_0: 0.8484 (+0.0175)
- fold_1: 0.9305 (+0.0587) ← very strong
- No lockbox or full_dev run available for this combo

**v13 risk signals:**
- Doc flip rate: 6.29% (vs v7: 0%)
- Page change rate: 16.49%
- Per-domain delta: Domain 1 −0.0155, Domain 2 +0.0123 → imbalanced
- Full-dev CI crosses zero → was never a decisive win
Source: `docs/paper/ablations/why_v13_failed.md`

**Interpretation:** The prompt change is a locality effect on the visible domain distribution, not a robust task improvement.

### 8.7 False Positive: Neighbor Bleed Removal (no_bleed_v1)

**Concept:** The standard pipeline includes "neighbor bleed" — context from adjacent pages is included alongside the retrieved page to improve coverage. `candidate_no_bleed_v1` disables this.

| Split | v7 | no_bleed | Delta |
|-------|----|---------|----|
| fold_0 (73 q) | 0.8309 | **0.8577** | **+0.0268** |
| fold_1 (78 q) | 0.8718 | 0.8622 | −0.0096 |
| lockbox (95 q) | 0.8877 | 0.8575 | −0.0302 |

**The fold-0 win (+0.027) is the strongest offline win observed in the entire project.** Yet it is a false positive: fold-1 and lockbox both regressed. The fold-0 win reflects noise amplification on 73 questions, not a robust signal.

This candidate was correctly caught by the multi-split gating protocol and was NOT submitted to Kaggle. Without the three-split protocol, this would have been a confident submission that would have scored ~0.84 publicly.

### 8.8 Hybrid Retrieval

**BM25 + BGE-M3 RRF (fold_0):**
- fold_0: 0.8210 (−0.0098 vs v7)
- Consistent with Phase 1 finding: BM25 hurts rather than helps

### 8.9 Extraction Blocks

**Concept:** Extract text in block-reading order rather than raw page flow, intended to improve context coherence.
- fold_0: 0.8175 (−0.0134) → **Clear regression; also significantly slower**

### 8.10 Locked Top Doc

**Concept:** Instead of reranking across documents, lock to the top-1 document from dense retrieval and search all pages within that document exhaustively.

| Split | v7 | Locked | Delta |
|-------|----|----|---|
| fold_0 | 0.8309 | 0.8320 | +0.0011 |
| fold_1 | 0.8718 | 0.8723 | +0.0005 |
| lockbox | 0.8877 | 0.8874 | −0.0003 |

**Verdict:** Essentially neutral (near-tie on all splits). Page-only movement with almost no doc flips. Not promotable but the closest near-miss among document-selection variants.

### 8.11 Chat Template

**Early experiment (fold_1, initial pipeline version):**
- With chat template on fold_1: −0.028 vs. v7

**Later named-preset experiments (corrected pipeline version):**

| Preset | fold_0 | fold_0 delta | fold_1 | fold_1 delta |
|--------|--------|-------------|--------|-------------|
| chat_tmpl_v1 | 0.8502 | +0.0193 | 0.8741 | +0.0023 |
| chat_tmpl_v2 | 0.8518 | +0.0209 | 0.8933 | +0.0215 |

The sign flip (early: negative, later: positive) is attributed to the base preset having changed between experiments. The earlier −0.028 result was on a different pipeline configuration. The later positive results remain untested on Kaggle and are treated as exploratory. **v7 does not use chat template.**

### 8.12 LapaLLM

**Concept:** Alternative Ukrainian LLM (LapaLLM) as a replacement for MamayLM.
- fold_0: 0.8430 (+0.0121 vs. v7 on fold_0)
- Introduces different error patterns; does not consistently outperform v7's multi-split robustness
- **Not tested on fold_1, lockbox; not promotable**

### 8.13 Pass Count Variants

**Varying voting passes (override experiments on fold_0):**
- 1-pass: 0.8309 (identical to 3-pass on fold_0)
- 3-pass: 0.8309 (v7 default)
- 5-pass: 0.8309 (identical)
- 7-pass: 0.8309 (identical)

**On fold_1:**
- 1-pass: 0.8690 (−0.0028 vs. v7 3-pass = 0.8718)

**Conclusion:** 3-pass is the conservative default. Single-pass achieves near-equivalent quality on fold_0 but is marginally weaker on fold_1. The cost of 3-pass over 1-pass: approximately 2× additional LLM inference time.

### 8.14 Additional Supplementary Candidates

| Candidate | Split | Composite | Delta vs v7 | Verdict |
|-----------|-------|-----------|-------------|---------|
| candidate_avir_threshold_v1 | fold_0 | 0.8308 | −0.0001 | Neutral |
| candidate_avir_threshold_v1 | full_dev | 0.8663 | +0.0029 | Not promotable |
| candidate_confidence_gated_v1 | fold_0 | 0.8164 | −0.0145 | Negative |
| candidate_confidence_gated_v1 | full_dev | 0.8617 | −0.0017 | Negative |
| candidate_evidence_first_v1 | fold_0 | 0.8308 | −0.0001 | Neutral |
| candidate_page_summary_selector_v1 | fold_0 | 0.8301 | −0.0008 | Neutral |

---

## 9. Error Analysis (v7, Full Dev)

Source: `docs/paper/paper_analysis_report.md` §4

### Error Category Breakdown (N=461)

| Category | N | % |
|----------|---|---|
| All correct (answer + doc + page) | 205 | 44.5% |
| Correct answer, right doc, wrong page | 169 | 36.7% |
| Wrong answer, right doc, wrong page | 39 | 8.5% |
| Correct answer, wrong doc | 21 | 4.6% |
| Wrong answer, right page | 17 | 3.7% |
| Wrong answer, wrong doc | 10 | 2.2% |

**Dominant failure mode (36.7%):** Correct answer with wrong page. The system found the right document, ranked the wrong page first, but MamayLM answered correctly because the true evidence appeared in the 2-page context window. This is a near-miss, not a retrieval failure.

**Page proximity resilience:** The 37-point gap between page accuracy (48.2%) and answer accuracy (88.5%) demonstrates that feeding multi-page context to the LLM provides substantial resilience against page-level retrieval errors.

### Wrong-Page Distance Distribution (correct-doc, wrong-page cases)

The distance analysis was run on **181 correct-doc/wrong-page cases** (from `paper_analysis_report.md` §4). Note: the error breakdown table above shows 169 + 39 = 208 right-doc wrong-page cases; the 181 figure in the analysis script uses a slightly different definition (e.g., excluding edge cases where predicted page equals ground truth page but doc definition differs). The 55.8% within-3-pages figure and median=2.0 are computed over the 181-case set and are what is cited in the paper.

Source: `docs/paper/paper_analysis_report.md` §4

| Distance threshold | Count | % of wrong-page cases |
|-------------------|-------|----------------------|
| ≤ 1 page | 71 | 39.2% |
| ≤ 2 pages | 91 | 50.3% |
| ≤ 3 pages | 101 | 55.8% |
| ≤ 5 pages | 115 | 63.5% |
| ≤ 10 pages | 137 | 75.7% |
| > 10 pages | 44 | 24.3% |
| **Median distance** | **2.0** | |
| **Mean distance** | **11.2** | |

### Per-Domain Page Distance Breakdown

| Domain | n wrong-page | Median dist | Within-3-pages |
|--------|-------------|-------------|----------------|
| Domain 1 (sports, long docs) | 102 | **7.0** | **40.2%** |
| Domain 2 (medical, short docs) | 79 | **1.0** | **75.9%** |

**Critical insight:** Domain 1 (long docs, avg 78.6 pages) has dramatically harder page localization (median 7 pages off) compared to Domain 2 (short docs, avg 8.5 pages, median 1 page off). Long-document heuristics (DDL, structure chunks) that improve Domain 1 page retrieval show large local gains but are tuned to a distributional property (long docs) that may not transfer to the hidden Domain 3.

### Document-Level Analysis
- Wrong document rate: (21+10)/461 = 6.7%, consistent with BGE-M3 Doc@1 = 0.933
- Domain 1 retrieval: Doc@1 = 0.975 (11 distinctive docs → easy to identify), but Page@1 = 0.409 (78 pages on average → hard to pinpoint)
- Domain 2 retrieval: Doc@1 = 0.901, Page@1 = 0.536

Verification: weighted average Doc@1 = (198×0.975 + 263×0.901)/461 = 0.932 ≈ 0.933 ✓
Verification: weighted average Page@1 = (198×0.409 + 263×0.536)/461 = 0.481 ≈ 0.482 ✓

---

## 10. Generalization Gap Analysis

The central finding of this project is that **every offline improvement regressed on Kaggle**.

### Summary of Gap

| System | Local full-dev | Local delta | Kaggle public | Public delta | Kaggle private | Private delta |
|--------|---------------|-------------|--------------|-------------|----------------|---------------|
| v7 | 0.8634 | 0.0000 | 0.8688 | +0.0054 | 0.8722 | +0.0088 |
| v13 | 0.8637 | +0.0003 | 0.8231 | −0.0457 | 0.8441 | −0.0281 |
| v14 | 0.8812 | +0.0178 | 0.8320 | −0.0368 | 0.8396 | −0.0326 |

Source: `docs/paper/tables/local_vs_kaggle_mismatch.md`

Note: v7 is the **only** evaluated submission where Kaggle public > local full-dev. All locally improved systems regressed.

### Three Predictive Risk Signals

**1. High doc flip rate:** When a candidate changes which document is selected for >5% of questions (vs. v7), it indicates fragile routing that breaks under domain shift.
- v13 doc flip rate: 6.29%
- v14 doc flip rate: 6.94%
- v7 doc flip rate: 0% (by definition, as the reference)

**2. Per-domain imbalance:** Local gains concentrated in one domain are unreliable when the private test has an unseen third domain.
- v13: Domain 1 −0.016, Domain 2 +0.012 → strongly imbalanced
- v14: Domain 1 +0.035, Domain 2 +0.005 → strongly biased to long docs

**3. Long-document proxy dependence:** Heuristics that target long-document properties (DDL ≥25 pages, structure chunks, exhaustive search within top doc) overfit to Domain 1's distributional property and fail when test documents don't match that profile.

### v9 as Negative Control

v9 bundled six simultaneous changes, including switching from page-level to doc-level reranking. The result: public 0.6716 (−0.1972 vs v7), private 0.7265 (−0.1457). The majority of regressions followed the doc-level rerank change — a single change in routing behavior that destabilized 165 previously-correct questions. This demonstrates why compound changes are dangerous and why changes should be evaluated one at a time.

---

## 11. Paper Status

**File:** `paper/unlp2026_paper.tex`  
**Compiled:** `paper/unlp2026_paper.pdf`  
**Last updated:** 2026-04-07  
**Format:** ACL Short Paper (4 pages content + unlimited references)  
**Compilation:** pdflatex + bibtex, TeX Live 2022. Load `module load texlive/20220321` before compiling.

**Compile command (run from `paper/` directory):**
```bash
pdflatex -interaction=nonstopmode unlp2026_paper.tex && \
bibtex unlp2026_paper && \
pdflatex -interaction=nonstopmode unlp2026_paper.tex && \
pdflatex -interaction=nonstopmode unlp2026_paper.tex
```

**Output:** 5 pages (content: pages 1–4, references: pages 4–5). References do not count toward the 4-page content limit.

**Known compilation warning:** Overfull \hbox of 1.2pt at lines 202–203 (TikZ figure caption). This is negligible (< half a millimeter) and does not affect layout.

### Paper Sections and Claims

| Section | Claims | Verified? |
|---------|--------|-----------|
| Abstract | Private 0.8722, rank 7/15, +21.7 pp retrieval gain | ✓ |
| §2 Task | 461 questions, Domain 1 (11 docs, 865 pp, 78.6/doc), Domain 2 (30 docs, 256 pp, 8.5/doc) | ✓ |
| §3 System | BGE-M3, Qwen3-0.6B, MamayLM Q4_K_M, 3-pass voting, 30 min A30 benchmark | ✓ |
| §4 Retrieval | Doc@1=0.933, BM25=0.458, RRF=0.659, ColSmol=0.039 | ✓ |
| §5 Main Results | v7 0.8634, fair_4B 0.8866, CI [+0.006,+0.041] | ✓ |
| §6 Prompt ablations | CoT 14.3%, k=2 best, LapaLLM +0.012, 1-pass −0.003 fold_1 | ✓ |
| §7 Generalization gap | Figure with 6 data points, no_bleed false positive | ✓ |
| §8 Error analysis | 36.7% correct-answer-wrong-page, domain page distances | ✓ |
| Conclusion | 4B reranker +0.023, 36.7% page-bottleneck | ✓ |

**Unverified claims (in paper, not found in project files):**
- "the winning team scored 0.9598 privately" — CLAUDE.md records public leader as 0.9377; 0.9598 private score not independently confirmed from a project file
- "comprised 15 teams" — not independently confirmed from a project file

### Citations Added (2026-04-07 update)

| Key | Description |
|-----|-------------|
| unlp2026task | UNLP 2026 Shared Task (placeholder — update with final proceedings) |
| chen2024bge | BGE-M3, Chen et al. ACL 2024 |
| faysse2024colpali | ColPali, Faysse et al. EMNLP 2024 |
| qwen3_2025 | Qwen3 Technical Report, Qwen Team 2025 |
| mamaylm2025 | MamayLM (INSAIT Institute, 2025) — fixed from 2024 |
| llamacpp | llama.cpp, Gerganov 2023 |
| johnson2019billion | FAISS, Johnson et al. IEEE Transactions on Big Data 2019 |
| robertson2009probabilistic | BM25, Robertson & Zaragoza 2009 |
| luan2021sparse | Sparse dense retrieval, Luan et al. TACL 2021 |
| cormack2009reciprocal | RRF, Cormack et al. SIGIR 2009 |

---

## 12. Evidence Trust Hierarchy

| Level | Description | Example |
|-------|-------------|---------|
| `trusted_public` | Kaggle official score | v7 public 0.8688 |
| `trusted_offline` | Named preset, run manifest in `benchmark_manifest.json` | fair_4B fold_1 0.9141 |
| `traceable_override` | Override JSON preserved in manifest; reproducible | 1pass_fold0 0.8309 |
| `historical_only` | Pre-audit; comparison not fair | stronger_reranker_v1 at max_length=1024 |

**Key rule (from `docs/paper/ablations/methodology_audit.md`):** Only `trusted_offline` or `trusted_public` evidence should be cited in the paper as supporting claims. Historical confounded experiments should only appear as methodology warnings.

---

## 13. Infrastructure and Environment

### Compute
- **Benchmark runs:** `gilbreth-b000` (A30 GPU, 24 GB VRAM) — interactive compute node
- **Kaggle submission target:** T4×2 (two T4 GPUs, ~32 GB total VRAM shared)
- **WARNING:** Do NOT run benchmark jobs on the Gilbreth frontend host — causes `llama_cpp` native crashes

### Python Environment
- Python 3.9
- `local_packages/` directory is on `sys.path` at runtime (pinned wheels, offline Kaggle-compatible)
- `HF_HOME` defaults to `cache/hf_cache/` (pre-downloaded model weights)
- Key models in `models/` and/or `kaggle_datasets/`

### Key Files

| File | Purpose |
|------|---------|
| `notebooks/pipeline_shared.py` | All pipeline logic (~34,000 lines) |
| `notebooks/pipeline_presets.json` | 34 named preset configs |
| `scripts/benchmark_candidate.py` | Main benchmark runner |
| `scripts/00_eval_harness.py` | Metric computation |
| `scripts/00_create_splits.py` | CV fold + lockbox generation |
| `scripts/check_kaggle_parity.py` | Verify shared runtime matches Kaggle bundle |
| `scripts/diff_benchmark_runs.py` | Per-question diff between two runs |
| `docs/paper/paper_analysis_report.md` | Computed analysis report (authoritative numbers source) |
| `docs/paper/tables/master_ablation_ledger.csv` | 101-row comprehensive ablation table |
| `docs/paper/tables/runtime_table.md` | Timing data for all systems |
| `docs/paper/ablations/methodology_audit.md` | Fairness and interpretation flags |
| `paper/unlp2026_paper.tex` | Final competition paper |
| `paper/unlp2026_paper.pdf` | Compiled paper |

### Benchmark Commands

```bash
# Full dev benchmark
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --questions data/dev_questions.csv \
  --output-dir outputs/benchmarks/v7_baseline

# Fold evaluation
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --split fold0 \
  --output-dir outputs/benchmarks/v7_baseline_fold0

# Smoke run (10 questions)
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --n-questions 10 \
  --output-dir outputs/benchmarks/smoke
```

---

## 14. What Was Not Done / Open Questions

1. **Domain 3 analysis:** The hidden third domain is not analyzable with available data. Its document length distribution, subject matter, and linguistic characteristics are unknown.

2. **8B reranker full test:** Qwen3-8B reranker was planned as a feasibility experiment but no complete run exists. The concern is GPU memory — co-loading 8B reranker + MamayLM-12B (6.8 GB) on a single A30 may exceed 24 GB VRAM.

3. **v5refocus + 4B combo on lockbox/full_dev:** The combo of v5refocus prompt + 4B reranker was only run on fold_0 (+0.0175) and fold_1 (+0.0587). No lockbox or full_dev result exists for this system.

4. **Post-competition analysis of private test set:** No access to private test questions or ground truth; the local-vs-Kaggle gap in Domain 3 is mechanistically unconfirmable.

5. **MamayLM without quantization:** All experiments used Q4_K_M quantization (6.8 GB). A full-precision or Q8 version might recover reasoning capabilities for CoT, but was not tested due to VRAM constraints.

---

## 15. Key Lessons

1. **Full-dev alone is not a reliable promotion gate.** Candidates can overfit to the full-dev distribution (461 questions, 2 domains) in ways that fail under domain shift. Multi-split gating (fold_0 + fold_1 + lockbox) is necessary.

2. **The strongest offline win can be a false positive.** `no_bleed_v1` had the best fold_0 delta (+0.027) of any candidate; it regressed −0.030 on lockbox. Without the three-split protocol, it would have been submitted.

3. **Compound changes are dangerous.** v9's 0.67 public score resulted from six simultaneous changes. Individual ablations are mandatory.

4. **Long-document heuristics overfit to visible domains.** Any improvement that targets long-document properties (DDL, structure chunks) is suspect when the hidden test domain may have shorter documents.

5. **Context budget matters as much as model size.** The confounded 4B reranker experiment (max_length=1024 vs v7's 2048) showed a catastrophic regression that was actually caused by halving context, not by the larger model.

6. **BGE-M3 dense-only is the right retrieval foundation.** Any fusion (BM25, ColSmol) hurts. The dense embedding already captures Ukrainian semantics well; lexical and visual signals add noise.

7. **Conservative systems generalize better under hidden domain shift.** v7 (no doc locking, standard page reranking, simple direct prompt) is the most robust system across public and private Kaggle. The pattern across all submissions: greater local improvement → greater Kaggle regression.

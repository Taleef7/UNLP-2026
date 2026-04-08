"""
Quick test: BGE-M3 sparse/hybrid vs dense-only retrieval on dev set.
Decision rule: include hybrid in v7 only if hybrid Page@1 >= dense Page@1 + 0.01

Usage: python3 scripts/01_bge_hybrid_test.py
Output: prints Doc@1, Page@1 for dense, sparse, colbert, and hybrid (RRF)
"""
import sys, csv, json
from pathlib import Path
from collections import defaultdict

BASE = Path('/scratch/gilbreth/tamst01/unlp2026')
LOCAL = str(BASE / 'local_packages')
if LOCAL not in sys.path: sys.path.insert(0, LOCAL)

import numpy as np
import fitz

# ── Load questions ────────────────────────────────────────────────────────────
with open(BASE / 'data/dev_questions.csv', 'r', encoding='utf-8') as f:
    questions = list(csv.DictReader(f))
print(f'Questions: {len(questions)}')

# ── Extract page texts ────────────────────────────────────────────────────────
pages = []
for domain_dir in sorted((BASE / 'data/raw_pdfs').iterdir()):
    if not domain_dir.is_dir(): continue
    for pdf_path in sorted(domain_dir.glob('*.pdf')):
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            pages.append({'doc_id': pdf_path.name, 'page_num': i+1, 'text': page.get_text()})
        doc.close()
print(f'Pages: {len(pages)}')

# ── Load BGEM3FlagModel ───────────────────────────────────────────────────────
from FlagEmbedding import BGEM3FlagModel

print('Loading BGE-M3 with FlagEmbedding (all modes)...')
model = BGEM3FlagModel(
    str(BASE / 'kaggle_datasets/bge-m3'),
    use_fp16=True,
    device='cuda'
)

# ── Encode pages ──────────────────────────────────────────────────────────────
page_texts = [p['text'] for p in pages]
print(f'Encoding {len(page_texts)} pages (dense + sparse + colbert)...')
page_out = model.encode(
    page_texts,
    batch_size=8,
    max_length=1536,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
page_dense  = page_out['dense_vecs']          # (N, 1024) float32
page_sparse = page_out['lexical_weights']     # list of dicts
page_colbert = page_out['colbert_vecs']       # list of (seq_len, 1024) arrays

# ── Encode queries ────────────────────────────────────────────────────────────
query_texts = [q['Question'] for q in questions]
print(f'Encoding {len(query_texts)} queries...')
q_out = model.encode(
    query_texts,
    batch_size=32,
    max_length=256,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
q_dense   = q_out['dense_vecs']
q_sparse  = q_out['lexical_weights']
q_colbert = q_out['colbert_vecs']

TOP_K = 10

# ── Dense rankings ────────────────────────────────────────────────────────────
print('Computing dense rankings...')
dense_scores = q_dense @ page_dense.T           # (Q, P)
dense_ranks  = np.argsort(-dense_scores, axis=1)[:, :TOP_K]

# ── Sparse rankings ───────────────────────────────────────────────────────────
print('Computing sparse rankings...')
sparse_scores = np.zeros((len(questions), len(pages)), dtype=np.float32)
for qi, qsparse in enumerate(q_sparse):
    for qterm, qweight in qsparse.items():
        for pi, psparse in enumerate(page_sparse):
            if qterm in psparse:
                sparse_scores[qi, pi] += float(qweight) * float(psparse[qterm])
sparse_ranks = np.argsort(-sparse_scores, axis=1)[:, :TOP_K]

# ── ColBERT rankings ──────────────────────────────────────────────────────────
print('Computing ColBERT rankings...')
colbert_scores = np.zeros((len(questions), len(pages)), dtype=np.float32)
for qi, qvec in enumerate(q_colbert):
    # MaxSim: for each query token, max similarity over doc tokens, then mean
    qvec_arr = np.array(qvec, dtype=np.float32)   # (q_len, dim)
    for pi, pvec in enumerate(page_colbert):
        pvec_arr = np.array(pvec, dtype=np.float32)  # (p_len, dim)
        sim = qvec_arr @ pvec_arr.T                   # (q_len, p_len)
        colbert_scores[qi, pi] = float(sim.max(axis=1).mean())
colbert_ranks = np.argsort(-colbert_scores, axis=1)[:, :TOP_K]

# ── RRF fusion ────────────────────────────────────────────────────────────────
K_RRF = 60
print('Computing RRF hybrid rankings...')
rrf_scores = np.zeros((len(questions), len(pages)), dtype=np.float64)
for qi in range(len(questions)):
    for ri, pi in enumerate(dense_ranks[qi]):
        rrf_scores[qi, pi]  += 1.0 / (K_RRF + ri + 1)
    for ri, pi in enumerate(sparse_ranks[qi]):
        rrf_scores[qi, pi]  += 1.0 / (K_RRF + ri + 1)
    for ri, pi in enumerate(colbert_ranks[qi]):
        rrf_scores[qi, pi]  += 1.0 / (K_RRF + ri + 1)
hybrid_ranks = np.argsort(-rrf_scores, axis=1)[:, :TOP_K]

# ── Dense+Sparse only (no colbert) ────────────────────────────────────────────
ds_scores = np.zeros((len(questions), len(pages)), dtype=np.float64)
for qi in range(len(questions)):
    for ri, pi in enumerate(dense_ranks[qi]):
        ds_scores[qi, pi] += 1.0 / (K_RRF + ri + 1)
    for ri, pi in enumerate(sparse_ranks[qi]):
        ds_scores[qi, pi] += 1.0 / (K_RRF + ri + 1)
ds_ranks = np.argsort(-ds_scores, axis=1)[:, :TOP_K]

# ── Evaluate all modes ────────────────────────────────────────────────────────
def evaluate(ranks_matrix, label):
    doc_at_1, page_at_1 = 0, 0
    for qi, q in enumerate(questions):
        top_pages = [pages[pi] for pi in ranks_matrix[qi]]
        pred_doc, pred_page = top_pages[0]['doc_id'], top_pages[0]['page_num']
        if pred_doc == q['Doc_ID']:
            doc_at_1 += 1
            if pred_page == int(q['Page_Num']):
                page_at_1 += 1
    d1 = doc_at_1 / len(questions)
    p1 = page_at_1 / len(questions)
    print(f'{label:30s}  Doc@1={d1:.4f}  Page@1={p1:.4f}')
    return d1, p1

print('\n── Retrieval Results (N={}) ─────────────────────────────'.format(len(questions)))
d_dense,    p_dense    = evaluate(dense_ranks,   'Dense only (baseline)')
d_sparse,   p_sparse   = evaluate(sparse_ranks,  'Sparse only')
d_colbert,  p_colbert  = evaluate(colbert_ranks, 'ColBERT only')
d_ds,       p_ds       = evaluate(ds_ranks,      'Dense+Sparse RRF')
d_hybrid,   p_hybrid   = evaluate(hybrid_ranks,  'Dense+Sparse+ColBERT RRF')
print(f'\nDecision: {"USE HYBRID" if p_hybrid >= p_dense + 0.01 else "KEEP DENSE-ONLY"} '
      f'(hybrid Page@1={p_hybrid:.4f} vs dense Page@1={p_dense:.4f}, '
      f'delta={p_hybrid-p_dense:+.4f})')

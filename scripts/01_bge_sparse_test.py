"""
Fast test: BGE-M3 sparse vs dense-only retrieval on dev set.
Skips ColBERT (too slow). Tests dense, sparse, and dense+sparse RRF.

Usage: python3 scripts/01_bge_sparse_test.py
"""
import sys, csv, json
from pathlib import Path

BASE = Path('/scratch/gilbreth/tamst01/unlp2026')
LOCAL = str(BASE / 'local_packages')
if LOCAL not in sys.path: sys.path.insert(0, LOCAL)

import numpy as np
import fitz
import time

t0 = time.time()

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
print(f'Pages: {len(pages)} [{time.time()-t0:.1f}s]')

# ── Load BGE-M3 (dense + sparse ONLY, no colbert) ────────────────────────────
from FlagEmbedding import BGEM3FlagModel
print('Loading BGE-M3...')
model = BGEM3FlagModel(
    str(BASE / 'kaggle_datasets/bge-m3'),
    use_fp16=True, device='cuda'
)

# ── Encode pages ──────────────────────────────────────────────────────────────
page_texts = [p['text'] for p in pages]
print(f'Encoding {len(page_texts)} pages (dense+sparse)...')
page_out = model.encode(
    page_texts, batch_size=8, max_length=1536,
    return_dense=True, return_sparse=True, return_colbert_vecs=False,
)
page_dense  = page_out['dense_vecs']
page_sparse = page_out['lexical_weights']
print(f'Pages encoded [{time.time()-t0:.1f}s]')

# ── Encode queries (dense+sparse) ────────────────────────────────────────────
query_texts_base = [q['Question'] for q in questions]
query_texts_full = [
    f"{q['Question']} " + ' '.join(f"{l}: {q[l]}" for l in 'ABCDEF' if q.get(l,'').strip())
    for q in questions
]

print(f'Encoding queries (base + full w/ options)...')
q_base = model.encode(query_texts_base, batch_size=32, max_length=256,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
q_full = model.encode(query_texts_full, batch_size=32, max_length=512,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False)
print(f'Queries encoded [{time.time()-t0:.1f}s]')

TOP_K = 10
K_RRF = 60

def get_dense_ranks(q_dense_arr):
    scores = q_dense_arr @ page_dense.T
    return np.argsort(-scores, axis=1)[:, :TOP_K], scores

def get_sparse_ranks(q_sparse_list):
    sp_scores = np.zeros((len(q_sparse_list), len(pages)), dtype=np.float32)
    for qi, qsparse in enumerate(q_sparse_list):
        for qterm, qweight in qsparse.items():
            for pi, psparse in enumerate(page_sparse):
                if qterm in psparse:
                    sp_scores[qi, pi] += float(qweight) * float(psparse[qterm])
    return np.argsort(-sp_scores, axis=1)[:, :TOP_K], sp_scores

def rrf_combine(ranks_a, ranks_b):
    rrf = np.zeros((len(ranks_a), len(pages)), dtype=np.float64)
    for qi in range(len(ranks_a)):
        for ri, pi in enumerate(ranks_a[qi]): rrf[qi, pi] += 1.0 / (K_RRF + ri + 1)
        for ri, pi in enumerate(ranks_b[qi]): rrf[qi, pi] += 1.0 / (K_RRF + ri + 1)
    return np.argsort(-rrf, axis=1)[:, :TOP_K]

def evaluate(ranks_matrix, label):
    doc_at_1 = page_at_1 = 0
    for qi, q in enumerate(questions):
        top = pages[ranks_matrix[qi][0]]
        if top['doc_id'] == q['Doc_ID']:
            doc_at_1 += 1
            if top['page_num'] == int(q['Page_Num']):
                page_at_1 += 1
    d1 = doc_at_1 / len(questions)
    p1 = page_at_1 / len(questions)
    print(f'{label:45s}  Doc@1={d1:.4f}  Page@1={p1:.4f}')
    return d1, p1

print(f'\nComputing dense rankings...')
dense_base_ranks, _ = get_dense_ranks(q_base['dense_vecs'])
dense_full_ranks, _ = get_dense_ranks(q_full['dense_vecs'])
print(f'Dense done [{time.time()-t0:.1f}s]')

print(f'Computing sparse rankings (may take ~2min)...')
sparse_base_ranks, _ = get_sparse_ranks(q_base['lexical_weights'])
sparse_full_ranks, _ = get_sparse_ranks(q_full['lexical_weights'])
print(f'Sparse done [{time.time()-t0:.1f}s]')

hybrid_base_ranks = rrf_combine(dense_base_ranks, sparse_base_ranks)
hybrid_full_ranks = rrf_combine(dense_full_ranks, sparse_full_ranks)
print(f'RRF done [{time.time()-t0:.1f}s]')

print(f'\n── BGE-M3 Retrieval Results (N={len(questions)}) ──────────────────────────')
evaluate(dense_base_ranks,  'Dense, base query (Q only)')
evaluate(dense_full_ranks,  'Dense, full query (Q + options)')
evaluate(sparse_base_ranks, 'Sparse, base query')
evaluate(sparse_full_ranks, 'Sparse, full query')
evaluate(hybrid_base_ranks, 'Dense+Sparse RRF, base query')
evaluate(hybrid_full_ranks, 'Dense+Sparse RRF, full query')

print(f'\nBaseline (v6): Doc@1=0.9328  Page@1=0.4816')
print(f'Total time: {time.time()-t0:.1f}s')

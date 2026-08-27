#!/usr/bin/env python3
"""Vectorized BM25 (scipy sparse) with tunable k1/b.

Replaces `rank_bm25.BM25Okapi`, which we used at library defaults (k1=1.5, b=0.75) and never
tuned. That was a strawman: every "sparse fusion damages dense" conclusion in the first round was
confounded with "our sparse arm is misconfigured". It is also pure Python and
O(n_docs) per query per term, which does not survive a 200k-document corpus.

Exact Okapi BM25:

    score(q, d) = sum_{t in q}  idf(t) * ( tf(t,d) * (k1 + 1) )
                                       / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )        [Lucene's non-negative variant]

We precompute the weighted document-term matrix W (same sparsity as the count matrix), so scoring a
query batch is one sparse matmul. Rebuilding W for a new (k1, b) costs O(nnz), which makes a grid
search cheap -- so k1/b can be TUNED on a query split that is disjoint from the evaluation split.
"""
import numpy as np
from scipy import sparse as sp


class SparseBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.vocab = None
        self._tf = None      # raw count matrix (docs x terms), CSR
        self._idf = None
        self._dl = None
        self._avgdl = None
        self.W = None        # weighted doc-term matrix for the current (k1, b)

    def fit(self, docs_tokens):
        """docs_tokens: list[list[str]]"""
        vocab = {}
        indptr, indices, data = [0], [], []
        counts = {}
        for toks in docs_tokens:
            counts.clear()
            for t in toks:
                j = vocab.get(t)
                if j is None:
                    j = vocab[t] = len(vocab)
                counts[j] = counts.get(j, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))
        n_docs, n_terms = len(docs_tokens), len(vocab)
        tf = sp.csr_matrix((np.asarray(data, dtype=np.float32),
                            np.asarray(indices, dtype=np.int32),
                            np.asarray(indptr, dtype=np.int64)),
                           shape=(n_docs, n_terms))
        self.vocab = vocab
        self._tf = tf
        self._dl = np.asarray([len(t) for t in docs_tokens], dtype=np.float32)
        self._avgdl = float(self._dl.mean()) if n_docs else 0.0
        df = np.diff(tf.tocsc().indptr).astype(np.float32)          # docs containing each term
        self._idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._rebuild()
        return self

    def _rebuild(self):
        """Recompute W for the current (k1, b). O(nnz)."""
        k1, b = self.k1, self.b
        tf = self._tf
        # denominator depends on the DOC (row), numerator on the cell
        norm = (1.0 - b + b * (self._dl / (self._avgdl + 1e-9))).astype(np.float32)   # per doc
        W = tf.copy()
        # expand row-wise denominator to nnz
        row_norm = np.repeat(norm, np.diff(W.indptr))
        W.data = (W.data * (k1 + 1.0)) / (W.data + k1 * row_norm)
        # multiply each column by its idf
        W = W.multiply(self._idf[np.newaxis, :]).tocsr().astype(np.float32)
        self.W = W

    def set_params(self, k1, b):
        self.k1, self.b = k1, b
        self._rebuild()
        return self

    def _qmat(self, queries_tokens):
        """Binary-presence query matrix (Lucene ignores query TF)."""
        n_q, n_terms = len(queries_tokens), len(self.vocab)
        rows, cols = [], []
        for i, toks in enumerate(queries_tokens):
            seen = set()
            for t in toks:
                j = self.vocab.get(t)
                if j is not None and j not in seen:
                    seen.add(j)
                    rows.append(i); cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(n_q, n_terms))

    def scores(self, queries_tokens):
        """-> dense (n_queries, n_docs) float32"""
        Q = self._qmat(queries_tokens)
        return np.asarray((Q @ self.W.T).todense(), dtype=np.float32)


def tune_k1_b(bm25, tune_q_tokens, tune_qrels, tune_qids, doc_ids, ndcg_fn,
              k1_grid=(0.6, 0.9, 1.2, 1.5, 1.8, 2.1), b_grid=(0.3, 0.5, 0.75, 0.9, 1.0)):
    """Grid-search k1/b on a query split. Returns (best_k1, best_b, best_ndcg, grid).

    IMPORTANT: the caller must pass a TUNE split disjoint from the evaluation split, otherwise the
    tuned baseline is selected on the reporting set -- the exact sin we are correcting.
    """
    best, grid = (None, None, -1.0), {}
    for k1 in k1_grid:
        for b in b_grid:
            bm25.set_params(k1, b)
            S = bm25.scores(tune_q_tokens)
            order = np.argsort(-S, axis=1)[:, :10]
            tot = 0.0
            for i, qid in enumerate(tune_qids):
                ranked = [doc_ids[j] for j in order[i]]
                tot += ndcg_fn(ranked, tune_qrels.get(qid, {}), 10)
            m = tot / max(1, len(tune_qids))
            grid[f"k1={k1},b={b}"] = m
            if m > best[2]:
                best = (k1, b, m)
    bm25.set_params(best[0], best[1])
    return best[0], best[1], best[2], grid

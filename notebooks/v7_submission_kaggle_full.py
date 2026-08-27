# %% [code]
# %% [code]
# ═══ CELL 0: Setup ═══
import os, sys, time, csv, json, gc, re
from pathlib import Path
from collections import defaultdict, Counter
START_TIME = time.time()

def elapsed():
    return f"{(time.time()-START_TIME)/60:.1f}min"

# Paths
MAMAYLM_DIR        = Path('/kaggle/input/datasets/taleeftamsal/mamaylm-gemma3-12b-gguf')
BGE_M3_DIR         = Path('/kaggle/input/datasets/taleeftamsal/bge-m3')
RERANKER_DIR       = Path('/kaggle/input/datasets/taleeftamsal/bge-reranker-v2-m3')
QWEN3_RERANKER_DIR = Path('/kaggle/input/datasets/taleeftamsal/qwen3-reranker-0-6b')
WHEELS_DIR         = '/kaggle/input/datasets/taleeftamsal/unlp2026-wheels'
PDF_DIR            = Path('/kaggle/input/competitions/unlp-2026-shared-task-on-multi-domain-document-understanding/test')
QUESTIONS          = str(Path('/kaggle/input/competitions/unlp-2026-shared-task-on-multi-domain-document-understanding/test.csv'))
OUTPUT_CSV         = '/kaggle/working/submission.csv'

gguf_files = list(MAMAYLM_DIR.glob('*.gguf')) if MAMAYLM_DIR.exists() else []
MAMAYLM_GGUF = str(gguf_files[0]) if gguf_files else ''

# Which reranker to use
USE_QWEN3 = QWEN3_RERANKER_DIR.exists() and (QWEN3_RERANKER_DIR / 'config.json').exists()
USE_BGE   = RERANKER_DIR.exists() and (RERANKER_DIR / 'config.json').exists()

print(f'GPU: {os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()}')
print(f'MamayLM GGUF:         {MAMAYLM_GGUF}')
print(f'BGE-M3 exists:        {BGE_M3_DIR.exists()}')
print(f'Qwen3 reranker exists:{USE_QWEN3}')
print(f'BGE reranker exists:  {USE_BGE}')
print(f'PDF dir exists:       {PDF_DIR.exists()}')
print(f'Reranker: {"Qwen3-0.6B" if USE_QWEN3 else "BGE-Reranker-v2-M3" if USE_BGE else "NONE (fallback to retrieval order)"}')

# ═══ CELL 1: Install Dependencies ═══
import subprocess, sys
from pathlib import Path

WHEELS_DIR = '/kaggle/input/datasets/taleeftamsal/unlp2026-wheels'

def run(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and r.stderr:
        print(f'  stderr: {r.stderr[:300]}')
    return r.returncode == 0

# PyMuPDF
try:
    import fitz; print('PyMuPDF: OK')
except ImportError:
    run(f'pip install --no-index --find-links={WHEELS_DIR} pymupdf -q')
    import fitz; print('PyMuPDF: OK')

# llama-cpp-python
try:
    from llama_cpp import Llama; print('llama-cpp-python: OK')
except ImportError:
    cp = f'cp{sys.version_info.major}{sys.version_info.minor}'
    for pkg in ['diskcache', 'jinja2', 'markupsafe', 'typing_extensions']:
        pkgs = list(Path(WHEELS_DIR).glob(f'{pkg}*'))
        if pkgs:
            run(f'pip install "{pkgs[0]}" -q --no-deps')
    wheels = list(Path(WHEELS_DIR).glob(f'llama_cpp_python*{cp}*.whl'))
    if not wheels:
        raise FileNotFoundError(f'No {cp} llama wheel in {WHEELS_DIR}')
    run(f'pip install "{wheels[0]}" -q --no-deps')
    from llama_cpp import Llama; print('llama-cpp-python: OK')

# FlagEmbedding (for potential hybrid retrieval — installed but not used by default)
try:
    import FlagEmbedding; print('FlagEmbedding: OK')
except ImportError:
    fe_wheel = list(Path(WHEELS_DIR).glob('FlagEmbedding*.tar.gz'))
    if fe_wheel:
        run(f'pip install "{fe_wheel[0]}" --no-deps -q')
        try:
            import FlagEmbedding; print('FlagEmbedding: OK (installed from wheel)')
        except Exception as e:
            print(f'FlagEmbedding install failed: {e} — proceeding without')
    else:
        print('FlagEmbedding wheel not found — skipping')

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
print(f'torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
print(f'All deps ready [{elapsed()}]')

# ═══ CELL 2: Extract Text (with page boundary overlap) ═══
import fitz

def extract_all_pages(pdf_root):
    """Extract text from all PDFs; stitch ±200 chars across page boundaries."""
    pages = []
    doc_meta = {}
    # Use rglob for robust discovery regardless of directory depth/naming
    all_pdfs = sorted(Path(pdf_root).rglob('*.pdf'))
    print(f'  PDF discovery: found {len(all_pdfs)} PDFs under {pdf_root}')
    if all_pdfs:
        print(f'  First 5 paths: {[str(p.relative_to(pdf_root)) for p in all_pdfs[:5]]}')
    for pdf_path in all_pdfs:
        doc_id = pdf_path.name
        domain = pdf_path.parent.name  # 'domain_1', 'domain_2', etc.
        doc = fitz.open(str(pdf_path))
        doc_meta[doc_id] = {'n_pages': len(doc), 'domain': domain}
        # Collect raw page texts first
        raw = [page.get_text() for page in doc]
        doc.close()
        # Augment each page with tail of previous and head of next (P8)
        for i, text in enumerate(raw):
            prefix = raw[i-1][-200:] if i > 0 else ''
            suffix = raw[i+1][:200] if i < len(raw)-1 else ''
            pages.append({
                'doc_id': doc_id,
                'page_num': i+1,
                'text': (prefix + text + suffix).strip()
            })
    return pages, doc_meta

print('Extracting text...')
t0 = time.time()
pages, doc_meta = extract_all_pages(PDF_DIR)
print(f'Extracted {len(pages)} pages from {len(doc_meta)} docs in {time.time()-t0:.1f}s')

text_lens = [len(p['text']) for p in pages]
print(f'Page text: min={min(text_lens)}, median={sorted(text_lens)[len(text_lens)//2]}, '
      f'max={max(text_lens)}, mean={sum(text_lens)/len(text_lens):.0f} chars')

with open(QUESTIONS, 'r', encoding='utf-8') as f:
    questions = list(csv.DictReader(f))
print(f'Loaded {len(questions)} questions')
print(f'Domains: {Counter(m["domain"] for m in doc_meta.values())}')
print(f'[{elapsed()}]')

# ═══ CELL 3: BGE-M3 Retrieval (dense, options-enriched queries, TOP_K=20) ═══
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BGE_PATH = str(BGE_M3_DIR)

# ── Device config ──
USE_GPU = False
if torch.cuda.is_available():
    try:
        test = torch.zeros(1, device='cuda')
        del test
        USE_GPU = True
    except Exception as e:
        print(f'CUDA test failed: {e}')

DEVICE = 'cuda' if USE_GPU else 'cpu'
DTYPE  = torch.float16 if USE_GPU else torch.float32
MAX_LENGTH = 1536 if USE_GPU else 1024
BATCH_SIZE = 8   if USE_GPU else 16

TOP_K_RETRIEVAL = 20  # P10: was 10

print(f'BGE-M3 config: device={DEVICE}, dtype={DTYPE}, max_length={MAX_LENGTH}, batch={BATCH_SIZE}')
print(f'Loading BGE-M3...')
t0 = time.time()
bge_tok   = AutoTokenizer.from_pretrained(BGE_PATH)
bge_model = AutoModel.from_pretrained(BGE_PATH, torch_dtype=DTYPE).to(DEVICE)
bge_model.eval()
print(f'BGE-M3 loaded in {time.time()-t0:.1f}s')
if USE_GPU:
    print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')

def encode_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, desc=""):
    all_embs = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = bge_tok(batch, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out  = bge_model(**encoded)
            embs = out.last_hidden_state[:, 0]
            embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
        done = min(i+batch_size, len(texts))
        if (i // batch_size) % 20 == 0 or done == len(texts):
            rate = done / (time.time() - t0) if time.time() > t0 else 0
            eta  = (len(texts) - done) / rate if rate > 0 else 0
            print(f'  {desc} {done}/{len(texts)} ({rate:.1f}/s, ETA {eta:.0f}s)')
    return np.vstack(all_embs).astype(np.float32)

# ── Embed pages ──
page_texts = [p['text'] for p in pages]
print(f'Embedding {len(page_texts)} pages...')
t0 = time.time()
page_embs = encode_texts(page_texts, desc="pages")
print(f'Page embedding: {time.time()-t0:.1f}s')

# ── Embed queries (question text only — tested: including options hurts Doc@1) ──
query_texts = [q['Question'] for q in questions]
print(f'Embedding {len(query_texts)} queries...')
t0 = time.time()
query_embs = encode_texts(query_texts, batch_size=32, max_length=256, desc="queries")
print(f'Query embedding: {time.time()-t0:.1f}s')

# ── Cosine similarity → top-20 ──
scores      = query_embs @ page_embs.T
top_indices = np.argsort(-scores, axis=1)[:, :TOP_K_RETRIEVAL]

bge_results = {}
for i, q in enumerate(questions):
    bge_results[q['Question_ID']] = [
        (pages[idx]['doc_id'], pages[idx]['page_num'], float(scores[i, idx]))
        for idx in top_indices[i]
    ]

# ── P6: Build per-page embedding dict for answer-aware page selection later ──
# Keep as float32 numpy vectors on CPU (only ~5MB for 1121 pages × 1024-dim)
page_embs_dict = {
    (pages[idx]['doc_id'], pages[idx]['page_num']): page_embs[idx].copy()
    for idx in range(len(pages))
}

# ── Free large matrices but KEEP bge_model + bge_tok for answer-aware reranking ──
# VRAM footprint: BGE-M3 stays at ~2.2GB during reranker + MCQ cells (9GB total safe on P100)
del page_embs, query_embs, scores, top_indices
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()
    print(f'VRAM after matrix free: {torch.cuda.memory_allocated()/1e9:.2f}GB')

print(f'Retrieval complete [{elapsed()}]')

# ═══ CELL 4: Cross-Encoder Reranking (Qwen3-0.6B preferred, BGE fallback) ═══
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

page_text_index = {(p['doc_id'], p['page_num']): p['text'] for p in pages}

RERANKER_MAX_LENGTH = 2048  # Qwen3 supports 32K; 2048 covers full pages
BGE_MAX_LENGTH      = 1024  # P1 BUG FIX: was 512, model trained at 1024

if USE_QWEN3:
    # ── Qwen3-Reranker-0.6B ──────────────────────────────────────────────────
    print('Loading Qwen3-Reranker-0.6B...')
    t0 = time.time()
    rerank_tok   = AutoTokenizer.from_pretrained(str(QWEN3_RERANKER_DIR), padding_side='left')
    rerank_model = AutoModelForCausalLM.from_pretrained(
        str(QWEN3_RERANKER_DIR), torch_dtype=torch.float16
    ).to(DEVICE).eval()
    print(f'Qwen3 reranker loaded in {time.time()-t0:.1f}s')
    if USE_GPU:
        print(f'  VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    token_true_id  = rerank_tok.convert_tokens_to_ids("yes")
    token_false_id = rerank_tok.convert_tokens_to_ids("no")
    assert token_true_id != rerank_tok.unk_token_id, "Could not find 'yes' token in vocabulary"

    TASK_INSTRUCTION = (
        "Given a Ukrainian question about a document, "
        "determine if the document passage contains the answer to the question."
    )
    SYSTEM_MSG = (
        "Judge whether the Document meets the requirements based on the Query and the "
        "Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )

    def format_qwen3_pair(query, doc_text):
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": (
                f"<Instruct>: {TASK_INSTRUCTION}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc_text[:4000]}"
            )}
        ]
        text = rerank_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Empty <think> tags disable chain-of-thought → faster, deterministic
        return text + "<think>\n\n</think>\n"

    RERANK_BATCH = 4  # smaller batches for CausalLM with 2K context

    @torch.no_grad()
    def rerank_pages(question_text, candidates):
        if not candidates: return candidates
        pairs = [
            format_qwen3_pair(question_text, page_text_index.get((d, p), ''))
            for d, p, _ in candidates
        ]
        all_scores = []
        for b in range(0, len(pairs), RERANK_BATCH):
            batch = pairs[b:b+RERANK_BATCH]
            enc = rerank_tok(
                batch, padding=True, truncation=True,
                max_length=RERANKER_MAX_LENGTH, return_tensors='pt'
            ).to(DEVICE)
            logits = rerank_model(**enc).logits[:, -1, :]  # last-token logits
            yes_l  = logits[:, token_true_id]
            no_l   = logits[:, token_false_id]
            probs  = torch.softmax(torch.stack([no_l, yes_l], dim=1), dim=1)[:, 1]
            all_scores.extend(probs.cpu().float().tolist())
        reranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
        return [(d, p, float(s)) for (d, p, _), s in reranked]

elif USE_BGE:
    # ── BGE-Reranker-v2-M3 (P1: max_length bug fix 512→1024) ────────────────
    print(f'Loading BGE-Reranker-v2-M3 (max_length={BGE_MAX_LENGTH})...')
    t0 = time.time()
    rerank_tok   = AutoTokenizer.from_pretrained(str(RERANKER_DIR))
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        str(RERANKER_DIR), torch_dtype=DTYPE
    ).to(DEVICE).eval()
    print(f'BGE reranker loaded in {time.time()-t0:.1f}s')

    RERANK_BATCH = 8

    @torch.no_grad()
    def rerank_pages(question_text, candidates):
        if not candidates: return candidates
        pairs = [
            [question_text, page_text_index.get((d, p), '')[:4000]]
            for d, p, _ in candidates
        ]
        all_scores = []
        for b in range(0, len(pairs), RERANK_BATCH):
            enc = rerank_tok(
                pairs[b:b+RERANK_BATCH], padding=True, truncation=True,
                return_tensors='pt', max_length=BGE_MAX_LENGTH  # fixed: was 512
            ).to(DEVICE)
            sc = rerank_model(**enc).logits.view(-1).float().cpu().tolist()
            all_scores.extend(sc)
        reranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
        return [(d, p, float(s)) for (d, p, _), s in reranked]

else:
    print('WARNING: No reranker found — using BGE-M3 retrieval order directly')
    def rerank_pages(question_text, candidates):
        return candidates

# ── Rerank all questions; preserve all TOP_K_RETRIEVAL candidates ──
HAS_RERANKER = USE_QWEN3 or USE_BGE
print(f'Reranking {len(questions)} questions...')
t0 = time.time()
reranked_results = {}
for i, q in enumerate(questions):
    qid = q['Question_ID']
    candidates = bge_results.get(qid, [])[:TOP_K_RETRIEVAL]
    reranked_results[qid] = rerank_pages(q['Question'], candidates)
    if (i+1) % 50 == 0 or (i+1) == len(questions):
        rate = (i+1) / (time.time()-t0)
        print(f'  {i+1}/{len(questions)} ({rate:.1f} q/s, ETA {(len(questions)-i-1)/rate:.0f}s)')

print(f'Reranking done in {(time.time()-t0)/60:.1f}min')

if HAS_RERANKER:
    changed = sum(
        1 for qid in bge_results
        if bge_results[qid][0][:2] != reranked_results[qid][0][:2]
    )
    print(f'Reranking changed top-1 for {changed}/{len(questions)} ({100*changed/len(questions):.1f}%)')

# ── Free reranker weights (keep bge_model + bge_tok alive for Cell 5) ──
if HAS_RERANKER:
    del rerank_model, rerank_tok
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print(f'VRAM after reranker free: {torch.cuda.memory_allocated()/1e9:.2f}GB')

print(f'[{elapsed()}]')

# ═══ CELL 5: MamayLM MCQ + Self-Consistency + Answer-Aware Page Selection ═══
from llama_cpp import Llama

ANSWER_CHOICES   = ['A', 'B', 'C', 'D', 'E', 'F']
TOP_K_CONTEXT    = 4     # P4: was 3
MAX_CHARS_PER_PAGE = 3500  # P4: was 2500
N_PASSES         = 3     # P7: self-consistency passes (pass 0 = greedy)
VOTE_TEMP        = 0.5   # temperature for stochastic passes
UA_TO_LATIN = {'А':'A','В':'B','С':'C','Д':'D','Е':'E','Ф':'F',
               'а':'A','в':'B','с':'C','д':'D','е':'E','ф':'F'}

assert MAMAYLM_GGUF, 'No GGUF file found!'

# Load MamayLM (bge_model already loaded, total VRAM ~9GB < 16GB P100)
print(f'Loading MamayLM...')
if USE_GPU:
    print(f'  VRAM before MamayLM: {torch.cuda.memory_allocated()/1e9:.2f}GB')
t0 = time.time()
try:
    llm = Llama(model_path=MAMAYLM_GGUF, n_gpu_layers=-1, n_ctx=8192, verbose=False)
    N_CTX = 8192
    print(f'MamayLM loaded with n_ctx=8192 in {time.time()-t0:.1f}s')
except Exception as e:
    print(f'n_ctx=8192 failed ({e}), falling back to 4096...')
    llm = Llama(model_path=MAMAYLM_GGUF, n_gpu_layers=-1, n_ctx=4096, verbose=False)
    N_CTX = 4096
    MAX_CHARS_PER_PAGE = 1500
    print(f'MamayLM loaded with n_ctx=4096 in {time.time()-t0:.1f}s')

if USE_GPU:
    print(f'  VRAM with both models: {torch.cuda.memory_allocated()/1e9:.2f}GB')

def get_context(ranked_pages, top_k=TOP_K_CONTEXT):
    parts, seen = [], set()
    for item in ranked_pages:
        doc_id, page_num = item[0], item[1]
        key = (doc_id, page_num)
        if key not in seen:
            text = page_text_index.get(key, '').strip()
            if text:
                parts.append(f'[Сторінка {page_num}]\n{text[:MAX_CHARS_PER_PAGE]}')
                seen.add(key)
        if len(parts) >= top_k: break
    return '\n\n'.join(parts) or '[Контекст недоступний]'

def extract_answer(text):
    text = text.strip()
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter): return letter
    for cyr, lat in UA_TO_LATIN.items():
        if text.startswith(cyr): return lat
    m = re.search(r'\b([A-F])\b', text.upper())
    if m: return m.group(1)
    for c in text.upper():
        if c in 'ABCDEF': return c
    return 'A'

# P2: v4_evidence prompt (86.8% oracle accuracy vs 84.6% for v1_direct)
def build_prompt(q, context):
    opts = '\n'.join(f'{l}. {q[l]}' for l in ANSWER_CHOICES)
    return (
        f"Документ:\n{context}\n\n"
        f"Питання: {q['Question']}\n\n"
        f"Варіанти:\n{opts}\n\n"
        f"Знайди відповідний фрагмент тексту, потім вибери правильний варіант.\n"
        f"Правильний варіант:"
    )

# P7: Self-consistency scoring with majority vote
def score_mcq_voting(q, context):
    prompt = build_prompt(q, context)
    votes = []
    # Pass 0: greedy (deterministic baseline)
    resp = llm(prompt, max_tokens=3, temperature=0.0, stop=['\n', '.'])
    votes.append(extract_answer(resp['choices'][0]['text']))
    # Passes 1+: stochastic for diversity
    for _ in range(N_PASSES - 1):
        resp = llm(prompt, max_tokens=3, temperature=VOTE_TEMP, stop=['\n', '.'])
        votes.append(extract_answer(resp['choices'][0]['text']))
    winner = Counter(votes).most_common(1)[0][0]
    return winner, votes

# P6: Answer-aware page selection (within-doc only)
# Cross-encoder top-1 is used for doc selection (preserves doc accuracy).
# Answer-aware cosine similarity only picks the page within that doc.
def answer_aware_page_select(q, pred_answer, candidates):
    """Doc = cross-encoder top-1; page = answer-aware cosine similarity within that doc."""
    if not candidates:
        return (list(doc_meta.keys())[0], 1, 0.0)
    # Lock doc to cross-encoder top-1 — prevents doc accuracy regression
    top_doc = candidates[0][0]
    same_doc = [c for c in candidates if c[0] == top_doc]
    if len(same_doc) <= 1:
        return same_doc[0] if same_doc else candidates[0]
    # Answer-aware page selection within the top doc
    answer_text = q.get(pred_answer, '').strip()
    if not answer_text:
        return same_doc[0]
    aug_query = f"{q['Question']} {answer_text}"
    enc = bge_tok([aug_query], padding=True, truncation=True,
                  max_length=256, return_tensors='pt').to(next(bge_model.parameters()).device)
    with torch.no_grad():
        out   = bge_model(**enc)
        q_emb = F.normalize(out.last_hidden_state[:, 0], dim=-1).cpu().float().numpy()[0]
    best_score, best_cand = -2.0, same_doc[0]
    for cand in same_doc:
        p_emb = page_embs_dict.get((cand[0], cand[1]))
        if p_emb is None: continue
        s = float(np.dot(q_emb, p_emb))
        if s > best_score:
            best_score, best_cand = s, cand
    return best_cand

# ── Main MCQ loop ──
print(f'Scoring {len(questions)} MCQs '
      f'(n_ctx={N_CTX}, {MAX_CHARS_PER_PAGE} chars/page, top_k={TOP_K_CONTEXT}, '
      f'{N_PASSES}-pass voting, answer-aware page selection)...')
predictions = []
t0 = time.time()
vote_diversity = []  # track how often votes disagree (for diagnostics)

for i, q in enumerate(questions):
    qid    = q['Question_ID']
    ranked = reranked_results.get(qid, bge_results.get(qid, []))

    # Build context from top-K reranked pages for the LLM
    context = get_context(ranked)

    # 1. Select answer (with self-consistency)
    pred_answer, votes = score_mcq_voting(q, context)
    vote_diversity.append(len(set(votes)) > 1)

    # 2. Answer-aware page selection from all TOP_K_RETRIEVAL candidates
    best_cand = answer_aware_page_select(q, pred_answer, ranked[:TOP_K_RETRIEVAL])
    pred_doc, pred_page = best_cand[0], best_cand[1]

    predictions.append({
        'Question_ID': qid,
        'Correct_Answer': pred_answer,
        'Doc_ID': pred_doc,
        'Page_Num': str(pred_page)
    })

    if (i+1) % 25 == 0 or (i+1) == len(questions):
        rate = (i+1) / (time.time()-t0)
        eta  = (len(questions)-i-1) / rate
        print(f'  Q{i+1}/{len(questions)}: ans={pred_answer} votes={votes} | '
              f'{rate:.2f} q/s | ETA {eta/60:.1f}min | total [{elapsed()}]')

print(f'MCQ done in {(time.time()-t0)/60:.1f}min [{elapsed()}]')
print(f'Vote diversity (votes differed): {sum(vote_diversity)}/{len(vote_diversity)} '
      f'({100*sum(vote_diversity)/max(len(vote_diversity),1):.1f}%)')

# Free both models
del llm, bge_model, bge_tok
gc.collect()
if USE_GPU:
    torch.cuda.empty_cache()

# ═══ CELL 6: Write & Validate ═══
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['Question_ID', 'Correct_Answer', 'Doc_ID', 'Page_Num'])
    w.writeheader()
    w.writerows(predictions)

print(f'=== FINAL TIMING: {elapsed()} (budget: 9h) ===')
print(f'Submission: {OUTPUT_CSV}')

with open(OUTPUT_CSV) as f:
    rows = list(csv.DictReader(f))

print(f'Rows: {len(rows)}')
print(f'Answers: {dict(Counter(r["Correct_Answer"] for r in rows))}')

assert all(r['Correct_Answer'] in 'ABCDEF' for r in rows), 'Bad answer!'
assert all(r['Doc_ID'].endswith('.pdf') for r in rows), 'Bad doc_id!'
assert all(r['Page_Num'].isdigit() for r in rows), 'Bad page_num!'
print(f'Validation PASSED — {len(rows)} rows ready for submission.')

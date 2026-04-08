#!/bin/bash
# Run all three GPU experiments for the paper.
# Execute this from an A30 interactive session:
#   sinteractive -A pfw-cs -p a30 --gres=gpu:1 --mem=32G --time=03:00:00 --qos=standby
# Then:
#   cd /scratch/gilbreth/tamst01/unlp2026
#   bash scripts/paper_analysis/run_gpu_experiments.sh

set -euo pipefail
REPO=/scratch/gilbreth/tamst01/unlp2026
cd "$REPO"

echo "============================================"
echo " PAPER GPU EXPERIMENTS"
echo "============================================"

# ── Experiment 1: Closed-book baseline ─────────────────────────────────────
echo ""
echo "[1/3] Closed-book baseline (top_k_context=0) ..."
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --override-json '{"llm": {"top_k_context": 0}}' \
  --questions data/dev_questions.csv \
  --output-dir outputs/benchmarks/paper_closed_book \
  --hardware-tag a30

echo "  → Closed-book done."
python3 -c "
import json
with open('outputs/benchmarks/paper_closed_book/summary.json') as f:
    s = json.load(f)
print(f'  answer_acc={s[\"answer_accuracy\"]:.4f}  doc_acc={s[\"doc_accuracy\"]:.4f}  composite={s[\"composite_score\"]:.4f}')
"

# ── Experiment 2: Context-size sensitivity (100-question sample) ────────────
echo ""
echo "[2/3] Context-size sensitivity sweep (k=1,2,3,5 on 100 questions) ..."

for k in 1 2 3 5; do
  echo "  k=$k ..."
  python3 scripts/benchmark_candidate.py \
    --preset v7_baseline \
    --override-json "{\"llm\": {\"top_k_context\": $k}}" \
    --n-questions 100 \
    --seed 42 \
    --output-dir "outputs/benchmarks/paper_ctx_k${k}_n100" \
    --hardware-tag a30
done

echo "  → Context sweep done."
python3 - << 'EOF'
import json
print("\n  k | answer | composite")
print("  --|--------|----------")
for k in [0, 1, 2, 3, 5]:
    path = f"outputs/benchmarks/paper_closed_book/summary.json" if k == 0 else \
           f"outputs/benchmarks/paper_ctx_k{k}_n100/summary.json"
    try:
        with open(path) as f:
            s = json.load(f)
        label = "0 (closed)" if k == 0 else str(k)
        print(f"  {label:2s} | {s['answer_accuracy']:.4f} | {s['composite_score']:.4f}")
    except FileNotFoundError:
        pass
EOF

# ── Experiment 3: Query formulation (question-only vs question+options) ─────
echo ""
echo "[3/3] Query formulation ablation (options_in_query, 100 questions) ..."

# question text + options in query (currently NOT in v7)
python3 scripts/benchmark_candidate.py \
  --preset v7_baseline \
  --override-json '{"retrieval": {"query_include_options": true}}' \
  --n-questions 100 \
  --seed 42 \
  --output-dir outputs/benchmarks/paper_query_with_options_n100 \
  --hardware-tag a30 2>/dev/null || echo "  (query_include_options override not supported — skipping)"

echo ""
echo "============================================"
echo " ALL GPU EXPERIMENTS COMPLETE"
echo "============================================"
echo "Run scripts/paper_analysis/compute_analyses.py to extract all numbers."

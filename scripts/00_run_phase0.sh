#!/bin/bash
# Run all Phase 0 steps in order
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026

echo "=== Phase 0: Setup & Data Audit ==="

echo "[0.3] Profiling data..."
python scripts/00_profile_data.py

echo "[0.4] Running eval harness (random baseline)..."
python scripts/00_eval_harness.py

echo "[0.5] Creating train/val splits..."
python scripts/00_create_splits.py

echo "=== Phase 0 complete. See outputs/ for results. ==="

#!/bin/bash
# Phase 0: Setup — clone repo, extract data, install deps
set -euo pipefail
cd /scratch/gilbreth/tamst01/unlp2026

echo "=== UNLP 2026 Setup ==="

# 1. Clone shared task repo (skip if already done)
if [ ! -d "shared_task_repo" ]; then
    git clone https://github.com/unlp-workshop/unlp-2026-shared-task.git shared_task_repo
else
    echo "Repo already cloned."
fi

# 2. Copy dev questions
cp shared_task_repo/data/dev_questions.csv data/dev_questions.csv

# 3. Extract PDFs
for domain in domain_1 domain_2; do
    target="data/raw_pdfs/$domain"
    if [ -z "$(ls -A $target 2>/dev/null)" ]; then
        unzip -o "shared_task_repo/data/$domain/dev.zip" -d "$target/"
        # Flatten if extracted into dev/ subdirectory
        if [ -d "$target/dev" ]; then
            mv "$target/dev/"*.pdf "$target/"
            rmdir "$target/dev"
        fi
    else
        echo "$domain PDFs already extracted."
    fi
done

echo "Domain 1 PDFs: $(ls data/raw_pdfs/domain_1/*.pdf | wc -l)"
echo "Domain 2 PDFs: $(ls data/raw_pdfs/domain_2/*.pdf | wc -l)"
echo "Dev questions: $(wc -l < data/dev_questions.csv) lines"

# 4. Install dependencies
pip install pymupdf 2>/dev/null || true

echo "=== Setup complete ==="

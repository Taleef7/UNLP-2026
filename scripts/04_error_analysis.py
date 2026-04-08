#!/usr/bin/env python3
"""Phase 4.2: Error analysis — categorize retrieval and answer errors.

Error categories:
  1. Wrong doc retrieved (doc_id mismatch)
  2. Right doc, wrong page (correct doc but off-page)
  3. Right page, wrong answer (retrieval perfect but LLM wrong)
  4. All correct
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUT = BASE / "outputs"


def categorize_error(pred, gt):
    """Categorize a single prediction error."""
    pred_answer = pred["Correct_Answer"]
    pred_doc = pred["Doc_ID"]
    pred_page = int(pred["Page_Num"])

    true_answer = gt["Correct_Answer"]
    true_doc = gt["Doc_ID"]
    true_page = int(gt["Page_Num"])
    n_pages = int(gt["N_Pages"])

    correct_answer = pred_answer == true_answer
    correct_doc = pred_doc == true_doc
    correct_page = correct_doc and pred_page == true_page

    if correct_answer and correct_doc and correct_page:
        return "all_correct"
    elif correct_answer and not correct_doc:
        return "correct_answer_wrong_doc"
    elif correct_answer and correct_doc and not correct_page:
        return "correct_answer_wrong_page"
    elif not correct_answer and correct_page:
        return "wrong_answer_correct_page"
    elif not correct_answer and correct_doc and not correct_page:
        return "wrong_answer_wrong_page_right_doc"
    elif not correct_answer and not correct_doc:
        return "wrong_answer_wrong_doc"
    else:
        return "other"


def analyze(pred_path, gt_path=None):
    """Analyze prediction errors."""
    if gt_path is None:
        gt_path = DATA / "dev_questions.csv"

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = list(csv.DictReader(f))
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = list(csv.DictReader(f))

    gt_by_id = {row["Question_ID"]: row for row in ground_truth}

    categories = []
    domain_errors = defaultdict(list)

    for pred in predictions:
        qid = pred["Question_ID"]
        if qid not in gt_by_id:
            continue
        gt = gt_by_id[qid]
        cat = categorize_error(pred, gt)
        domain = gt.get("Domain", "unknown")
        categories.append(cat)
        domain_errors[domain].append(cat)

    N = len(categories)
    print(f"\n=== Error Analysis ({pred_path.name}) ===")
    print(f"Total: {N} predictions\n")

    counts = Counter(categories)
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / N
        print(f"  {cat:<40s}: {cnt:4d} ({pct:5.1f}%)")

    print("\nPer-domain:")
    for domain, errs in sorted(domain_errors.items()):
        n = len(errs)
        dc = Counter(errs)
        correct = dc.get("all_correct", 0) + dc.get("correct_answer_wrong_doc", 0) + dc.get("correct_answer_wrong_page", 0)
        print(f"  {domain} (n={n}):")
        for cat, cnt in sorted(dc.items(), key=lambda x: -x[1]):
            print(f"    {cat:<40s}: {cnt}")

    return {"n": N, "categories": dict(counts), "per_domain": {
        d: dict(Counter(e)) for d, e in domain_errors.items()
    }}


def main():
    import sys
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "submission.csv"
    if not pred_path.exists():
        # Try to find any submission file
        subs = list((OUT / "submission").glob("*.csv")) if (OUT / "submission").is_dir() else []
        subs += list(OUT.glob("submission*.csv"))
        if subs:
            pred_path = max(subs, key=lambda p: p.stat().st_mtime)
            print(f"Using: {pred_path}")
        else:
            print("No submission file found. Run 03_pipeline.py first.")
            return

    results = analyze(pred_path)

    out_file = OUT / "error_analysis.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()

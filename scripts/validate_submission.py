#!/usr/bin/env python3
"""Validate submission.csv format before Kaggle upload."""
import csv
import sys
from pathlib import Path

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
ANSWER_CHOICES = set("ABCDEF")


def validate(submission_path, questions_path=None):
    """Validate a submission CSV."""
    if questions_path is None:
        questions_path = BASE / "data" / "dev_questions.csv"

    print(f"Validating: {submission_path}")

    # Load submission
    with open(submission_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Load ground truth (for doc/page validation)
    with open(questions_path, "r", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    gt_by_id = {r["Question_ID"]: r for r in gt_rows}

    errors = []
    warnings = []

    # Check required columns
    required_cols = {"Question_ID", "Correct_Answer", "Doc_ID", "Page_Num"}
    if rows:
        missing = required_cols - set(rows[0].keys())
        if missing:
            errors.append(f"Missing columns: {missing}")
            return errors, warnings

    # Check each row
    question_ids = set()
    for i, row in enumerate(rows):
        qid = row.get("Question_ID", "")
        answer = row.get("Correct_Answer", "")
        doc_id = row.get("Doc_ID", "")
        page_num = row.get("Page_Num", "")

        # Duplicate check
        if qid in question_ids:
            errors.append(f"Row {i}: Duplicate Question_ID: {qid}")
        question_ids.add(qid)

        # Answer validity
        if answer not in ANSWER_CHOICES:
            errors.append(f"Row {i} (Q{qid}): Invalid Correct_Answer: '{answer}'")

        # Doc_ID format
        if not doc_id.endswith(".pdf"):
            errors.append(f"Row {i} (Q{qid}): Doc_ID should end with .pdf: '{doc_id}'")

        # Page_Num
        if not page_num.isdigit():
            errors.append(f"Row {i} (Q{qid}): Page_Num must be integer: '{page_num}'")
        else:
            pn = int(page_num)
            if pn < 1:
                errors.append(f"Row {i} (Q{qid}): Page_Num must be >= 1: {pn}")
            # Check against N_Pages if in GT
            if qid in gt_by_id:
                n_pages = int(gt_by_id[qid]["N_Pages"])
                if pn > n_pages:
                    warnings.append(f"Row {i} (Q{qid}): Page_Num {pn} > N_Pages {n_pages}")

        # NaN check
        for col, val in row.items():
            if val in ("nan", "NaN", "", None):
                errors.append(f"Row {i} (Q{qid}): Empty/NaN in column '{col}'")

    # Check coverage
    if questions_path.exists():
        gt_ids = set(gt_by_id.keys())
        sub_ids = question_ids
        missing_from_sub = gt_ids - sub_ids
        extra_in_sub = sub_ids - gt_ids
        if missing_from_sub:
            warnings.append(f"Missing {len(missing_from_sub)} question IDs from submission "
                            f"(first 5: {list(missing_from_sub)[:5]})")
        if extra_in_sub:
            warnings.append(f"Extra {len(extra_in_sub)} question IDs in submission not in GT")

    # Summary
    print(f"\nRows: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\n=== ERRORS ===")
        for e in errors[:20]:
            print(f"  ✗ {e}")
    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings[:10]:
            print(f"  ⚠ {w}")

    if not errors:
        print("\n✓ Submission is VALID")
    else:
        print(f"\n✗ Submission has {len(errors)} ERRORS — fix before submission")

    return errors, warnings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py outputs/submission.csv")
        sys.exit(1)
    errors, _ = validate(sys.argv[1])
    sys.exit(0 if not errors else 1)

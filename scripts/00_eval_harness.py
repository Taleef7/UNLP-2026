#!/usr/bin/env python3
"""Phase 0.4: Evaluation harness using the shared competition metric implementation."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.pipeline_shared import evaluate_predictions, load_csv

BASE = REPO_ROOT


def format_results(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("UNLP 2026 Evaluation Results")
    lines.append("=" * 60)
    lines.append(f"N questions:      {results['n_questions']}")
    lines.append(f"Answer accuracy:  {results['answer_accuracy']:.4f} (weight=0.50)")
    lines.append(f"Doc accuracy:     {results['doc_accuracy']:.4f} (weight=0.25)")
    lines.append(f"Page proximity:   {results['page_proximity']:.4f} (weight=0.25)")
    lines.append(f"COMPOSITE SCORE:  {results['composite_score']:.4f}")
    lines.append("")
    lines.append("Per-domain breakdown:")
    for domain, domain_results in results["per_domain"].items():
        lines.append(f"  {domain} (n={domain_results['n']}):")
        lines.append(
            f"    Answer={domain_results['answer_acc']:.4f}  "
            f"Doc={domain_results['doc_acc']:.4f}  "
            f"Page={domain_results['page_prox']:.4f}  "
            f"Composite={domain_results['composite']:.4f}"
        )
    return "\n".join(lines)


def generate_random_baseline(ground_truth: list) -> list:
    import random
    from collections import defaultdict

    random.seed(42)
    predictions = []
    all_docs = {}
    for row in ground_truth:
        all_docs[row["Doc_ID"]] = {"n_pages": int(row["N_Pages"]), "domain": row["Domain"]}

    docs_by_domain = defaultdict(list)
    for doc_id, info in all_docs.items():
        docs_by_domain[info["domain"]].append(doc_id)

    for row in ground_truth:
        doc_id = random.choice(docs_by_domain[row["Domain"]])
        predictions.append(
            {
                "Question_ID": row["Question_ID"],
                "Correct_Answer": random.choice(["A", "B", "C", "D", "E", "F"]),
                "Doc_ID": doc_id,
                "Page_Num": str(random.randint(1, all_docs[doc_id]["n_pages"])),
            }
        )
    return predictions


def main() -> None:
    ground_truth = load_csv(BASE / "data" / "dev_questions.csv")

    if len(sys.argv) > 1:
        predictions = load_csv(sys.argv[1])
    else:
        print("No prediction file provided. Running random baseline.\n")
        predictions = generate_random_baseline(ground_truth)

    results = evaluate_predictions(predictions, ground_truth)
    print(format_results(results))

    output_path = BASE / "outputs" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

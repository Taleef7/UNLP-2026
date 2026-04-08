#!/usr/bin/env python3
"""Phase 0.3: Data profiling — count PDFs, pages, questions, text extractability."""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
OUTPUT = BASE / "outputs"


def profile_pdfs():
    """Profile all PDFs: page counts, text extractability."""
    pdf_stats = {}
    domain_stats = defaultdict(lambda: {"n_pdfs": 0, "total_pages": 0, "pages_list": []})

    for domain_dir in sorted((DATA / "raw_pdfs").iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        pdfs = sorted(domain_dir.glob("*.pdf"))
        for pdf_path in pdfs:
            doc_id = pdf_path.name
            try:
                doc = fitz.open(str(pdf_path))
                n_pages = len(doc)
                # Check text extractability per page
                page_chars = []
                scanned_pages = 0
                for page in doc:
                    text = page.get_text()
                    char_count = len(text.strip())
                    page_chars.append(char_count)
                    if char_count < 50:
                        scanned_pages += 1
                doc.close()

                pdf_stats[doc_id] = {
                    "domain": domain,
                    "n_pages": n_pages,
                    "total_chars": sum(page_chars),
                    "avg_chars_per_page": sum(page_chars) / max(n_pages, 1),
                    "min_chars_page": min(page_chars) if page_chars else 0,
                    "max_chars_page": max(page_chars) if page_chars else 0,
                    "scanned_pages": scanned_pages,
                    "pct_scanned": scanned_pages / max(n_pages, 1),
                }
                domain_stats[domain]["n_pdfs"] += 1
                domain_stats[domain]["total_pages"] += n_pages
                domain_stats[domain]["pages_list"].append(n_pages)
            except Exception as e:
                print(f"  ERROR reading {pdf_path}: {e}")
                pdf_stats[doc_id] = {"domain": domain, "error": str(e)}

    return pdf_stats, dict(domain_stats)


def profile_questions():
    """Profile dev_questions.csv."""
    questions = []
    with open(DATA / "dev_questions.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    domain_counts = Counter(q["Domain"] for q in questions)
    answer_dist = Counter(q["Correct_Answer"] for q in questions)
    doc_ids = set(q["Doc_ID"] for q in questions)

    # Page number stats per domain
    domain_page_stats = defaultdict(list)
    for q in questions:
        domain_page_stats[q["Domain"]].append(int(q["Page_Num"]))

    # N_Pages distribution
    n_pages_values = [int(q["N_Pages"]) for q in questions]

    return {
        "total_questions": len(questions),
        "domain_counts": dict(domain_counts),
        "answer_distribution": dict(sorted(answer_dist.items())),
        "unique_doc_ids": len(doc_ids),
        "doc_ids_per_domain": dict(Counter(q["Domain"] for q in questions)),
        "n_pages_range": [min(n_pages_values), max(n_pages_values)],
        "n_pages_mean": sum(n_pages_values) / len(n_pages_values),
        "page_num_stats": {
            domain: {
                "min": min(pages),
                "max": max(pages),
                "mean": sum(pages) / len(pages),
            }
            for domain, pages in domain_page_stats.items()
        },
    }


def main():
    print("=" * 60)
    print("UNLP 2026 Data Profile")
    print("=" * 60)

    # Profile PDFs
    print("\n--- PDF Profiling ---")
    pdf_stats, domain_stats = profile_pdfs()

    for domain, stats in sorted(domain_stats.items()):
        pages = stats["pages_list"]
        print(f"\n{domain}:")
        print(f"  PDFs: {stats['n_pdfs']}")
        print(f"  Total pages: {stats['total_pages']}")
        if pages:
            print(f"  Pages/PDF: min={min(pages)}, max={max(pages)}, "
                  f"mean={sum(pages)/len(pages):.1f}")

    # Check scanned pages
    total_scanned = sum(s.get("scanned_pages", 0) for s in pdf_stats.values())
    total_pages = sum(s.get("n_pages", 0) for s in pdf_stats.values())
    print(f"\nTotal pages: {total_pages}")
    print(f"Scanned/low-text pages (<50 chars): {total_scanned} "
          f"({100*total_scanned/max(total_pages,1):.1f}%)")

    # Profile questions
    print("\n--- Question Profiling ---")
    q_stats = profile_questions()
    print(f"Total questions: {q_stats['total_questions']}")
    print(f"Unique documents: {q_stats['unique_doc_ids']}")
    print(f"Questions per domain: {q_stats['domain_counts']}")
    print(f"Answer distribution: {q_stats['answer_distribution']}")
    print(f"N_Pages range: {q_stats['n_pages_range']}")
    print(f"N_Pages mean: {q_stats['n_pages_mean']:.1f}")

    for domain, ps in q_stats["page_num_stats"].items():
        print(f"  {domain} page_num: min={ps['min']}, max={ps['max']}, mean={ps['mean']:.1f}")

    # Random baseline
    random_answer_acc = 1.0 / 6.0  # 6 options
    print(f"\nRandom baseline answer accuracy: {random_answer_acc:.4f} ({100*random_answer_acc:.2f}%)")

    # Save results
    profile = {
        "pdf_stats": pdf_stats,
        "domain_stats": {k: {kk: vv for kk, vv in v.items() if kk != "pages_list"}
                         for k, v in domain_stats.items()},
        "question_stats": q_stats,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT / 'data_profile.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 1.1: Extract text from PDFs page-by-page using PyMuPDF."""

import json
import time
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
PDF_DIR = BASE / "data" / "raw_pdfs"
OUT_DIR = BASE / "data" / "extracted_text"


def extract_pdf(pdf_path):
    """Extract text from each page of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_num": i + 1,  # 1-indexed
            "text": text,
            "char_count": len(text.strip()),
        })
    doc.close()
    return pages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    total_pdfs = 0
    total_pages = 0
    low_text_pages = 0
    manifest = {}

    for domain_dir in sorted(PDF_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        pdfs = sorted(domain_dir.glob("*.pdf"))
        print(f"\n{domain}: {len(pdfs)} PDFs")

        for pdf_path in pdfs:
            doc_id = pdf_path.name
            pages = extract_pdf(pdf_path)

            # Save per-document JSON
            out_path = OUT_DIR / f"{doc_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"doc_id": doc_id, "domain": domain, "pages": pages},
                          f, ensure_ascii=False, indent=1)

            n_low = sum(1 for p in pages if p["char_count"] < 50)
            total_pdfs += 1
            total_pages += len(pages)
            low_text_pages += n_low

            manifest[doc_id] = {
                "domain": domain,
                "n_pages": len(pages),
                "total_chars": sum(p["char_count"] for p in pages),
                "low_text_pages": n_low,
            }
            print(f"  {doc_id}: {len(pages)} pages, "
                  f"{sum(p['char_count'] for p in pages)} chars, "
                  f"{n_low} low-text pages")

    # Save manifest
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\nDone: {total_pdfs} PDFs, {total_pages} pages in {elapsed:.1f}s")
    print(f"Low-text pages (<50 chars): {low_text_pages}/{total_pages}")


if __name__ == "__main__":
    main()

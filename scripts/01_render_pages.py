#!/usr/bin/env python3
"""Phase 1.2: Render PDF pages to PNG images using PyMuPDF."""

import json
import time
from pathlib import Path

import fitz  # PyMuPDF

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
PDF_DIR = BASE / "data" / "raw_pdfs"
IMG_DIR = BASE / "data" / "page_images"
DPI = 144


def render_pdf(pdf_path, out_dir):
    """Render each page as PNG."""
    doc = fitz.open(str(pdf_path))
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []

    for i, page in enumerate(doc):
        page_num = i + 1
        pix = page.get_pixmap(matrix=mat)
        img_path = out_dir / f"page_{page_num:04d}.png"
        pix.save(str(img_path))
        pages.append({
            "page_num": page_num,
            "width": pix.width,
            "height": pix.height,
            "path": str(img_path),
        })

    doc.close()
    return pages


def main():
    start = time.time()
    total_pages = 0
    manifest = {}

    for domain_dir in sorted(PDF_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        pdfs = sorted(domain_dir.glob("*.pdf"))
        print(f"\n{domain}: {len(pdfs)} PDFs")

        for pdf_path in pdfs:
            doc_id = pdf_path.stem  # without .pdf
            out_dir = IMG_DIR / doc_id
            out_dir.mkdir(parents=True, exist_ok=True)

            pages = render_pdf(pdf_path, out_dir)
            total_pages += len(pages)

            manifest[pdf_path.name] = {
                "domain": domain,
                "n_pages": len(pages),
                "image_dir": str(out_dir),
                "dpi": DPI,
            }
            print(f"  {pdf_path.name}: {len(pages)} pages rendered")

    # Save manifest
    with open(IMG_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\nDone: {total_pages} pages rendered at {DPI} DPI in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Build the paper. Requires a TeX Live installation with pdflatex and bibtex on PATH.
set -e
cd "$(dirname "$0")"
J=harness_answers_a
pdflatex -interaction=nonstopmode -halt-on-error "$J.tex"
bibtex "$J"
pdflatex -interaction=nonstopmode -halt-on-error "$J.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$J.tex"

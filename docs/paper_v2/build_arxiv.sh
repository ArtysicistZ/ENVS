#!/bin/bash
# Build and package ENVS paper for ArXiv submission.
# Usage: cd docs/paper && bash build_arxiv.sh
set -e

SOURCE_FILES=(
    ENVS.tex
    references.bib
    teaser.png
    pipeline.png
    data-efficiency.png
)

SUBMISSION_FILES=(
    ENVS.tex
    ENVS.bbl
    references.bib
    teaser.png
    pipeline.png
    data-efficiency.png
)

echo "=== Checking source files ==="
missing=0
for f in "${SOURCE_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        missing=1
    else
        echo "  OK: $f ($(du -h "$f" | cut -f1))"
    fi
done
if [ "$missing" -eq 1 ]; then
    echo "ERROR: Missing source files. Aborting."
    exit 1
fi

echo ""
echo "=== Cleaning previous build artifacts ==="
rm -f ENVS.aux ENVS.bbl ENVS.blg ENVS.log ENVS.out ENVS.pdf ENVS.synctex.gz ENVS-arxiv.tar.gz 2>/dev/null || true

echo ""
echo "=== Compiling paper (pdflatex + bibtex + pdflatex + pdflatex) ==="
pdflatex -interaction=nonstopmode ENVS.tex > /dev/null
bibtex ENVS
pdflatex -interaction=nonstopmode ENVS.tex > /dev/null
pdflatex -interaction=nonstopmode ENVS.tex > /dev/null
echo "Compilation successful."

echo ""
echo "=== Verifying bibliography resolved ==="
undef=$(grep -c "Citation.*undefined" ENVS.log || true)
if [ "$undef" -gt 0 ]; then
    echo "ERROR: $undef undefined citations remain in ENVS.log"
    grep "Citation.*undefined" ENVS.log | head -5
    exit 1
fi
ref_undef=$(grep -c "Reference.*undefined" ENVS.log || true)
if [ "$ref_undef" -gt 0 ]; then
    echo "WARNING: $ref_undef undefined references (cross-refs):"
    grep "Reference.*undefined" ENVS.log | head -5
fi
echo "All citations resolved."

echo ""
echo "=== Verifying submission files ==="
for f in "${SUBMISSION_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f missing after build."
        exit 1
    fi
    echo "  OK: $f ($(du -h "$f" | cut -f1))"
done

echo ""
echo "=== Creating ArXiv archive ==="
tar czf ENVS-arxiv.tar.gz "${SUBMISSION_FILES[@]}"
echo "Archive: ENVS-arxiv.tar.gz"
echo "Size:    $(du -h ENVS-arxiv.tar.gz | cut -f1)"
echo "Files:   ${#SUBMISSION_FILES[@]}"
echo ""
echo "Contents:"
tar tzf ENVS-arxiv.tar.gz

echo ""
echo "=== PDF info ==="
if command -v pdfinfo > /dev/null 2>&1; then
    pdfinfo ENVS.pdf | grep -E "Pages|File size"
else
    echo "ENVS.pdf ($(du -h ENVS.pdf | cut -f1))"
fi

echo ""
echo "=== Done ==="
echo "Upload ENVS-arxiv.tar.gz to https://arxiv.org/submit"

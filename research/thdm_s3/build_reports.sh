#!/usr/bin/env bash
# Rebuild the report PDFs (report_*.tex -> report_*.pdf) for reading on GitHub.
#
# The .tex files are the source; the PDFs are committed (see .gitignore) so the
# reports render in the browser.  Figures are re-extracted from the notebooks'
# stored outputs first, so a report never quotes a stale plot.
#
# Run from anywhere:  bash research/thdm_s3/build_reports.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 extract_figures.py

status=0
for tex in report_*.tex; do
    name="${tex%.tex}"
    # twice, so cross-references and citations resolve
    for _ in 1 2; do
        pdflatex -interaction=nonstopmode -halt-on-error "$tex" > /dev/null || true
    done
    if grep -qE '^!' "$name.log"; then
        echo "FAIL $tex: LaTeX error"; grep -E '^!' "$name.log" | head -3; status=1
    elif grep -qE 'undefined (references|citations)|Reference .* undefined|Citation .* undefined' "$name.log"; then
        echo "FAIL $tex: undefined reference or citation"; status=1
    else
        echo "ok   $name.pdf"
    fi
    rm -f "$name.aux" "$name.log" "$name.out"
done
exit $status

#!/usr/bin/env bash
# Rebuild the report PDFs (reports/report_*.tex -> reports/report_*.pdf) and the derivation-ledger
# PDFs (derivations_0N.tex, emitted by the notebooks) for reading on GitHub.
#
# The .tex files are the source; the PDFs are committed (see research/.gitignore) so they
# render in the browser.  Figures are re-extracted from the notebooks'
# stored outputs first, so a report never quotes a stale plot.
#
# Run from anywhere:  bash research/thdm_s3/build_reports.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 extract_figures.py
# the headline numbers the reports quote, read from results/*.json (reports/numbers.tex)
python3 report_numbers.py

status=0
# each file compiles in its own folder, where its \input and figure paths resolve
for path in reports/report_*.tex derivations_*.tex; do
    dir="$(dirname "$path")"; tex="$(basename "$path")"; name="${tex%.tex}"
    pushd "$dir" > /dev/null
    # twice, so cross-references and citations resolve
    for _ in 1 2; do
        pdflatex -interaction=nonstopmode -halt-on-error "$tex" > /dev/null || true
    done
    if grep -qE '^!' "$name.log"; then
        echo "FAIL $tex: LaTeX error"; grep -E '^!' "$name.log" | head -3; status=1
    elif grep -qE 'undefined (references|citations)|Reference .* undefined|Citation .* undefined' "$name.log"; then
        echo "FAIL $tex: undefined reference or citation"; status=1
    else
        echo "ok   $dir/$name.pdf"
    fi
    rm -f "$name.aux" "$name.log" "$name.out"
    popd > /dev/null
done
exit $status

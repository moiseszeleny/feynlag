"""Write ``numbers.tex``: the headline numbers the reports quote, read from the results.

A report that types "610 of 2021" by hand goes stale the moment a notebook is
re-run, and three commits on this branch have fixed exactly that. Instead, the
reports ``\\input{numbers}`` and write ``\\nMEGallowed{}`` of ``\\nSoftPoints{}``.
`build_reports.sh` runs this script first, so a rebuilt PDF always quotes the
JSON.

``check`` also lists every place where a report still types one of these
values literally, so new hand-typed copies are caught (citation years, such as
"(2021)", are skipped).

Usage (from anywhere)::

    python research/thdm_s3/report_numbers.py           # write numbers.tex
    python research/thdm_s3/report_numbers.py check     # also list literal copies

To add a number: add a `NUMBERS` entry, rebuild, and use the macro in the report.

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _get(doc, path):
    for key in path.split("/"):
        doc = doc[int(key)] if isinstance(doc, list) else doc[key]
    return doc


def count(n):
    """An integer; a LaTeX thin space separates thousands from five digits up."""
    n = int(n)
    return f"{n:,}".replace(",", r"\,") if n >= 10_000 else str(n)


def sci(x):
    """A sample size as m×10^e, e.g. 110000000 → 1.1\\times10^8."""
    e = int(math.floor(math.log10(x)))
    m = f"{x / 10**e:.2g}"
    return (rf"{m}\times10^{{{e}}}" if m != "1" else rf"10^{{{e}}}")


def pct(x):
    return rf"{100 * x:.2f}\%"


def pct1(x):
    return rf"{100 * x:.1f}\%"


def pct2(x):
    """A percentage to two significant figures: 0.000201 → 0.020\\%."""
    p = 100 * x
    digits = max(0, 1 - int(math.floor(math.log10(abs(p))))) if p else 0
    return rf"{p:.{digits}f}\%"


#: macro -> (results file, key path, formatter, what it is)
NUMBERS = {
    "nExactPoints": ("viable_points.json", "n_points", count,
                     "points surviving the exact-S3 scan"),
    "nExactSample": ("viable_points.json", "sample_size", sci,
                     "points sampled in the exact-S3 scan (use in maths mode)"),
    "nSoftPoints": ("viable_points_soft.json", "n_points", count,
                    "points surviving the soft-broken scan"),
    "nSoftSample": ("viable_points_soft.json", "sample_size", sci,
                    "points sampled in the soft-broken scan (use in maths mode)"),
    "nLFVexcluded": ("soft_decay_benchmarks.json",
                     "n_points_lfv_excluded_for_every_lepton_config", count,
                     "soft points with no lepton configuration under the h->ll' limits"),
    "nMuEConfigs": ("mu_e_gamma.json", "summary/n_configurations", count,
                    "(point, mu3, branch) configurations in the soft scan"),
    "fMEGconfigs": ("mu_e_gamma.json", "summary/frac_configurations_under_meg", pct,
                    "fraction of configurations under MEG II"),
    "nHLFVallowed": ("mu_e_gamma.json", "summary/n_points_allowed_h", count,
                     "soft points allowed by the h->ll' limits"),
    "nMEGallowed": ("mu_e_gamma.json", "summary/n_points_allowed_meg", count,
                    "soft points allowed by MEG II"),
    "nBothAllowed": ("mu_e_gamma.json", "summary/n_points_allowed_both", count,
                     "soft points allowed by MEG II and the h->ll' limits"),
    "nMEGnoQuark": ("mu_e_gamma.json", "summary/n_points_allowed_meg_variants/no_quark",
                    count, "MEG II survivors without the quark Barr-Zee term"),
    "nMEGtopFlip": ("mu_e_gamma.json", "summary/n_points_allowed_meg_variants/top_flip",
                    count, "MEG II survivors with the quark couplings' sign flipped"),
    "nMEGwHKZ": ("mu_e_gamma.json", "summary/n_points_allowed_meg_variants/w_hkz",
                 count, "MEG II survivors with the W loop at the HKZ13 value"),
    "xMEGmedianBR": ("mu_e_gamma.json", "summary/median_br", sci,
                     "median B(mu -> e gamma) over the configurations (use in maths mode)"),
    "xMEGoverHemu": ("mu_e_gamma.json", "summary/median_meg_over_hemu_ratio", sci,
                     "median (B/MEG)/(B(h->e mu)/CMS) per configuration (use in maths mode)"),
    "nTauConfigs": ("tau_l_gamma.json", "summary/n_configurations", count,
                    "(point, mu3, branch) configurations in the tau scan"),
    "nExactEvals": ("tau_l_gamma.json", "summary/exact/n_evaluations", count,
                    "(exact point, mu3) evaluations at the exact-S3 vacuum"),
    "fExactOverBelle": ("tau_l_gamma.json", "summary/exact/frac_over_belle", pct2,
                        "exact-vacuum evaluations over the Belle tau->mu gamma limit"),
    "fExactOverCMS": ("tau_l_gamma.json", "summary/exact/frac_over_cms_taumu", pct1,
                      "exact-vacuum evaluations over the CMS h->tau mu limit"),
    "nExactBelleNotCMS": ("tau_l_gamma.json", "summary/exact/n_over_belle_under_cms", count,
                          "exact-vacuum evaluations over Belle but under CMS h->tau mu"),
    "fSoftOverBelle": ("tau_l_gamma.json", "summary/frac_configurations_over_limit/tau_mu",
                       pct2, "soft configurations over the Belle tau->mu gamma limit"),
    "fSoftOverBaBar": ("tau_l_gamma.json", "summary/frac_configurations_over_limit/tau_e",
                       pct2, "soft configurations over the BaBar tau->e gamma limit"),
    "nTauBothAllowed": ("tau_l_gamma.json", "summary/n_points_allowed/tau_both", count,
                        "soft points with a configuration under both tau limits"),
    "nAllRadAllowed": ("tau_l_gamma.json", "summary/n_points_allowed/all_rad", count,
                       "soft points under all three radiative limits"),
    "nAllRadHAllowed": ("tau_l_gamma.json", "summary/n_points_allowed/all_rad_h", count,
                        "soft points under all three radiative and the h->ll' limits"),
    "nRadNoQuark": ("tau_l_gamma.json",
                    "summary/n_points_allowed_all_radiative_variants/no_quark", count,
                    "all-radiative survivors without the quark Barr-Zee term"),
    "nRadTopFlip": ("tau_l_gamma.json",
                    "summary/n_points_allowed_all_radiative_variants/top_flip", count,
                    "all-radiative survivors with the quark couplings' sign flipped"),
    "nRadwHKZ": ("tau_l_gamma.json",
                 "summary/n_points_allowed_all_radiative_variants/w_hkz", count,
                 "all-radiative survivors with the W loop at the HKZ13 value"),
}


def values():
    cache, out = {}, {}
    for macro, (fname, path, fmt, _) in NUMBERS.items():
        if fname not in cache:
            cache[fname] = json.loads((HERE / "results" / fname).read_text())
        out[macro] = fmt(_get(cache[fname], path))
    return out


def write():
    vals = values()
    lines = ["% Generated by report_numbers.py from results/*.json -- do not edit by hand.",
             "% Use as \\macro{} in text; the sample sizes belong in maths mode."]
    for macro, (fname, path, _, what) in NUMBERS.items():
        lines.append(f"\\newcommand{{\\{macro}}}{{{vals[macro]}}}  % {what} ({fname}: {path})")
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote numbers.tex ({len(vals)} numbers)")
    return vals


_CITATION_YEAR = re.compile(r"\(\d{4}\)|\\textbf\{\d+\},\s*\d+")


def literal_copies(vals):
    """Report lines that still type a macro's value by hand."""
    hits = []
    for tex in sorted(HERE.glob("report_*.tex")):
        for n, line in enumerate(tex.read_text().splitlines(), 1):
            text = _CITATION_YEAR.sub("", re.split(r"(?<!\\)%", line)[0])
            for macro, v in vals.items():
                if not re.fullmatch(r"[\d\\,]{3,}", v):
                    continue      # plain counts of 3+ digits; "5" would match anything
                if re.search(rf"(?<![\d.]){re.escape(v)}(?!\d|\.\d|\\,\d)", text):
                    hits.append(f"{tex.name}:{n}: {v} (\\{macro}{{}})")
    return hits


if __name__ == "__main__":
    vals = write()
    if sys.argv[1:] == ["check"]:
        hits = literal_copies(vals)
        print("\n".join(hits) if hits else "no hand-typed copies of a tracked number")

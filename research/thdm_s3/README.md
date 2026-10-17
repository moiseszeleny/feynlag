# 3HDM-S₃: scalar sector, fermion sector, decays and LFV

Research project on the S₃-symmetric three-Higgs-doublet model, built end to end with
`feynlag`: the potential, vacuum, mass matrices, parameter scans, Yukawa sectors, decays and
lepton-flavour-violating Higgs rates. These are findings and may be revised. The running log
is [`NOTES.md`](NOTES.md).

## How to review, by time available

**30 minutes: what is claimed, and on what assumptions.**
1. The summary column of the report table below.
2. [`CLAIMS.md`](CLAIMS.md) §1, the load-bearing assumptions. For each one, ask whether the
   conclusions it "affects" would survive if it changed.
3. `python research/thdm_s3/provenance.py check`: are the results you are reading current?

**2 hours: the reasoning.**
1. The [benchmark cards](BENCHMARKS.md): three points followed through notebooks 05-08.
   Check one number per card against its notebook.
2. The reports, in the order of the table. Each figure in a report comes from a notebook's
   stored output.
3. [`CLAIMS.md`](CLAIMS.md) §2, the imported results. Open each cited source and check the
   equation number, the sign conventions and the factors of 2.

**1 day: the algebra.**
1. Print the `derivations_0N.tex` ledgers and work through each step on paper.
2. For each derived claim in [`CLAIMS.md`](CLAIMS.md) §3, ask whether its falsifier could
   really have failed.
3. Re-run one notebook end to end, then `provenance.py check` and `stamp` it.

Record every verdict in [`review_status.csv`](review_status.csv) (`ok` / `doubt` / `wrong`,
plus a note), then run `python research/thdm_s3/claims.py`.

## Tools

| tool | what it answers | output |
|---|---|---|
| `claims.py` | what is claimed, and what is derived, imported or assumed | [`CLAIMS.md`](CLAIMS.md), [`review_status.csv`](review_status.csv) |
| `provenance.py` | which results, notebooks and reports are out of date | `check`/`stamp`/`audit` on [`results/MANIFEST.json`](results/MANIFEST.json) |
| `report_numbers.py` | the headline numbers the reports quote, read from `results/` | `numbers.tex` (run by `build_reports.sh`) |
| `benchmarks.py` | what one point looks like from end to end | [`BENCHMARKS.md`](BENCHMARKS.md) |
| `run_pipeline.py` | regenerates `cache/` and whatever is out of date, in order | the per-point data in `cache/` |

The data flow they share (which notebook reads and writes which file) is
[`pipeline.py`](pipeline.py). Update it when you add a notebook.

**Where the data lives.** `results/` holds small committed summaries: counts, cut flows,
conventions. The per-point data (the scan points, one row per lepton configuration) lives in
the gitignored `cache/`, and each summary records the SHA-256 of its cache file
(`points_sha256`). [`store.py`](store.py) is the only code that knows the split. The heavy
per-point loops of notebooks 06-08 run in `compute_soft_decays.py`, `compute_mu_e_gamma.py`
and `compute_tau_l_gamma.py`; the notebooks load their output and keep the narrative, the
validations and the figures.

## Reports

| report | source notebook | main result |
|---|---|---|
| [Scalar sector](report_scalar_sector.pdf) | [01](01_scalar_parameter_space.ipynb) | The Das–Dey boundedness conditions are necessary but not sufficient; the corrected condition, and a scan with a coupling-based cut centred on m_H = 125.20 GeV (1973 points from 6×10⁷). |
| [Fermion sector](report_fermion_sector.pdf) | [02](02_s3_fermion_sector.ipynb) | Three defects in the LFVHD draft, corrected; exact S₃ forces V_us = 0, and soft breaking turns the Cabibbo angle on. |
| [Scalar decays and LFV](report_scalar_decays.pdf) | [03](03_scalar_decays.ipynb) | h₀ is gauge-phobic for every δ; LFV data bound the lepton parameter μ₃, not the scalar sector; at the exact vacuum the SM-like state gives τμ only and h₀ only eτ and eμ. |
| [Soft-broken vacuum](report_soft_vacuum.pdf) | [05](05_soft_scalar_scan.ipynb) | The soft-broken spectrum is decoupling-like; the exact scan's light gauge-phobic scalars do not survive generic soft breaking. |
| [Soft-vacuum decays and LFV](report_soft_decays.pdf) | [06](06_soft_decays.ipynb) | The lepton fit is an exact arrowhead cubic and soft breaking pins μ₃ near ±m_τ; SM-like LFV is pure non-vacuum admixture; h→eμ binds hardest; only 5 of 2028 scalar points are LFV-excluded. |
| [μ→eγ](report_mu_e_gamma.pdf) | [07](07_mu_to_e_gamma.ipynb) | Forbidden to all orders at the exact vacuum (residual Z₂); in the soft vacuum the W-loop Barr–Zee diagram dominates and MEG II keeps 618 of 2028 scalar points, against 2023 for the Higgs LFV limits. |
| τ→μγ, τ→eγ (no report yet) | [08](08_tau_to_l_gamma.ipynb) | At the exact vacuum τ→eγ vanishes but τ→μγ is allowed, and Belle excludes a light-scalar corner that h→τμ misses; in the soft vacuum both τ modes are ~10⁻¹² and remove no point beyond MEG II. |

Notebook [04](04_bfb_conditions.ipynb) (the Boto–Romão–Silva boundedness method) has no
report yet. Each notebook also emits a printable derivation ledger, `derivations_0N.tex`.

## Rebuilding

**On a fresh clone `cache/` is empty**, so the notebooks that read per-point data cannot
run yet. Regenerate it first:

```bash
python research/thdm_s3/run_pipeline.py --dry-run   # what would run, and why
python research/thdm_s3/run_pipeline.py             # WALLTIME
```

It runs the scans, the compute scripts and the notebooks in `pipeline.py` order, re-running
only what is stale or missing, stamps each node in `results/MANIFEST.json`, and ends by
comparing each regenerated cache file with the hash its committed summary records
("reproduced" means byte-identical per-point data).

The `.tex` files are the source; the PDFs are committed so they read on GitHub. After
editing a report or re-executing a notebook:

```bash
bash research/thdm_s3/build_reports.sh
```

This re-extracts the figures from the notebooks' stored outputs (`extract_figures.py`),
regenerates `numbers.tex` from `results/*.json` (`report_numbers.py`), and compiles each
report twice. It fails on any LaTeX error or undefined reference. Quote headline numbers
through the `numbers.tex` macros (`\nMEGallowed{}`), not by typing them;
`python research/thdm_s3/report_numbers.py check` lists any number still typed by hand.

After re-executing a notebook, also run `claims.py` and `benchmarks.py`, and record the
run with `provenance.py stamp <node>`.

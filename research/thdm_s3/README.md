# 3HDM-S₃: scalar sector, fermion sector, decays and LFV

Research project on the S₃-symmetric three-Higgs-doublet model, built end to end with
`feynlag`: the potential, vacuum, mass matrices, parameter scans, Yukawa sectors, decays and
lepton-flavour-violating Higgs rates. These are findings and may be revised. The running log
is [`NOTES.md`](NOTES.md).

**Reviewing?** Start with [`CLAIMS.md`](CLAIMS.md). It lists every claim across the notebooks, load-bearing assumptions first. Record review marks in [`review_status.csv`](review_status.csv).

## Reports

| report | source notebook | main result |
|---|---|---|
| [Scalar sector](report_scalar_sector.pdf) | [01](01_scalar_parameter_space.ipynb) | The Das–Dey boundedness conditions are necessary but not sufficient; the corrected condition, and a scan with a coupling-based 125 GeV cut (1964 points from 6×10⁷). |
| [Fermion sector](report_fermion_sector.pdf) | [02](02_s3_fermion_sector.ipynb) | Three defects in the LFVHD draft, corrected; exact S₃ forces V_us = 0, and soft breaking turns the Cabibbo angle on. |
| [Scalar decays and LFV](report_scalar_decays.pdf) | [03](03_scalar_decays.ipynb) | h₀ is gauge-phobic for every δ; LFV data bound the lepton parameter μ₃, not the scalar sector; at the exact vacuum the SM-like state gives τμ only and h₀ only eτ and eμ. |
| [Soft-broken vacuum](report_soft_vacuum.pdf) | [05](05_soft_scalar_scan.ipynb) | The soft-broken spectrum is decoupling-like; the exact scan's light gauge-phobic scalars do not survive generic soft breaking. |
| [Soft-vacuum decays and LFV](report_soft_decays.pdf) | [06](06_soft_decays.ipynb) | The lepton fit is an exact arrowhead cubic and soft breaking pins μ₃ near ±m_τ; SM-like LFV is pure non-vacuum admixture; h→eμ binds hardest; only 5 of 2021 scalar points are LFV-excluded. |
| [μ→eγ](report_mu_e_gamma.pdf) | [07](07_mu_to_e_gamma.ipynb) | Forbidden to all orders at the exact vacuum (residual Z₂); in the soft vacuum the W-loop Barr–Zee diagram dominates and MEG II keeps 610 of 2021 scalar points, against 2016 for the Higgs LFV limits. |
| τ→μγ, τ→eγ (no report yet) | [08](08_tau_to_l_gamma.ipynb) | At the exact vacuum τ→eγ vanishes but τ→μγ is allowed, and Belle excludes a light-scalar corner that h→τμ misses; in the soft vacuum both τ modes are ~10⁻¹² and remove no point beyond MEG II. |

Notebook [04](04_bfb_conditions.ipynb) (the Boto–Romão–Silva boundedness method) has no
report yet. Each notebook also emits a printable derivation ledger, `derivations_0N.tex`.

## Rebuilding

The `.tex` files are the source; the PDFs are committed so they read on GitHub. After
editing a report or re-executing a notebook:

```bash
bash research/thdm_s3/build_reports.sh
```

This re-extracts the figures from the notebooks' stored outputs (`extract_figures.py`) and
compiles each report twice. It fails on any LaTeX error or undefined reference.

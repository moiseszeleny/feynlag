# 3HDM-S₃: scalar sector, fermion sector, decays and LFV

Research project on the S₃-symmetric three-Higgs-doublet model, built end to end with
`feynlag`: the potential, vacuum, mass matrices, parameter scans, Yukawa sectors, decays and
lepton-flavour-violating Higgs rates. These are findings and may be revised. The running log
is [`NOTES.md`](NOTES.md).

## Reports

| report | source notebook | main result |
|---|---|---|
| [Scalar sector](report_scalar_sector.pdf) | [01](01_scalar_parameter_space.ipynb) | The Das–Dey boundedness conditions are necessary but not sufficient; the corrected condition, and a scan with a coupling-based 125 GeV cut (1964 points from 6×10⁷). |
| [Fermion sector](report_fermion_sector.pdf) | [02](02_s3_fermion_sector.ipynb) | Three defects in the LFVHD draft, corrected; exact S₃ forces V_us = 0, and soft breaking turns the Cabibbo angle on. |
| [Scalar decays and LFV](report_scalar_decays.pdf) | [03](03_scalar_decays.ipynb) | h₀ is gauge-phobic for every δ; LFV data bound the lepton parameter μ₃, not the scalar sector; at the exact vacuum the SM-like state gives τμ only and h₀ only eτ and eμ. |
| [Soft-broken vacuum](report_soft_vacuum.pdf) | [05](05_soft_scalar_scan.ipynb) | The soft-broken spectrum is decoupling-like; the exact scan's light gauge-phobic scalars do not survive generic soft breaking. |

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

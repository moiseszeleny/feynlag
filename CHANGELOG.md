# Changelog

All notable changes to feynlag are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — unreleased

First public release (beta).

### Model building and checks
- Parameters split into external and internal, with dependencies resolved in
  order; scalar, Weyl-fermion, Majorana-fermion and gauge-boson fields.
- Gauge groups `U1` and `SUN` (any SU(N) irrep, generators built in the
  Gelfand–Tsetlin basis); discrete groups `ZN` and `S3` with Clebsch–Gordan
  products.
- Lagrangian building blocks: `Dmu`, `dag`, `Bilinear`, `MajoranaBilinear`,
  `fermion_gauge_current`.
- Invariance checks (gauge, discrete, hermiticity, mass dimension with an
  EFT `max_dim` flag) and a `Model.validate()` umbrella adding anomaly
  cancellation, electric-charge conservation and a UFO round-trip.
- Operator suggestion (`suggest_potential`, `suggest_yukawa`,
  `suggest_kinetic`, `build_lagrangian`), including the dim-5 Weinberg
  operator.
- Reusable Standard Model builders (`feynlag.models`) and CKM mixing
  (`standard_ckm`).

### Symmetry breaking, masses and vertices
- VEV expansion, tadpole extraction and solving, and scalar, charged,
  gauge-boson, Dirac and Majorana mass matrices, including the type-I seesaw.
- Diagonalization: orthogonal and analytic 2×2, SVD (biunitary), Takagi,
  and `MajoranaRotation` for Majorana mass eigenstates.
- Momentum-space Feynman rules for SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS,
  FFV, four-fermion (FFFF) and Majorana vertices.

### Export
- LaTeX vertex tables.
- UFO model directories for MadGraph, with SU(3) colour tensors and
  four-fermion Lorentz structures; the exported SM reproduces MadGraph's
  stock `sm` cross sections (`docs/benchmark.md`).

### Phenomenology (`feynlag.pheno`)
- Tree-level 1→2 partial widths, total widths and branching ratios, with a
  `DiracParticle` abstraction for fermions.
- Off-shell `h→WW*/ZZ*` widths (1→3 with a Breit–Wigner propagator).
- Loop-induced `h→gg`, `h→γγ` and `h→Zγ` from standard effective vertices.
- Tree-level single-diagram 2→2 cross sections, including the γ₅ (ε·ε) term
  and the forward–backward asymmetry.

### Known limitations
- No R_ξ gauge fixing or ghosts.
- Majorana vertices are not exported to UFO.
- 2→2 scattering does not yet sum interfering diagrams; `VVV` decays are not
  implemented.

# Changelog

All notable changes to feynlag are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `diagonalize_svd(method="auto"|"symbolic"|"numeric", dps=50)`: a numeric
  matrix that is complex, floating-point or larger than 2×2 is decomposed with
  mpmath's complex SVD, returning unitary rotations with
  `R_L M R_R† = diag(m ≥ 0)` (rows by increasing mass, a round-off singular
  value as an exact zero). Generic complex 3×3 Yukawas now give their masses
  and `V_CKM = R_uL·R_dL†` inside feynlag (FG-6). Exact 2×2 calls are unchanged.
- `Rotation.bar(old_bar, new_bar)`: the bar-leg partner of a fermion rotation,
  with the conjugate matrix `R*` (identical to `R` for a real rotation).

### Fixed
- `diagonalize_svd` no longer returns a silently wrong answer on a complex
  matrix: the symbolic route raises on an explicit `I`. It also no longer
  crashes with `TypeError` on a real floating-point input.

## [0.2.0] — 2026-09-26

### Added
- `diagonalize_takagi(method="auto"|"symbolic"|"numeric", dps=50)`: for a
  numeric matrix larger than 2×2, `auto` factorises with mpmath's `eigsy` at
  high precision (columns by increasing mass, round-off eigenvalues as exact
  zero modes), so a generic 3+2 seesaw 5×5 finishes instead of stalling in the
  exact route. Exact 2×2 and symbolic calls are unchanged.
- Per-symbol LaTeX: `component_tex=[...]` on `Scalar`, `WeylFermion`/
  `MajoranaFermion`, `GaugeBoson` and `GaugeGroup.bosons()`, `tex=` on
  `conjugate_pair`, and a parameter's existing `tex=` now reach plain
  `sympy.latex` through the symbol itself (`TexSymbol`), so downstream code
  needs no name→LaTeX map. A one-component field's `tex=` also names its
  component, and fermion bar legs print as `\overline{…}`.
- Opt-in LaTeX names in the SM builders and `Scalar.expand_vev`: each takes
  `tex=`, a map from the *name* of a symbol it creates to a LaTeX name
  (`electroweak_gauge`, `higgs_doublet`, `electroweak_scaffold`,
  `weinberg_rotation`, `charged_current_rotation`, `to_physical_basis`,
  `standard_ckm`), so one global table covers the couplings, the Higgs
  parameters, the doublet and its fluctuations (`H0_r`→`h`), and the physical
  `Z`, `A`, `W±`, `G⁻`. Names missing from the map stay plain. An explicit
  `symbol_names` printer setting still overrides a symbol's own tex.
  `GaugeGroup.bosons()` raises if a later call asks for a different tex than
  the cached bosons carry.
- A `TexSymbol` sorts exactly like a plain `Symbol` of the same name, so
  adding a tex never reorders vertex legs or printed terms.

### Fixed
- The UFO writer now escapes `texname`/`antitexname`. It wrote them as bare
  `'...'` literals, so a tex containing a quote (`g'`) made `parameters.py`
  a `SyntaxError`, and a backslash was read as a Python escape (`\tau` →
  TAB + `au`).

### Changed
- A field component or parameter declared with a tex is a `TexSymbol`, which
  is not equal to a plain `Symbol` of the same name. Code that rebuilds such a
  symbol by name (`sp.Symbol("lam")`) must use the object instead. Without a
  tex nothing changes.

## [0.1.0] — 2026-09-16

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

# Changelog

All notable changes to feynlag are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- 2→2 scattering, Tier 5 stage 5c: **`gg→gg`**, the first amplitude with no
  fermion line and the first four-point contact vertex in a diagram. New
  `feynlag.pheno.boson` (`BosonTerm`, `BosonAmplitude`): terms are a
  coefficient times loose tensors, squared through the new trace-free
  `contraction.contract_tensors` with the same polarization sums and Ward
  check as the line engine. `topology.enumerate_boson_diagrams` builds three
  colour structures `f^{ije}f^{ekl}`, each an exchange diagram through two
  three-gluon vertices plus its share of the contact vertex
  (`boson_vertices.vvvv_tensor`, coupling `adjoint_vvvv = i g²`);
  `colour.four_boson_colour_tensor` gives the colour matrix. All four gluons
  are summed over physical polarizations. Reproduces the
  Ellis–Stirling–Webber entry `9/2 (3 − tu/s² − su/t² − st/u²)` exactly, and
  SU(3) and SU(2) match an explicit polarization-vector evaluator. Any other
  four-boson process still raises. This completes Tier 5.
- `line.check_ward_identities`: the Ward probe of `LineAmplitude.squared`,
  now shared with `BosonAmplitude.squared`.
- 2→2 scattering, Tier 5 stage 5b: **external gluons** on one quark line —
  `q q̄→gg`, `gg→q q̄`, `qg→qg`, and the mixed `q q̄→gγ`. The external
  state is `calc.gluon`. `ScatteringCalculator` adds the three-gluon vertex
  (`adjoint_vvv`, colour `f^{abc}`), builds the colour matrix with
  `colour.line_colour_tensor`, and sums two external gluons over their
  physical polarizations: `VectorLeg(reference=n)` gives
  `−g_{μν} + (k_μn_ν + n_μk_ν)/(k·n)`. `LineAmplitude.squared(colour=)`.
  Reproduces the three gluon entries of the Ellis–Stirling–Webber parton
  table exactly; massive quarks (`gg→Q Q̄`) match an explicit evaluator.
- `ExternalBoson.color`, `LineDiagramInfo.order`,
  `line_amplitude(..., references=)`.
- 2→2 scattering, Tier 5 stage 5a: **coloured four-fermion processes**
  (`qq′→qq′`, `q q̄→q′q̄′`, `qq→qq`, `q q̄→q q̄`, `e⁺e⁻→q q̄`, `q q̄→ℓ⁺ℓ⁻`).
  `ScatteringCalculator(colour_group=SU3c, gluon=G)` turns the
  colour-component gluon vertices into one `q̄ q g` vertex with a
  colour-stripped coupling (`calc.gluon` is the mediator symbol); a quark is
  a `DiracParticle` on one colour component with `color=3`.
  `feynlag.pheno.colour` (`four_fermion_colour_tensor`, `colour_matrix`)
  builds each diagram's colour tensor from the group generators, and
  `Amplitude.squared(kin, colour=)` weights `M_d M̄_{d′}` by `C_{dd′}`.
  Reproduces the four-quark entries of the Ellis–Stirling–Webber parton
  table exactly. `examples/qcd_scattering.py`.
- `DiagramInfo.lines`: the external positions each fermion line joins.
- 2→2 scattering, Tier 4 stage 4c: **the `VSS` derivative coupling**. A
  vector meeting two scalars is now a diagram vertex: `e⁺e⁻→φ⁺φ⁻` through a
  photon or Z, and a scalar mediator with one external vector and one
  external scalar (`e⁻γ→χφ⁻`). `boson_vertices.vss_tensor` resolves each
  `p(φ)` momentum tag of the extracted coupling to `VSS_KAPPA = −1` times
  the momentum flowing into the vertex on field leg `φ`. Scalar QED
  reproduces `σ(e⁺e⁻→φ⁺φ⁻) = πα²β³/(3s)`; the sign is pinned by the Ward
  identity of `e⁻γ→χφ⁻`. `examples/scalar_qed.py`. This completes Tier 4.
- 2→2 scattering, Tier 4 stage 4b: **boson exchange through a bosonic
  three-point vertex** — `e⁺e⁻→W⁺W⁻` and `e⁺e⁻→Zh`.
  `ScatteringCalculator(gauge_groups=[SU2L])` collects the triple-gauge
  vertices from `Model.gauge_vertices`; `feynlag.pheno.boson_vertices`
  (`vvv_tensor`, `vvs_tensor`, `sss_tensor`, `VVV_KAPPA`) holds the Lorentz
  structures and `line.boson_exchange_terms` assembles the diagrams.
  `σ(e⁺e⁻→W⁺W⁻) = 19.447 pb` at √s = 200 GeV against MadGraph's
  19.498 ± 0.058 pb; the high-energy cancellation between the ν exchange and
  the γ/Z s channel is pinned exactly at a rational parameter point;
  `σ(e⁺e⁻→Zh)` matches the Higgsstrahlung closed form.
  `examples/ee_to_ww.py`.
- 2→2 scattering, Tier 4 stage 4a: **external bosons** on one open fermion
  line, by fermion exchange. `ScatteringCalculator.process` now takes two
  fermions and two bosons (`e⁺e⁻→γγ`, `e⁺e⁻→ZZ`, `e⁺e⁻→Zγ`, Compton);
  `feynlag.pheno.line` (`LineAmplitude`, `LineTerm`, `VectorLeg`,
  `fermion_exchange_term`) holds the amplitude and
  `topology.enumerate_line_diagrams` builds it from the extracted vertices.
  Massive vectors use the full polarization sum; a massless one uses `−g`
  only after its Ward identity has been checked on the amplitude. A process
  that also has a boson-exchange diagram (`e⁺e⁻→W⁺W⁻`, `e⁺e⁻→Zh`) raises
  until stage 4b. Checked against the Klein–Nishina form with a massive
  electron and an explicit spinor + polarization-vector evaluator.
- `feynlag.pheno.contraction.contract_trace`: a fully contracted Dirac trace
  with loose metrics and momentum vectors, reduced to dot products without
  SymPy tensors.
- `DiracParticle(..., right=None)`: a left-handed-only species (the SM
  neutrino).
- `ExternalBoson`, `fermion_propagator_phase`.
- 2→2 scattering, Tier 3: multi-diagram **interference**.
  `Amplitude.squared` sums `Σ_{d,d'} M_d M̄_{d'}`. A pair of diagrams that
  pair the external fermions the same way (γ/Z s channel) gives a product of
  two cross traces (`SpinorChain.cross_trace`). An exchanged pair (Bhabha
  s × t, Møller t × u) gives one trace through all four legs, whose γ₅ part
  is proven zero by the Gram-determinant guard.
- `feynlag.pheno.topology.enumerate_diagrams`: s/t/u enumeration of the
  two-line tree diagrams of a 2→2 four-fermion process, from the extracted
  vertices. Fermion signs come from permutation parity. Charged mediators
  join through an explicit `conjugates` map.
- `ScatteringCalculator` / `ScatteringProcess`: diagrams, `Σ|M|²`, `dσ/dcosθ`,
  `σ`, `A_FB` and a quadrature `numeric_cross_section` straight from a
  model and `DiracParticle`s. It reproduces MadGraph's `e+e-→μ+μ-`
  (2.7876 pb inside `|η|<2.5` vs 2.7878 ± 0.0027 pb). Bhabha/Møller match
  their closed forms, and `ν_e e` matches the Fierz "g_L+1" shift.
  `examples/ee_scattering.py`.
- `cross_section(..., cos_range=)` angular acceptance;
  `BosonPropagator.phase()`/`.denominator()`; `integrate.quad_1d`.

### Changed
- `ScatteringCalculator` no longer refuses an external gluon or coloured
  fermions with external bosons. A gluon *component* symbol (`G_1`) as an
  external state raises `KeyError`; four external bosons (`gg→gg`) raise
  until stage 5c.
- `ScatteringCalculator` no longer refuses a coloured `DiracParticle` in a
  four-fermion process when `colour_group` and `gluon` are given; without
  them it still raises. External gluons raise until stage 5b.
- `ScatteringProcess.initial_states` carries each fermion's colour
  multiplicity, so the initial-state average divides by `N_c` per coloured
  leg.
- `Model.physical_lagrangian` no longer raises for a model with no VEV'd
  scalar (QED, scalar QED): there is nothing to shift, so it returns the
  Lagrangian as written (with any rotations applied).
- `collect_decay_vertices` applies `conjugate_map` on the fermion track too.
  Before, the `ψ̄ψφ†` half of a charged-scalar Yukawa had no recognised
  boson leg and was silently missing from the vertex list (so from
  `DecayCalculator` and `ScatteringCalculator`). A model whose charged
  scalars are all Goldstones left out of `boson_fields` is unaffected.
- `ScatteringCalculator` evaluates bosonic vertex couplings at zero for
  fields left out of `boson_fields`, dropping the three-leg "vertices" the
  extractor reports for four-leg terms with a Goldstone (`A h W⁺ ∝ G⁻`).
  Four-fermion processes never used a bosonic vertex, so no earlier result
  changes.
- A two-fermion + two-boson process with a boson-exchange diagram no longer
  raises (stage 4a refused it). A charged external vector still raises when
  `gauge_groups` is not given.

### Fixed
- The Ward-identity guard of `LineAmplitude.squared` did not catch two
  non-abelian massless vectors both summed with `−g`: there
  `k_μM^{μν} ∝ k′^ν`, and `−g` squares that to `k′² = 0`. The check now also
  contracts the other `−g` leg with the probed leg's momentum, so that sum
  is refused. No shipped process was affected (none had two external
  non-abelian massless vectors).
- The γ₅ guard (`assert_epsilon_single_vanishes`) refused valid processes
  whose masses were floats: the Gram determinant of four dependent momenta
  then survives as a round-off residue (`3e-17·s³t²`) instead of `0`. A
  float-valued table is now tested for singularity numerically; exact input
  is treated as before.
- `ScatteringCalculator`'s stray-field filter read the scalar symbols inside
  a `VSS` coupling's `p(φ)` tags as field dependence and would have turned
  them into `p(0)`.
- `epsilon.gram_determinant` returned `nan` when a massless leg's mass was
  given as the float `0.0` (Bareiss elimination pivoting on `k·k = 0.0`), so
  the γ₅ guard refused a valid process. A float zero entry is now an exact
  zero.
- `ExternalState.spin_states()` returned 1 for a massless vector; it is 2, as
  its docstring said. No shipped process had a vector in the initial state,
  so no published number changes.
- The scattering docs called Tier 1's QED-only 2.322 pb "the photon-only
  fraction" of MadGraph's 2.7878 pb. That number includes MadGraph's default
  `|η_ℓ|<2.5` cut; the full-angle γ+Z value is 2.8443 pb.
- `Amplitude.squared` no longer crashes on a line with zero couplings (its
  trace is a plain `0`).

## [0.4.0] — 2026-11-15

Model building for particle physicists: terms written at field level, the
physical basis found by the model, and a front door that starts from every
allowed operator (#46–#50). All additive; 0.3.0 code runs unchanged.

### Added
- Field-level term constructors (`feynlag.terms`): `yukawa(y, FL, S, FR)`,
  `dirac_mass(M, FL, FR)`, `majorana_mass(M, F)` and `weinberg(c, F, S)` build
  the gauge contraction, the flavour indices and the `+ h.c.` in one call;
  `tilde(H)` selects the conjugate doublet, and `inner(A, B)` / `eps(A, B)`
  give the scalar invariants `A†B` and `Aᵀ(iσ₂)B`. A coupling may be a
  parameter (flavour-universal) or an `IndexedBase` matrix (`Y[i, j]`). Each
  returns exactly the expression the component-level `Bilinear` code produces.
  A term whose U(1) charges do not balance raises a `ValueError` giving the
  charge sum and, where it applies, the `S` ↔ `tilde(S)` fix; an unsupported
  representation raises `NotImplementedError`. `feynlag.suggest` now uses the
  same contraction functions.
- `kinetic(*fields)` and `Lagrangian.add_kinetic(*fields)`: the scalar kinetic
  terms and fermion gauge currents of a list of fields.
- `FL_I`, `FL_J`: canonical flavour-index symbols, the defaults of the
  constructors above.
- `ParameterSet.external(...)` / `.internal(...)`: create, register and return
  a parameter, so it is declared once (`Model(parameters=P)`).
- Named component access: `H.H0`, `PsiL.EL`, `PsiL.ELbar`, and after
  `expand_vev` the fluctuations `H.H0_r` / `H.H0_i`; `Field.named()` lists
  them. The stored object is returned, so a tex-carrying symbol is never
  rebuilt by name.
- `Model.check_parameters()` / `Model.undeclared_symbols()`: Lagrangian or VEV
  symbols that are neither a declared parameter nor a field. Opt-in through
  `Model.validate(parameters=True)`; the default `validate()` is unchanged.
- `Model.summary()`: the field-content table (type, gauge representations,
  discrete and global assignments, flavours, VEVs) and the parameter table, as
  text or Markdown.
- `Parameter.__rpow__` (`2 ** p`).

- Guided physical basis. `Model.mass_blocks()` partitions the scalar
  fluctuations and the Dirac fermion legs into the blocks that actually mix,
  each tagged with its electric charge (derived from the vacuum), its CP
  content and its Goldstone directions (`i T^a⟨φ⟩`, checked to be zero modes;
  unsolved tadpoles are reported). Nothing is assumed about CP: a
  CP-violating potential gives one merged neutral block.
  `Model.diagonalize(block, names=[...])` picks the route (Goldstone-fixed
  rotation, `tan 2θ`, analytic 2×2 biunitary with bar-side rotations, or a
  numeric rotation with `values=`), verifies it, registers it, and creates the
  new symbols and conjugate pairs. With `values=` a symbolic 2×2 keeps its
  symbolic angle and the numbers choose the `tan 2θ` branch so the first name
  is the heavier state; `branch_valid_for` records where that holds.
  `Model.physical` accumulates `bosons`, `cmap`, `conjugates`, `charges`,
  `goldstones` and `masses`; `weinberg_rotation`, `charged_current_rotation`
  and `to_physical_basis` record into it too. A symbolic block with no closed
  form, a field mixing with a conjugate field, and flavoured or Majorana
  fermion blocks are refused or reported with the route to use instead.
- `Model.most_general(name, fields, gauge_groups, discrete_groups)`
  (`feynlag.general.most_general_lagrangian`): the most general Lagrangian
  the declared symmetries allow, built from the `suggest` operator basis with
  a named coupling of the right mass dimension per operator (`m2_k`, `mu_k`,
  `lam_k`, `M_k`, `y_k`, `c_k`; `names=` overrides), flavour-matrix Yukawas
  for multi-flavour fermions, and the kinetic terms. Gauge bosons missing from
  `fields` are added. A potential operator needing `+ h.c.` gets one real
  coupling (the CP-odd combination is not enumerated).
- `Model.term_coverage()` / `Model.missing_terms()`: decompose a hand-written
  potential and Yukawa sector on that basis and report the coefficient of each
  invariant, the allowed invariants that are absent, and the remainder (a
  non-invariant term, a complex coupling's CP-odd part, a soft breaking of a
  declared symmetry).
- `InvarianceReport.explain()` and `report.hints`: one plain sentence per
  failed check (which monomial carries a net U(1) charge, gauge bosons missing
  from `Model(fields=...)`, a sector missing its `+ h.c.`);
  `raise_on_failure()` includes it. `invariance.u1_charge_breakdown` is the
  underlying diagnostic. `report.failures` is unchanged.
- Gauge bosons in the guided physical basis. `Model.mass_blocks(kind="gauge")`
  returns the charged and neutral gauge blocks of every group that acts on a
  VEV, and `Model.diagonalize` rotates them: the charge eigenstates
  `(A_a − iA_b)/√2` are read off the unbroken charge, and the photon is fixed
  exactly as `A ∝ Σ (c_k/g_k) A_k`, with no diagonalization. One massive
  neutral state is then fixed too (`angle=` gives `rotation_2x2(−θ_W)`); two
  that mix (`Z`–`Z′`) take the `tan 2θ` route in the photon's complement and
  are registered as one composed rotation; two charged states that mix
  (`W_L`–`W_R`) likewise. For the SM the result equals `weinberg_rotation` /
  `charged_current_rotation` exactly, which remain available.
  `mass_blocks()` with no `kind` now returns scalar, gauge and fermion blocks.
- A scalar component can be expanded around a zero VEV
  (`expand_vev({comp: 0})`, e.g. the neutral component of an inert doublet):
  `Model.tadpoles()` / `solve_tadpoles` used to raise on it. Its stationarity
  condition is reported under the fluctuation symbol only when it does not
  vanish identically.

### Changed
- `examples/thdm.py` uses the guided physical basis; its previous output is
  unchanged and it now also prints the blocks, `tan β`, `m_A²`, `m_{H±}²` and
  the derived charges.
- `examples/sm_u1x.py` gets its gauge rotations from `mass_blocks` /
  `diagonalize` instead of two hand-chained rotations; its physics output is
  unchanged (the intermediate `(Z0, X)` matrix is no longer printed).
- `examples/sm_vll.py` is written with the new constructors; its printed
  physics is unchanged.
- The remaining model-building examples use the new API: `sm_scalar_gauge`,
  `sm_u1x` (now throughout: Yukawas, kinetic terms, parameters, and the h–s
  portal block through `diagonalize`), `sm_vll` (gauge and charged-lepton
  rotations through `diagonalize`), `sm_seesaw`, `sm_weinberg` (which also
  prints `term_coverage(max_dim=5)`), `thdm_s3` and `sm_decays`. Their physics
  output is unchanged. The differences are flavour indices printed as
  `fl_i`/`fl_j`, added summaries and mass-block listings, and larger
  invariance-check counts (`add_kinetic` adds one term per field).
  `fermi_theory`, `sm_ckm`, `ee_to_ff` and `sm_higgs_decays` keep their
  component-level terms (see `examples/README.md`).
- Tutorials: `SM_VLL`, `SM_U1X`, `SM_Seesaw` and `THDM_S3` follow the migrated
  scripts, with the same printed physics. `SM_U1X` replaces the hand-chained
  Weinberg → Z–Z′ rotations with `diagonalize` and checks that the composed
  rotation equals `R₂(θ′)·R₁(θ_W)`. `ModelBuilding` gains a section on
  `Model.most_general`, `term_coverage` and `report.explain()`.
  `SM_Feynman_Rules` keeps its hand-built model and ends with a section that
  asserts the constructors reproduce it.

### Fixed
- `term_coverage` with a flavour-matrix Weinberg coefficient `c[i, j]`: the
  coefficient came back as `(3c_ij + c_ji)/4` with a spurious remainder
  ∝ `c_ij − c_ji`. `ν_iᵀCP_Lν_j` is symmetric in `i ↔ j`, so only the
  symmetric part `(c_ij + c_ji)/2` is physical, and that is now returned with
  no remainder. The basis operators are also put in the same canonical
  flavour-index form as the model terms. Pinned in `tests/test_general.py`.

## [0.3.0] — 2026-10-07

### Added
- `diagonalize_svd(method="auto"|"symbolic"|"numeric", dps=50)`: a numeric
  matrix that is complex, floating-point or larger than 2×2 is decomposed with
  mpmath's complex SVD, returning unitary rotations with
  `R_L M R_R† = diag(m ≥ 0)` (rows by increasing mass, a round-off singular
  value as an exact zero). Generic complex 3×3 Yukawas now give their masses
  and `V_CKM = R_uL·R_dL†` inside feynlag (FG-6, #31). Exact 2×2 calls are unchanged.
- `Rotation.bar(old_bar, new_bar)`: the bar-leg partner of a fermion rotation,
  with the conjugate matrix `R*` (identical to `R` for a real rotation) (#31).
- `GlobalU1`: a global (ungauged) U(1) whose `assign` takes one charge per
  flavour for a fermion (e.g. Froggatt–Nielsen `Q_L: (3, 2, 0)`), checked by
  the new `check_global_invariance` and declared on a model with
  `Model(global_groups=[...])` (`check_invariance`/`validate` report
  `global:<name>`). Not part of the anomaly check (FG-7, #32). Legs sharing a
  symbolic flavour index are charged flavour by flavour, so a flavour-diagonal
  current `ψ̄_i Γ ψ_i` (a kinetic term, `fermion_gauge_current`) is neutral even
  with flavour-dependent charges; an out-of-range integer flavour index, or a
  symbolic index shared by fields with different flavour counts, raises a
  `ValueError`.

### Changed
- The SU(N) groups tutorial (#15) and the Particle Decays tutorial (#30) are
  rebuilt in the derive-then-check style: a prediction before each section,
  and an assert behind every claim instead of printed booleans. They add
  basis-independent invariants, real/complex/pseudo-real representations and
  singlet counting (SU(N)), hand-built Dirac-matrix cross-checks of the
  covariant traces (decays), and web-verified reference lists.

### Fixed
- `diagonalize_svd` no longer returns a silently wrong answer on a complex
  matrix: the symbolic route raises on an explicit `I`. It also no longer
  crashes with `TypeError` on a real floating-point input (#31).

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

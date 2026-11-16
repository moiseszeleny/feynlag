# Examples Gallery

`examples/` contains full worked models, used as both documentation and
manual smoke tests. Each file is runnable end to end
(`python examples/<name>.py`) and doubles as the "how do I actually write
this in feynlag" reference for the corresponding physics.

## `sm_scalar_gauge.py` — the complete Standard Model

Higgs sector + electroweak gauge (SU(2)_L × U(1)_Y) + one lepton generation
+ a full 3-generation quark/QCD sector (SU(2) doublet `QL`, SU(3)_c-triplet
singlets `uR`/`dR`, flavor-generic Yukawas — no CKM diagonalization, since
CKM mixing is orthogonal to SU(3) vertex dynamics). Exercises every stage of
the pipeline: gauge invariance, EWSB, tadpoles, the Higgs/Goldstone/gauge
mass matrices, the Weinberg rotation, fermion gauge currents and Yukawa
mass matrices via the bilinear track, and both LaTeX and UFO export
(including real SU(3) color-tensor strings for qqg/ggg/gggg).

## `sm_vll.py` — vector-like lepton doublet

Adds a vector-like SU(2) doublet Ψ = (N, E) with a bare Dirac mass to the
lepton sector. Demonstrates biunitary 2×2 diagonalization
(`diagonalize_svd_2x2`) of the resulting charged-lepton mass matrix,
fermion mass-basis rotations through `Indexed` fields, and the physics
signature of doublet VLL mixing: a right-handed-only Z FCNC, an h-coupling
sum rule, and the correct SM-decoupling limit as M → ∞. See
{doc}`tutorials/index` for the step-by-step notebook.

## `sm_u1x.py` — SM × U(1)_X with a Z′

Extends the SM by a second abelian factor with symbolic charge assignment
`X = a·Y + b·(B−L)`. An X-charged Higgs singlet VEV drives tree-level
Z–Z′ mixing through two *chained* 2×2 rotations (Weinberg, then Z–Z′). Also
adds 3 right-handed neutrinos with a Dirac Yukawa via H̃. Demonstrates
keeping physics symbolic in an extra charge parameter, Goldstone counting
across two broken U(1)s, and multi-stage rotation chaining
(`Model.physical_lagrangian` applies registered rotations in registration
order). No UFO export — UFO's particle table calls `float()` on charges,
incompatible with the symbolic `a`/`b`.

## `thdm.py` — softly-broken Z₂ 2HDM

The scalar sector of a two-Higgs-doublet model with a softly-broken Z₂
symmetry. Demonstrates the analytic 2×2 orthogonal diagonalization
(`diagonalize_orthogonal_2x2`) and the α mixing angle, cross-checked
against the standard Gunion–Haber/Branco expressions.

## `thdm_s3.py` — 3HDM with an S₃ flavor symmetry

Three Higgs doublets, (H1, H2) forming an S₃ doublet and HS a singlet. The
potential is built entirely from the library's own S₃ Clebsch–Gordan
products (`S3.doublet_product`) rather than hand-written invariants — a
stress test of the discrete-symmetry machinery. The tadpole system is
deliberately over-constrained and forces a specific vacuum alignment
(v₁ = √3 v₂ in the literature basis). See {doc}`tutorials/index` for the
step-by-step notebook.

## `sm_ckm.py` — CKM quark mixing

Quark flavour mixing done the FeynRules-SM way, without a symbolic 3×3
Yukawa SVD. The script works in the quark mass basis with diagonal Yukawas
and inserts the CKM matrix, built from the exact PDG standard
parametrization (`flavor.standard_ckm`, unitary by construction), into the
charged W current only. The neutral Z/γ currents stay flavour-diagonal (the
GIM mechanism), so no FCNCs appear. It ends with a UFO export whose nine
complex `V_ij` round-trip numerically.

## `fermi_theory.py` — four-fermion muon decay

The dim-6 Fermi operator
`−(4G_F/√2)(ν̄_μγ^μP_Lμ)(ēγ_μP_Lν_e) + h.c.`, written below the EW scale
with U(1)_em as the only gauge symmetry. Exercises the four-fermion (FFFF)
track from end to end: the `max_dim=6` invariance check, two-bilinear
extraction, and the `FFFF*` UFO Lorentz structures, exporting both
conjugate vertices as MadGraph requires.

## `sm_weinberg.py` — the dim-5 Weinberg operator

The unique dimension-5 SM operator `(c/Λ)(LᵀCεL)(HᵀεH) + h.c.`. Here
`suggest_yukawa(..., max_dim=5)` enumerates it rather than taking it as
input. After EWSB it gives a Majorana neutrino mass `m_ν = −c v²/Λ`, which
`majorana_mass_matrix` + `diagonalize_takagi` turn into sub-eV masses, plus
the `ν̄νh` coupling. Exercises `diracC` and `MajoranaBilinear`.

## `sm_seesaw.py` — type-I seesaw

The SM plus right-handed neutrinos with a Dirac Yukawa and a large Majorana
mass `M_R`. Builds the `[[0, m_D], [m_Dᵀ, M_R]]` matrix
(`seesaw_mass_matrix`), Takagi-diagonalizes it to a light `m_ν ≈ −m_D²/M_R`
plus a heavy `≈ M_R` spectrum, and uses `MajoranaRotation` to extract the
heavy-neutrino `W`/`Z` couplings. These are proportional to the mixing
`V ≈ m_D/M_R` and decouple as `M_R → ∞`. See {doc}`tutorials/index`.

## `sm_decays.py` — two-body decay widths

Builds the electroweak SM with one lepton generation via the `models.py`
scaffold and runs `feynlag.pheno.DecayCalculator` over the extracted Feynman
rules. Gives symbolic partial widths and branching ratios, plus numbers at
the PDG point: `Γ(Z→νν̄) = m_Z(g²+g'²)/96π`, `Γ(Z→ℓ⁺ℓ⁻)`, `Γ(W→ℓν)` and
`Γ(h→ℓ⁺ℓ⁻)`.

## `sm_higgs_decays.py` — the Higgs branching-ratio table

Every open tree-level `h → f f̄` channel (ττ, μμ, b̄b, c̄c, s̄s), each
declared as one `DiracParticle`, with running quark masses at `m_h`. Adds the
off-shell `h → WW*/ZZ*` widths (1→3, checked against Keung–Marciano) and
the loop-induced `gg`/`γγ`/`Zγ` from imported one-loop form factors. The
result is the canonical BR table, dominated by b̄b. The decays notebook in
{doc}`tutorials/index` walks this physics by hand.

## `ee_to_ff.py` — 2→2 scattering through the Z

`e⁺e⁻ → μ⁺μ⁻` through the Z alone, with the Z-lepton coupling extracted from
the built Lagrangian rather than typed in. Since the leptons are chiral,
the ε (γ₅) forward–backward term is non-zero. The script prints `Σ|M|²`,
`dσ/dcosθ` and `A_FB`, plus a chiral-vs-vector cross-check and the
`M_Z → ∞` contact limit. Companion: the scattering notebook in
{doc}`tutorials/index`.

Each model is described algorithmically, stage by stage, in the
{doc}`manual/pipeline`; the manual cites which pinned test in `tests/`
fixes each physical result these examples produce.

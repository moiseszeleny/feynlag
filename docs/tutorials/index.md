# Tutorials

Ten fully executed Jupyter notebooks, walking a worked model stage by
stage with real (stored) output — plots, mass matrices, Feynman rules. They
are tracked through the `nbstripout --keep-output` git filter (see the
repo's `CLAUDE.md`), so what you see below is exactly what re-running the
notebook produces.

```{toctree}
:maxdepth: 1

Particle_Decays_Tutorial
Scattering_Tutorial
SUN_Groups_Tutorial
DiscreteGroups_Tutorial
SM_Feynman_Rules_Tutorial
SM_VLL_Tutorial
SM_U1X_Tutorial
ModelBuilding_Tutorial
SM_Seesaw_Tutorial
THDM_S3_Tutorial
```

## Particle Decays Tutorial

How to get from a Lagrangian to a measured lifetime, for a student who has
met Feynman rules and Dirac spinors but never carried a decay calculation
through to a number — in the same *derive, then check* style as the group
tutorials: every section predicts a result before a cell settles it, and every
claim is an assertion (46 of them). Derives everything by hand first — two-body
phase space from energy conservation, the spin sums that turn
$|\mathcal{M}|^2$ into a Dirac trace, the trace theorems (the covariant engine
refereed by explicit $4\times4$ Dirac matrices), polarisation sums — and only
then reveals `DecayCalculator` as the automation of exactly those steps,
checking that it reproduces the hand results symbolically. Lands on
$\Gamma(h\to f\bar f) = N_c m_h m_f^2\beta^3/8\pi v^2$ (the $\beta^3$ as the
P-wave signature of a CP-even scalar), $\Gamma(Z\to\nu\bar\nu)$ and
$\Gamma(W\to\ell\nu)$ within ~1% of the PDG, and the Higgs
branching-ratio-versus-mass plot. Then completes the canonical Higgs picture:
the `DiracParticle` fermion sector, the **off-shell $1\to3$** $h\to WW^*/ZZ^*$
(reproducing Keung–Marciano, with a $W^*$ line-shape showing the virtual $W$
never reaches its mass shell), and the loop-induced $gg$/$\gamma\gamma$/$Z\gamma$
from imported one-loop form factors — $b\bar b$, $WW^*$, $gg$, $\tau\tau$,
$c\bar c$, $ZZ^*$ in the measured order. Closes with the two mistakes that fail
*silently* — forgetting that a Dirac fermion is two Weyl fields (an error that
vanishes in the massless limit) and closed channels turning $\sqrt\lambda$
imaginary — builds a $Z'$ from scratch to show the same machinery on a new
model, and ends with a recap of tools and traps and web-verified references.

## Scattering Tutorial

The first native **cross sections**, derived and checked in the same set-up / collect /
recognise / check style as the SU(N) and Decays notebooks: a prediction before most cells
and an `assert` behind every claim.

- **Kinematics and averaging.** It starts from two Mandelstam invariants, and from
  $d\sigma/d\cos\theta$ with the spin average applied once.
- **One photon, computed two ways.** QED $e^+e^-\to\mu^+\mu^-$ through the covariant engine
  *and* with literal $4\times4$ Dirac matrices, against Peskin & Schroeder.
- **Why a 2→2 process keeps a $\gamma_5$ term no decay had.** Three independent momenta,
  shown with Gram determinants. The constants $\kappa=-4i$ and $s_{\det}=-1$ are derived from
  the matrices, not quoted.
- **The Z alone.** Couplings extracted from a Lagrangian give LEP's
  $A_{FB}=\tfrac34A_eA_\mu$.
- **Interference (Tier 3), through `ScatteringCalculator`:**
  - γ+Z reproduces MadGraph's 2.7878 pb once its default $|\eta|<2.5$ cut is applied;
  - the γ–Z cross term is under 1% of σ but drives $A_{FB}$ to 0.57 (vector versus axial
    couplings);
  - Bhabha's s–t interference carries a relative minus sign, which the notebook shows to
    be physics by flipping it.
- **Capstone.** Møller scattering is built by hand from the slot and Wick-sign rules and
  held against the calculator, the crossed Bhabha form, and the identical-particle ½.
- It closes with a Recap (a tools table and the roadmap status) and verified References.

## SU(N) Groups Tutorial

The continuous counterpart to the discrete-groups notebook, for a reader new to
particle physics (linear algebra + basic QM only), in the same *derive, then
check* style: every section predicts a result before a cell settles it, and every
claim is an assertion. The structure constants are derived from the generator
matrices and the adjoint is built from them; Dynkin labels and the Weyl
dimension formula (with the ambiguous **15** of SU(3)); weights and the
Gelfand–Tsetlin ladder of the **6**; what survives a change of basis (weights,
`C₂`, the Dynkin index, anomalies — never generator entries); conjugates
`T̄ = −T*` and the three kinds of reality, including the pseudo-real SU(2)
doublet behind `H̃ = iσ₂H*`. Its core question — which terms can be singlets —
is answered before building them: `C₂` eigenspaces of tensor products, checked
against the null space, give mesons, baryons (the singlet *is* `ε_ijk`) and no
diquarks. Closes with the SU(5) `5̄ + 10` anomaly cancellation, a gauge-invariance
check in the fundamental of SU(4), and a capstone that builds the **6** of SU(3)
by hand from `3 ⊗ 3` and matches it to the library's construction on every
invariant. With a verified reference list.

## Discrete Groups Tutorial

The discrete counterpart to the SU(N) notebook: the finite flavour symmetries
($\mathbb{Z}_N$, $S_3$, $A_4$) that model builders impose to *forbid* terms.
Built around one practical question — **how many free parameters does an
invariant potential have, and can you know before writing a single term?** —
answered by deriving $S_3$'s character table from its generator matrices
(nothing typed in), verifying orthogonality and $\sum_r(\dim r)^2=|G|$, and
then predicting invariant counts by group averaging and checking each against
explicit Clebsch–Gordan construction. Covers the four traps that produce wrong
counts: the antisymmetric $\mathbf{1'}$ vanishing on a repeated multiplet;
**distinct legs versus one field** (four doublets give 3 quartic invariants,
one doublet gives 1 — characters versus the Molien series); basis conventions,
where CG coefficients differ between the real and complex $S_3$ bases but counts
do not; and the conjugate leg $X=(M^{-1})^{\mathsf T}$, which equals $M$ only for
real orthogonal irreps, so the $S_3$ coincidence hides the $\mathbb{Z}_N$ error.
Lands on the 3HDM: imposing $S_3$ cuts the potential from 33 operators to 10
(2 mass + 8 quartic), the number `THDM_S3_Tutorial` builds on. Closes by
defining $A_4$ — the workhorse flavour group, which the library does *not*
ship — on top of `DiscreteSymmetry`, deriving its character table and running
`reynolds_project` and `check_discrete_invariance` on it unchanged.

## SM Feynman Rules Tutorial

Builds the full Standard Model (Higgs, electroweak gauge, leptons, QCD)
from scratch and extracts its Feynman rules, mirroring
`examples/sm_scalar_gauge.py` one pipeline stage at a time.

## SM VLL Tutorial

Adds a vector-like lepton doublet to the SM and walks the biunitary
diagonalization of the resulting mass matrix, including a standalone demo
of why `expand_bilinear` is required for fermion mass-basis rotations to
extract correctly.

## SM U(1)_X Tutorial

Extends the SM by a second, symbolically-charged abelian gauge factor and
walks the chained Weinberg → Z–Z′ rotation that results from tree-level
kinetic/mass mixing.

## Model Building Tutorial

Goes one step earlier than the others: instead of analyzing a
hand-written Lagrangian, it shows the *model-building* tools. For a dark
`U(1)_D` sector with symbolic charges it uses `feynlag.anomalies` to derive
the anomaly-free charge assignment (forcing the dark fermion to be
vector-like), `feynlag.suggest` to enumerate the invariant operator basis
(and catch a mistuned charge that admits no mass term), and
`build_lagrangian` to assemble a validated model before running the full
pipeline to the dark-photon mass and `Z_D χχ` coupling.

## SM Seesaw Tutorial

The Standard Model extended by right-handed neutrinos with a large Majorana
mass — the **type-I seesaw**. Uses the Majorana machinery (`diracC`,
`MajoranaBilinear`, `majorana_mass_matrix`) to build the `[[0, m_D], [m_Dᵀ,
M_R]]` mass matrix, `diagonalize_takagi` for the light (sub-eV) + heavy (~M_R)
spectrum, and the charge-conjugation-aware `MajoranaRotation` to extract the
physical heavy-neutrino couplings — showing `W ℓ̄ N = (g/√2)·V` with the
light–heavy mixing `V ≈ m_D/M_R`, and its decoupling as `M_R → ∞`.

## 3HDM with S₃ Tutorial

The library's group-theory stress test: three Higgs doublets, with
`(H1, H2)` forming an `S3` doublet and `HS` an `S3` singlet, and the
potential built entirely from `S3.doublet_product`'s own
$2\otimes2=1\oplus1'\oplus2$ Clebsch–Gordan decomposition. Introduces the
one genuinely new invariance concept in the whole tutorial set — a
**finite** discrete-symmetry check (`check_discrete_invariance`, the exact
group substitution) rather than the infinitesimal linearization every
gauge check elsewhere relies on — and shows a model where the vacuum
isn't free to tune: with three VEVs but only two independent mass
parameters, the third tadpole condition **forces** the alignment
$v_1^2=v_2^2/3$, derived directly from the symbolic tadpole system rather
than assumed — matching, up to the basis swap, the tadpole solution of the
literature S₃-3HDM model (Gómez-Bock, Mondragón & Pérez-Martínez, EPJC 81,
942 (2021)) this example follows. Builds the pseudoscalar and charged mass
matrices alongside the CP-even one, then implements and verifies that
paper's geometric rotation ansatz: it exactly, symbolically diagonalizes
the Goldstone-protected pseudoscalar/charged sectors for any couplings,
while the CP-even sector needs one further dynamical mixing angle (reusing
the same 2×2 tool `thdm.py` uses for the plain 2HDM) — closing with a
numerical stability scan (mirroring the paper's own) that turns a mostly
tachyonic benchmark point into seven genuine physical scalar masses.

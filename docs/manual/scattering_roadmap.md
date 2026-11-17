# 17. Extending to 2→2 Scattering

## The gap, stated honestly

Everything in `feynlag.pheno` before this chapter computes a **decay**: one
particle in, two (or three, off-shell) out. It cannot compute a **cross
section** at all — there is no notion of two incoming particles, no
Mandelstam invariants, no flux factor, and no way to combine two vertices
joined by a propagator into an amplitude rather than an already-squared
number. The covariant-trace design was chosen partly with this in mind (see
`CLAUDE.md`'s note that it gives "a cleaner path to a later 2→2"), and the
repository already carries two MadGraph-validated cross-section benchmarks —
$\sigma(e^+e^-\to\mu^+\mu^-) = 2.7878\pm0.0027$ pb and
$\sigma(e^+e^-\to W^+W^-) = 19.498\pm0.058$ pb at $\sqrt s = 200$ GeV
(`docs/benchmark.md`) — so the acceptance oracles for a native 2→2 engine
already exist and cost nothing new to obtain.

Two facts drive the whole design that follows.

**The ε (γ₅) term does not vanish for 2→2.** The 1→2 engine's
{func}`~feynlag.pheno.lorentz.reduce_projectors` proves the γ₅ trace term is
zero whenever a chiral fermion current meets at most 2 independent momenta
and 2 free Lorentz indices — true for every decay, because a two-body final
state supplies only $P=p_1+p_2$, one combination, and the polarization sums
it meets are symmetric. A 2→2 process has **3** independent momenta
($k_4=k_1+k_2-k_3$ is not independent), and with two chiral fermion currents
$|\mathcal M|^2$ genuinely contains an $\varepsilon\cdot\varepsilon$ term — the
forward–backward asymmetry, a real, physically necessary piece of the answer,
not a computational nuisance. The guard's threshold was `n_momenta > 3`,
which **passes** at exactly the 2→2 value and would have silently returned a
wrong $d\sigma/d\cos\theta$; Tier 1 tightens it to `n_momenta > 2` (§17.5).

**The 1→2 squared-amplitude functions are terminal and cannot be
extended.** {func}`~feynlag.pheno.amplitudes.ffs_squared`/`ffv_squared`
return an already-squared scalar, with the 1→2 mass-interference term and the
parent's spin average (`ffv_squared`'s hardcoded `/3`) baked in as literals
for that one topology. There is no partial expression to attach a propagator
or a second diagram to. Tier 1 therefore builds the amplitude explicitly —
open fermion chains, propagators, a genuine sum over diagrams — reusing the
covariant engine (`pheno/lorentz.py`) unchanged underneath.

## 17.1 Tier 1 — kinematics, one diagram, no interference ✅ (delivered)

**Channels**: $e^+e^-\to\mu^+\mu^-$ through a single photon (pure QED);
$f_1\bar f_1\to f_2\bar f_2$ through one s-channel scalar. **Effort:
small–medium.**

What Tier 1 deliberately does *not* attempt: more than one diagram (γ/Z
interference — Tier 3). This is a hard guard, not an omission: the engine
raises `NotImplementedError` rather than returning a number that would be
wrong by an $O(1)$ fraction. (Chiral couplings on both vertices of the same
diagram — the ε·ε term above — were Tier 1's other hard guard; §17.2 below
computes it instead.)

### What was built

- {class}`~feynlag.pheno.kinematics.TwoToTwoKinematics` — Mandelstam
  invariants `s` (primary, `positive=True`) and `t` (primary, `real=True`,
  negative throughout the physical region), with `u` a **derived** property
  so momentum conservation holds by construction rather than assertion; the
  ten-entry on-shell `dot` table (including a dedicated momentum head for the
  dependent $k_4$, so conservation is a testable invariant of the table, not
  expression-level algebra colliding with `TensAdd` expansion inside
  {func}`~feynlag.pheno.lorentz.dirac_trace`); the flux factor
  $1/(2\sqrt{\lambda(s,m_1^2,m_2^2)})$; `t_bounds`/`cos_theta`/`t_of_cos`; and
  the $d\sigma/dt$, $d\sigma/d\cos\theta$ conversion factors. `s` and `t` are
  kept as the primary symbols rather than `(s,\cos\theta)` because the `dot`
  table is then linear with no radicals — a $\cos\theta$ parametrization would
  drag $\sqrt{\lambda_i\lambda_f}$ into every table entry and hence into every
  Dirac trace, exactly the reason
  {class}`~feynlag.pheno.kinematics.ThreeBodyKinematics` confines its own
  radicals to `s12_bounds`.
- {mod}`~feynlag.pheno.diagrams` — the amplitude-level object layer:
  {class}`~feynlag.pheno.diagrams.Leg`, {class}`~feynlag.pheno.diagrams.ChainVertex`
  (with `.bar()`, matching {func}`feynlag.dirac.dirac_conjugate`, and
  `.epsilon_coefficient()`), {class}`~feynlag.pheno.diagrams.SpinorChain`
  (the open, γ₅-free trace of one fermion line — diagonal and
  chirality-mixing pieces derived the same way as
  `ffs_squared`/`ffv_squared`, but left open on external Lorentz indices
  instead of immediately reduced), {class}`~feynlag.pheno.diagrams.BosonPropagator`
  and {class}`~feynlag.pheno.diagrams.Diagram`/`Amplitude`. `Amplitude.squared`
  combines two chains through one propagator and reduces to on-shell dot
  products with a single call to {func}`~feynlag.pheno.lorentz.contract_to_dots`
  — never per-chain — matching the library's existing rule that that function
  runs once, on a fully-contracted expression.
- {mod}`~feynlag.pheno.propagator` gained amplitude-level (not
  squared-modulus) propagators — `propagator_denominator`,
  `scalar_propagator`, `vector_propagator` — alongside the existing
  {func}`~feynlag.pheno.propagator.breit_wigner`/`vector_propagator_numerator`,
  which Tier 1 reuses verbatim rather than duplicating.
- {class}`~feynlag.pheno.particles.ExternalState` — spin/colour degrees of
  freedom for an external scattering state, and the home of the averaging
  factor. This is the direct fix for the anti-pattern in
  `ffv_squared`'s hardcoded `/3`: an initial-state spin average masquerading
  as part of the Lorentz algebra, which made that function unusable the
  moment the vector is *internal* rather than the parent.
  {func}`~feynlag.pheno.scattering.average_factor` computes
  $1/\prod_i \mathrm{dof}_i$ from declared data, and
  {func}`~feynlag.pheno.scattering.cross_section`/`differential_cross_section`
  apply it exactly once, never inside a squared amplitude.
- {mod}`~feynlag.pheno.scattering` — `average_factor`,
  `differential_cross_section`, `cross_section` (symbolic `t`-integral), and
  the two worked Tier-1 assemblers, `ffv_s_channel_squared`/
  `ffs_s_channel_squared`, in the same coupling-in/number-out style as
  {func}`~feynlag.pheno.offshell.scalar_offshell_vv_width`.

### `reduce_projectors`: tightened now, retired for chains in Tier 2

`n_momenta > 3` became `n_momenta > 2` in
{func}`~feynlag.pheno.lorentz.reduce_projectors` — the only two callers,
`ffs_squared`/`ffv_squared`, always pass `n_momenta=2`
(a chain's own two legs), so the change is invisible to the 1→2 suite; a
regression test now pins that `n_momenta=3` raises
(`tests/test_pheno.py::test_gamma5_guard_raises_outside_two_body`). This
function still correctly proves each *individual* chain's own ε term is zero
— that reasoning is unaffected by the overall process being 2→2, since a
chain's trace only ever involves its own two legs. What it cannot see is the
**cross-chain** ε·ε term that appears when *two* chiral chains meet through a
propagator; {class}`~feynlag.pheno.diagrams.Amplitude` computes that term
directly now (§17.2, delivered) instead of refusing it, and
`SpinorChain.trace` no longer calls `reduce_projectors` at all — it stays
frozen as the 1→2-only helper `ffs_squared`/`ffv_squared` use
(`tests/test_scattering.py::test_reduce_projectors_is_no_longer_on_the_chain_path`).

### Verified

Against the textbook QED closed form (Peskin & Schroeder §5.1 [PS95])
$\sigma(e^+e^-\to\mu^+\mu^-) = \frac{4\pi\alpha^2}{3s}\cdot\frac{\beta(3-\beta^2)}{2}$,
$\beta=\sqrt{1-4m_\mu^2/s}$ — symbolically exact, plus the massless limit
$4\pi\alpha^2/3s$ — and against an independent explicit-4×4-Dirac-matrix
oracle built in the CM frame (sharing no code with the covariant engine, in
the same spirit as `tests/test_pheno.py`'s `_oracle_*`). At the exact
`docs/benchmark.md` parameter point ($\alpha^{-1}=132.50698$, $\sqrt s=200$
GeV), the QED-only cross section comes out **2.322 pb** over the full
angular range. That is a photon-only number, not a fraction of MadGraph's
2.7878 pb as first written here: the full γ+Z result over the same range is
2.844 pb, and MadGraph's number sits inside its default lepton acceptance
$|\eta|<2.5$ (§17.3). `tests/test_scattering.py` also pins that a
chiral coupling on both vertices genuinely differs from the naive "half the
vector result" guess that *does* hold for a 1→2 decay
(`test_ffv_chiral_is_half_the_vector_result` in `test_pheno.py`) — by up to
$O(1)$, at several angles — which is what justified Tier 1's refusal to
compute that case rather than silently guessing, before §17.2 computed it
for real.

## 17.2 Tier 2 — the ε (γ₅) algebra ✅ (delivered)

**Channels**: $e^+e^-\to\mu^+\mu^-$ through the $Z$ alone; any process with a
single diagram carrying two chiral fermion currents. **Effort: large.**

This landed before Tier 3, not folded into it. It was needed the moment one
diagram has two chiral currents — no interference required — so it could be
validated in isolation, against one closed form
($d\sigma/d\cos\theta \propto (1+\cos^2\theta) + A_{FB}\cos\theta$, with the
sign and magnitude of the forward–backward asymmetry pinned against
[LEPEWWG06]), with exactly one new failure mode. Folded into Tier 3 alongside
multi-diagram interference, a wrong $A_{FB}$ could equally have been the ε
reduction, a relative diagram sign, or fermion-flow bookkeeping — three
candidate bugs instead of one.

### What was built

- {mod}`~feynlag.pheno.epsilon` (new) — the only module with ε knowledge.
  SymPy's own `sympy.physics.hep.gamma_matrices` ships neither a γ₅ object
  nor an ε-aware `gamma_trace` (checked directly against the installed
  version, not assumed), so both identities below are hand-rolled rather
  than delegated to a SymPy primitive:
  - $\mathrm{Tr}[\gamma^a\gamma^b\gamma^c\gamma^d\gamma_5] =
    \kappa\,\varepsilon^{abcd}$ — {func}`~feynlag.pheno.epsilon.gamma5_trace_coefficient`
    reads $\kappa$ off the literal $\gamma^0\gamma^1\gamma^2\gamma^3\gamma_5$
    matrix from `feynlag.dirac._dirac_rep`, against a stated
    $\varepsilon^{0123}=+1$ normalization
    ({func}`~feynlag.pheno.epsilon.levi_civita_array`) — $\kappa=-4i$,
    **derived, not quoted**, the same discipline
    `dirac.majorana_symmetry_sign` uses for its own sign.
  - $\varepsilon^{a\ldots}\varepsilon^{b\ldots} = s_{\det}\!\cdot\!\det[g^{a_ib_j}]$
    — {func}`~feynlag.pheno.epsilon.epsilon_product_sign` derives
    $s_{\det}=-1$ the same way, against the $(+,-,-,-)$ metric.
    {func}`~feynlag.pheno.epsilon.epsilon_pair_tensor` expands this as 24
    signed products of four metrics with all free Lorentz indices left
    **open**, each slot resolved by content (two momenta contract through a
    fresh dummy per pairing; a momentum meeting a free index just evaluates
    the head there; two free indices meet the ordinary `LorentzIndex.metric`)
    — no `Eps` tensor object needed, and the all-momentum case reduces on
    its own to the roadmap's $-\det[p_i\cdot p'_j]$ Gram determinant.
  - {func}`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes` — the
    single-ε cross terms (one chain's ε piece times the other chain's
    ordinary trace) reduce to a scalar times $\varepsilon$(four momenta drawn
    from the diagram's own external legs); since
    $\varepsilon(a,b,c,d)^2=-\det[\cdot]$ (identity 2 again), a vanishing
    Gram determinant over every 4-subset of the diagram's momenta **proves**
    every such term is zero — a computed prove-or-refuse guard reading the
    diagram's actual `kin.dot` table, not a momentum-count proxy that would
    need re-tuning for a future topology.
- {meth}`~feynlag.pheno.diagrams.SpinorChain.epsilon_structure` — a chain's
  ε piece as `(prefactor, slots)` or `None` (a scalar chain, or a
  pure-vector `g_L=g_R` chain, has none — exactly the cases the pre-Tier-2
  engine already handled correctly, so those results are unchanged
  bit-for-bit). `SpinorChain.trace` no longer calls `reduce_projectors` —
  see the retirement note above.
- {meth}`~feynlag.pheno.diagrams.Amplitude.squared` assembles the ε
  contribution alongside the existing non-ε piece — both are separately
  fully-contracted scalars before being summed, so the house rule that
  {func}`~feynlag.pheno.lorentz.contract_to_dots` runs **exactly once**, on a
  fully-contracted expression, survives literally. Contracting the two
  propagator numerators against the ε tensor stepwise (one numerator at a
  time, with `.expand()` between them) is ~3.5× faster than contracting both
  at once for a massive mediator.
- {func}`~feynlag.pheno.scattering.forward_backward_asymmetry` — the Tier-2
  acceptance observable, $A_{FB}=(\sigma_F-\sigma_B)/(\sigma_F+\sigma_B)$
  over the $\cos\theta\gtrless0$ hemispheres. Needs no
  {class}`~feynlag.pheno.particles.ExternalState`\ s, unlike
  `differential_cross_section`/`cross_section`: the spin average, any
  identical-particle symmetry factor, and the $d\sigma/d\cos\theta$
  conversion factor are all $\cos\theta$-independent and cancel exactly in
  the ratio.

### Verified

The headline test, `test_chiral_squared_amplitude_matches_explicit_matrix_oracle`:
with **fully symbolic**, independent chiral couplings $g_L,g_R,h_L,h_R$ on
both vertices (not the pure-vector case, where the ε term is trivially zero)
through a massless mediator, the covariant engine matches the independent
explicit-4×4-matrix oracle exactly (`tests/test_scattering.py`'s
`_oracle_qed_general`) — this one identity simultaneously fixes $\kappa$, the
$\mp$ sign in the $P_{L,R}$ split, and the $\varepsilon_{0123}$ convention,
and its full generality is what shows the single-ε cross terms genuinely
cancel rather than happening to vanish at one lucky point.
`test_z_massive_mediator_matches_explicit_matrix_oracle` extends the oracle
to a massive mediator (the only test needing that extension) and checks
`vector_propagator_numerator`'s sign convention composes correctly with the ε
piece; `test_massive_mediator_epsilon_piece_drops_the_qq_term` separately
confirms the propagator's $q_\mu q_\nu/M^2$ term drops out of the ε
contribution automatically (a repeated momentum in the same ε ⟹ two equal
Gram-determinant rows ⟹ 0), for four symbolic external masses.

Against the literature: the massless-final-state angular distribution
matches the LEP Born-level form,
$\Sigma|\mathcal M|^2=(g_L^2+g_R^2)(h_L^2+h_R^2)(1+\cos^2\theta)+2(g_L^2-g_R^2)(h_L^2-h_R^2)\cos\theta$
(Eq. (1.55) of [LEPEWWG06] at zero beam polarization), and
`forward_backward_asymmetry` reproduces
$A_{FB}^{0,f}=\tfrac34A_eA_f$ (Eq. (1.66)) symbolically, plus a numeric pin
at the LEP effective weak mixing angle
$\sin^2\theta_{\mathrm{eff}}^{\mathrm{lept}}=0.23153$ giving the standard
leptonic $A_\ell\approx0.147$, $A_{FB}^{0,\ell}\approx0.016$.
`test_z_only_total_cross_section_is_epsilon_independent` confirms the ε term
is odd in $\cos\theta$ and integrates away over the full range — a chiral
mediator's *total* cross section equals a pure-vector coupling's with the
same $g_L^2+g_R^2$, so Tier 1's 2.322 pb QED benchmark cannot shift from
Tier 2 landing; $A_{FB}$ is an angular observable only.

## 17.3 Tier 3 — interference and the first cross-section benchmark ✅ (delivered)

**Channels**: the full $e^+e^-\to f\bar f$ ($\gamma$/$Z$ interference), Bhabha
$e^+e^-\to e^+e^-$ (s × t), Møller $e^-e^-\to e^-e^-$ (t × u), $\nu_e e^-\to\nu_e
e^-$ (Z × W); s-, t-, u-channel topology enumeration; a `ScatteringCalculator`
mirroring {class}`~feynlag.pheno.calculator.DecayCalculator`. **Effort:
medium–large.**

### What was built

- {meth}`~feynlag.pheno.diagrams.Amplitude.squared` drops its
  `len(diagrams) == 1` guard and performs the full $\sum_{d,d'}M_d\bar M_{d'}$
  double sum. A diagonal term keeps the exact Tier-1/2 code path, so every
  single-diagram result is unchanged. An off-diagonal pair is classified by
  how its two fermion lines pair the external legs:
  - **same pairing** (γ and Z both in the s channel): still a product of two
    open traces, one per line, now each with $\Gamma$ from $d$ and $\bar\Gamma$
    from $d'$ ({meth}`~feynlag.pheno.diagrams.SpinorChain.cross_trace`). The
    ε·ε piece is assembled exactly as in Tier 2.
  - **exchanged pairing** (Bhabha's s × t, Møller's t × u, the Z × W of
    $\nu_ee$): the lines fuse into **one** trace through all four legs,
    $\mathrm{Tr}[S_{o_1}\Gamma_aS_{i_1}\bar\Gamma'S_{o_2}\Gamma_bS_{i_2}\bar\Gamma']$.
    Every Lorentz index of it is contracted with one of the two propagator
    numerators. So its γ₅ part is a sum of ε tensors of the diagram's own
    momenta, which the Tier-2 Gram-determinant guard
    ({func}`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes`) **proves**
    zero before it is dropped.
- The cross terms need projector algebra the hand-derived Tier-1 trace never
  did (a $\Gamma_d$ against a different $\bar\Gamma_{d'}$, including a scalar
  against a vector). `diagrams._chiral_trace` does it mechanically: it splits
  every spin sum and vertex, pushes each projector to the end of the string
  ($P\gamma=\gamma\bar P$, $PP'=\delta_{PP'}P$), then uses
  $\mathrm{Tr}[XP_{L,R}]=\tfrac12\mathrm{Tr}[X]\mp\tfrac12\mathrm{Tr}[X\gamma_5]$.
  `cross_trace(self)` reproduces the hand-derived `trace()` and
  `epsilon_structure()`, a pinned identity.
- **Two SymPy 1.14 defects** surfaced on the exchange trace, and it now
  avoids SymPy's tracer entirely:
  - `gamma_trace` returns a **wrong** value when one momentum is slashed
    twice in a string: $\mathrm{Tr}[\not q\gamma^a\not p\not p\gamma^b\not q]g_{ab}$ comes out
    $72p^2q^2$, not $16p^2q^2$. This is exactly what a massive propagator's
    $\not q=\not k_1+\not k_2$ next to the spin sum of $k_1$ produces. It first
    showed up as a $4\times10^{-5}$ disagreement with the oracle.
  - `kahane_simplify` raises `Repeated index` on valid interleaved
    contractions such as $\gamma^a\gamma^b\not k_3\gamma_a\not k_4\gamma_b$.

  The exchange route therefore removes each contracted
  $\gamma^\rho\cdots\gamma_\rho$ pair with the four-dimensional contraction
  identities. It then reduces the slash-only string recursively, straight to
  dot products. That is also about 700× faster: 104 s → 0.14 s for the
  exchange terms of a massless chiral-Z Bhabha.
  `diagrams._string_tensor` refuses a repeated momentum rather than hand it
  to SymPy.
- {class}`~feynlag.pheno.diagrams.BosonPropagator` gains `phase()`
  ($+i$ scalar, $-i$ vector) and `denominator()`. Both cancel in a diagonal
  term. In a cross term they give the relative phase of a scalar- and a
  vector-mediated diagram, and the complex $1/(D_d\bar D_{d'})$ of two
  Breit–Wigners.
- {mod}`~feynlag.pheno.topology` — {func}`~feynlag.pheno.topology.enumerate_diagrams`,
  the 2→2 analogue of `DecayCalculator.channels`:
  - Each external fermion takes the bar or field slot of its line, by the
    standard external-line table.
  - Both pairings of the two bar and two field slots are tried. A line
    $(X,Y)$ needs a `DecayVertex` $\bar\psi_X\Gamma\psi_Y$.
  - The two lines are joined by a boson $B$ on one and $B^\dagger$ on the
    other. Charged mediators use an explicit `conjugates={Wp: Wm, …}` map,
    the `check_hermiticity_pairing` convention.
  - **Fermion sign** = parity of the permutation
    $(\text{bar}_A,\text{field}_A,\text{bar}_B,\text{field}_B)$ of the
    external labels (Wick's theorem; each bilinear is bosonic, so the two
    lines' order is irrelevant).
- {class}`~feynlag.pheno.scattering_calculator.ScatteringCalculator` /
  `ScatteringProcess`:
  - Model + `DiracParticle`s + mediator masses/widths in; diagrams, `Σ|M|²`,
    `dσ/dcosθ`, `σ` and `A_FB` out.
  - The identical-final-state ½ and the spin averaging are applied once
    (`ExternalState`).
  - `process(..., numeric=True)` substitutes the parameter point into the
    couplings before the Dirac algebra.
  - `numeric_cross_section(s, cos_range)` integrates by quadrature
    (`integrate.quad_1d`). SymPy's symbolic integral of a Bhabha $|M|^2$
    with complex s- and t-channel Breit–Wigners is slow.
  - Coloured external states raise (colour flow is Tier 5). An undeclared
    mediator mass raises rather than defaulting to massless.
- {func}`~feynlag.pheno.scattering.cross_section` gains `cos_range`, an
  angular acceptance. This is needed both for a t-channel photon, whose
  full-angle integral diverges, and for the generator benchmark below.

### Verified

The independent oracle is an **explicit-spinor helicity evaluator**
(`tests/test_interference.py::_oracle`). It builds $u$/$v$ as the
eigenvectors of $(\not p\pm m)\gamma^0$ and sandwiches literal 4×4 vertex
matrices. It then sums *amplitudes* with explicit propagators and squares.
There are no traces, no projector algebra and no ε identities, so it shares
no code with the engine. Against it, the engine matches to $10^{-10}$ for:

- γ+Z+h in the s channel with massive final-state fermions, a Z width and
  independent chiral couplings. This includes the vector × scalar cross
  terms, which survive only through the masses.
- Bhabha with γ+Z, massive fermions and chiral couplings: the hardest
  exchange case.
- Bhabha built end to end from the Lagrangian by the calculator.

Against the literature and derived identities:

- **Bhabha**, massless QED: exactly the spin-summed form of Peskin &
  Schroeder Problem 5.2 [PS95],
  $\Sigma|\mathcal M|^2=8e^4[(s^2+u^2)/t^2+2u^2/(st)+(t^2+u^2)/s^2]$. With the
  s/t relative sign flipped to $+1$ the result differs by exactly
  $32e^4u^2/(st)$: the sign is physics, not convention.
- **Møller** through the photon: the same form crossed $s\leftrightarrow u$,
  with the calculator's identical-particle ½.
- **$\nu_ee^-\to\nu_ee^-$** in the contact limit. Fierz-rearranging the
  u-channel W into the t-channel structure shifts the electron's left-handed
  Z coupling $(g/c_W)(-\tfrac12+s_W^2)\to(g/c_W)(+\tfrac12+s_W^2)$, the
  familiar "$g_L+1$". This pins the W–Z relative sign that the permutation
  parity assigns; the opposite sign would give $-\tfrac32+s_W^2$.

**Oracle**: $\sigma(e^+e^-\to\mu^+\mu^-)$ with the γ and Z couplings extracted
from the Lagrangian, at the exact `docs/benchmark.md` parameter point:

- Over the full angular range it is **2.8443 pb**, against an independent
  helicity-amplitude closed form to $10^{-9}$. That is 2% (about 20σ) above
  MadGraph's $2.7878\pm0.0027$ pb.
- MadGraph's run-card default applies a lepton pseudorapidity cut
  $|\eta|<2.5$ ($|\cos\theta|<\tanh2.5$ for massless leptons). Inside that
  acceptance feynlag gives **2.7876 pb**, 0.1σ from MadGraph.

The QED-only 2.322 pb of Tier 1 is the photon diagram alone over the full
range. At $\sqrt s=200$ GeV the γ–Z interference also drives $A_{FB}$ to
**0.565**, against $\tfrac34A_eA_\mu\approx0.016$ for the Z alone on the pole.
`examples/ee_scattering.py` prints all of this, plus Bhabha (963 pb inside
$|\eta|<2.5$).

## 17.4 Tier 4 — external bosons, fermion exchange and derivative couplings

**Channels**: $e^+e^-\to\gamma\gamma$, Compton scattering, $e^+e^-\to ZZ/Z\gamma$
(stage 4a, delivered); $e^+e^-\to Zh$ and $e^+e^-\to W^+W^-$ (stage 4b,
delivered); processes through a `VSS` vertex, e.g. $e^+e^-\to\phi^+\phi^-$
(stage 4c, delivered). **Tier 4 is complete.**

The target is $e^+e^-\to W^+W^-$, and it needs four things Tiers 1–3 do not
have. Only the last is a derivative coupling:

1. **External bosons.** Tier 3's enumerator takes four fermions.
2. **Two vertices on one fermion line**, joined by a fermion propagator (the
   $\nu$ exchange).
3. **External-vector polarization sums** inside a 2→2 $|\mathcal M|^2$.
4. **Bosonic three-point vertices as Lorentz tensors.** `VVV` comes from
   {meth}`~feynlag.lagrangian.Model.gauge_vertices` as a scalar coefficient
   times the universal $g^{\mu\nu}(p_1-p_2)^\rho+\text{cyclic}$, with the
   momenta implied by the leg order; `VVS` is $g^{\mu\nu}$; `VSS` is the one
   whose coupling carries explicit `p(φ)` momentum tags from
   {func}`~feynlag.operators.to_momentum_space`.

Items 1–3 need no derivative coupling and have textbook closed forms, so the
tier is staged: each piece lands against its own oracle before the $WW$
benchmark depends on all of them at once.

### Stage 4a — fermion exchange with external vectors ✅ (delivered)

A process with two external fermions and two external bosons has **one open
fermion line** in every diagram, so there is no fermion-permutation sign; the
relative phases are those of the Feynman rules (the couplings' $i$, $+i$ for a
fermion or scalar propagator, $-i$ for a vector one).

- {mod}`feynlag.pheno.line` keeps the amplitude as a list of
  {class}`~feynlag.pheno.line.LineTerm`\ s — a Dirac string between the two
  spinors, loose tensors on the open vector indices, a scalar prefactor — and
  {meth}`~feynlag.pheno.line.LineAmplitude.squared` evaluates
  $\sum_{a,b}c_a\bar c_b\,\mathrm{Tr}[S\,\Gamma_a\,S\,\bar\Gamma_b]\prod_V P_V$.
- {mod}`feynlag.pheno.contraction` reduces that to dot products at the scalar
  level. Tier 3's `_expand_chiral_string → _contract_pairs → _slash_trace`
  chain moved there, and {func}`~feynlag.pheno.contraction.contract_trace`
  adds the step in front: a loose metric identifies its two indices, a loose
  vector meeting a $\gamma$ becomes a slash, two vectors a dot product. No
  SymPy tensor is built — a squared fermion-exchange diagram slashes the same
  momentum twice, which `gamma_trace` (SymPy 1.14) mis-evaluates.
- **The $\gamma_5$ part** of a fully contracted trace is a pseudoscalar of the
  process's own momenta. A 2→2 process has three independent ones, so it
  vanishes; `assert_epsilon_single_vanishes` proves that on the actual `dot`
  table before the coefficients are dropped.
- **Polarization sums.** A massive vector uses $-g_{\mu\mu'}+k_\mu k_{\mu'}/m^2$.
  A massless one uses $-g_{\mu\mu'}$, and the engine *checks* the Ward
  identity that justifies it: it replaces that leg's sum by $k_\mu k_{\mu'}$
  and requires zero, raising otherwise (one diagram of $e^+e^-\to\gamma\gamma$
  alone fails it; two external non-abelian massless vectors will, and need
  ghosts — Tier 5).
- **External bosons** follow the Wick rule Tier 3 uses for mediators: an
  incoming particle $B$ is annihilated by the field $B$, an outgoing one is
  created by $B^\dagger$ (`conjugates[B]`).
  {func}`~feynlag.pheno.topology.enumerate_line_diagrams` enumerates both
  attachments of the two bosons for every declared exchanged species, with
  the propagator momentum along the fermion arrow.
- {class}`~feynlag.pheno.particles.DiracParticle` accepts `right=None`, so
  the left-handed neutrino is a declarable species.
- **Refused, not approximated**: an exchanged fermion no `DiracParticle`
  claims raises instead of dropping its diagram. (Stage 4a also refused any
  process with a boson-exchange diagram; stage 4b computes those.)

**Verified** (`tests/test_line_scattering.py`): $e^+e^-\to\gamma\gamma$ and
Compton scattering with a **massive** electron match the Klein–Nishina form
and its crossing symbolically [PS95]; a general case (massive external
fermion, a different exchanged mass, two distinct massive vectors with
independent chiral couplings) matches an explicit spinor + polarization-vector
evaluator that shares no code with the engine; from the SM model,
$e^+e^-\to\gamma\gamma$, $ZZ$, $Z\gamma$, $\nu\bar\nu\to ZZ$ and Compton match the
same references evaluated with textbook couplings.

### Stage 4b — bosonic three-point vertices and the acid test ✅ (delivered)

The fermion line emits one boson, which meets the two external bosons at a
bosonic three-point vertex.

- {mod}`feynlag.pheno.boson_vertices` writes the Lorentz structures out as
  loose tensors: `sss_tensor`, `vvs_tensor` ($g^{\mu\nu}$) and `vvv_tensor`.
- **The triple-gauge rule.**
  {meth}`~feynlag.lagrangian.Model.gauge_vertices` returns
  $c=-gF_{ijk}$ for one canonical ordering of the field legs. Differentiating
  $-\tfrac14F^a_{\mu\nu}F^{a\mu\nu}$ with $D_\mu=\partial_\mu-igT^aA^a_\mu$ and
  incoming plane waves gives
  $i\,\delta^3\mathcal L=gF_{ijk}\,[g^{\mu\nu}(p_1-p_2)^\rho+\text{cyclic}]$, so the
  vertex is $\kappa\,c\,[\dots]$ with $\kappa=-1$ (`VVV_KAPPA`), each $p_i$ the
  momentum flowing into the vertex with field leg $i$. A test redoes the
  differentiation and finds the same $\kappa$ for $AWW$ and $ZWW$.
- **No sign is chosen by hand.** A leg is attached by the field symbol it
  carries (stage 4a's Wick rule: the outgoing $W^+$ sits on the $W^-$ field
  leg) with its own incoming momentum; the mediator leg carries
  $p_{\rm in}(\bar\psi)+p_{\rm in}(\psi)$. This is not the field→particle
  relabelling of `export/ufo/legs.py`, which is a statement about UFO leg
  names and is not used here.
- {func}`~feynlag.pheno.line.boson_exchange_terms` multiplies the line's
  vertex, the propagator ($i/D$ scalar; $-i(g_{\rho\sigma}-q_\rho q_\sigma/m^2)/D$
  vector, unitary gauge) and the vertex tensor;
  {func}`~feynlag.pheno.topology.enumerate_line_diagrams` adds these
  diagrams to the fermion-exchange ones.
- `ScatteringCalculator(gauge_groups=[SU2L])` collects the `VVV` vertices. A
  charged external vector **without** them raises: the ν exchange alone is
  the unitarity-violating wrong answer.
- The calculator now evaluates bosonic couplings **at zero fields**. With
  the Goldstones left out of `boson_fields` (unitary gauge) the extractor
  reported four-leg terms as three-leg "vertices" such as $A\,h\,W^+$ with a
  coupling $\propto G^-$; these are dropped.

**Verified** (`tests/test_ww.py`):

- $\sigma(e^+e^-\to W^+W^-)=19.447$ pb at $\sqrt s=200$ GeV against
  MadGraph's $19.498\pm0.058$ pb (0.9σ; `docs/benchmark.md`), with every
  coupling extracted from the model.
- $\sum|\mathcal M|^2$ matches an explicit spinor + polarization-vector
  evaluator with hand-written Feynman rules, at three kinematic points.
- **The gauge cancellation, exactly.** At a rational parameter point
  ($g=4$, $g'=3$, $v=1$, so $s_w=3/5$, $M_W=2$, $M_Z=5/2$) and along $t=-s/3$,
  $\sum|\mathcal M|^2$ is an exact rational function of $s$: the ν exchange
  alone grows like $s^2$, the sum of the three diagrams is bounded. Flipping
  $\kappa$ restores the $s^2$ growth and gives $\approx98$ pb at 200 GeV, the
  failure the benchmark notes describe.
- $\sigma(e^+e^-\to Zh)$ matches the Higgsstrahlung closed form
  [Djouadi08, Eq. 4.18] to $10^{-7}$ (240 fb at $\sqrt s=250$ GeV).

Worked example: `examples/ee_to_ww.py`.

### Stage 4c — `VSS` momentum tags and external scalars ✅ (delivered)

`feynlag.pheno.boson_vertices.vss_tensor`, wired into
{func}`~feynlag.pheno.topology.enumerate_line_diagrams`.

**The rule.** The extractor's `VSS` coupling is linear in the momentum tags
of its two scalar legs, $\sum_\phi c_\phi\,p(\phi)$, each $c_\phi$ already
carrying the Feynman-rule $i$. The tags were written with feynlag's
$\partial_\mu\phi\to+i\,p(\phi)\phi$; a physical particle entering on field
leg $\phi$ with momentum $k$ is $e^{-ik\cdot x}$, so each tag becomes
$\kappa_{VSS}\,k^\mu$ with $\kappa_{VSS}=-1$ (`VSS_KAPPA`), on the vector
leg's index. For a scalar of charge $Q$ under $D_\mu=\partial_\mu-igQA_\mu$
this is $i g Q\,(p_\phi-p_{\phi^\dagger})^\mu$ with both momenta flowing in.
As for the triple-gauge vertex, a leg is attached by the field symbol it
carries (the outgoing $\phi^+$ sits on the $\phi$ leg), so nothing else is
chosen by hand. A coupling that is not exactly linear in the two legs' tags,
or two scalar legs with the same symbol, raises.

Two diagram shapes come out of the same code: a vector mediator with two
external scalars ($e^+e^-\to\phi^+\phi^-$), and a scalar mediator with one
external vector and one external scalar ($e^-\gamma\to\chi\phi^-$), where
the mediator's tag resolves to $p_{\rm in}(\bar\psi)+p_{\rm in}(\psi)$.

**Three things the test model exposed** (scalar QED with a Yukawa: a U(1),
an electron, a neutral fermion $\chi$, a charged scalar):

- `Model.physical_lagrangian` raised for a model with no VEV. An unbroken
  model has nothing to shift; it now returns the Lagrangian as written.
- `collect_decay_vertices` did not apply `conjugate_map` on the fermion
  track, so the $\bar\psi\psi\phi^\dagger$ half of a charged-scalar Yukawa
  was silently missing.
- `ScatteringCalculator`'s stray-field filter (stage 4b) read the scalar
  symbols inside the tags as field dependence; tags are now opaque to it.

**Verified** (`tests/test_vss.py`):

- **The tag rule**, by putting plane waves $e^{-ik\cdot x}$ on the kinetic
  term directly ($\partial_\mu f\to-ik_f f$), bypassing
  `to_momentum_space` and the extractor.
- **$\sigma(e^+e^-\to\phi^+\phi^-)=\pi\alpha^2\beta^3/(3s)$**, the
  point-like scalar Born cross section [BFIK05], to $10^{-9}$ at three
  energies. One diagram: this fixes the normalisation and the $\beta^3$
  threshold shape, not the sign.
- **The sign, exactly, by the Ward identity.** $e^-\gamma\to\chi\phi^-$ has
  an electron-exchange diagram and a $\phi$-exchange diagram through
  $\gamma\phi\phi$; $k_\gamma\cdot\mathcal M=0$ only for the right relative
  sign, and the engine checks it before using $-g$ for the photon. With
  $\kappa_{VSS}$ flipped the check fails.
- **An explicit spinor evaluator** with hand-written rules, for both
  processes (massless and massive electron) and for a massive vector with a
  width meeting two scalars of different mass, where the propagator's
  $q_\mu q_\nu/M^2$ term contributes.

Worked example: `examples/scalar_qed.py`.

**Still outside Tier 4**: `VSS` in 1→2 decays (`amplitudes.py` raises),
four-point contact vertices, boson–boson scattering.

## 17.5 Tier 5 — coloured / hadronic 2→2 (parton level)

**Channels**: four-quark scattering and quark–lepton processes (stage 5a,
delivered); $q\bar q\to gg$, $qg\to qg$ (stage 5b, delivered); $gg\to gg$
(stage 5c, delivered).
**Effort: medium–large.** Parton level only: **no PDFs, no hadronic
luminosity, fixed $\alpha_s$** — the same "no proton structure" scope as the
rest of the library.

Three separate things are missing after Tier 4, and only the first is a
colour factor:

1. **Colour bookkeeping** for diagrams and their interference.
2. **External gluons.** Two external massless non-abelian vectors do not
   obey the abelian Ward identity diagram set by diagram set, so $-g$ is not
   their polarization sum; the engine already checks this and refuses
   (stage 4a). They need physical polarization sums.
3. **A four-boson amplitude** for $gg\to gg$: three exchange diagrams and the
   $gggg$ contact vertex (already assembled, colour-stripped, by
   `export/ufo/vvvv.py`'s `adjoint_vvvv`). Nothing in `pheno` has an
   amplitude without a fermion line.

So the tier is staged like Tier 4, one oracle per stage.

### Stage 5a — colour, and coloured four-fermion processes ✅ (delivered)

`feynlag.pheno.colour`, `Amplitude.squared(kin, colour=)`, and
`ScatteringCalculator(colour_group=, gluon=)`.

**Colour is factorised, not enumerated.** Every diagram is a colour tensor
$K_d$ over the four external colour indices times the Lorentz amplitude
$L_d$ the Tier 1–3 engine computes with a colour-stripped coupling, so

$$\sum_{\rm colours}\Big|\sum_d K_dL_d\Big|^2=\sum_{d,d'}C_{dd'}\,L_d\bar L_{d'},
\qquad C_{dd'}=\sum_{\rm colours}K_d\bar K_{d'} .$$

A fermion line $\bar\psi_x\ldots\psi_y$ carries $T^a_{xy}$ when it emits the
gluon and $\delta_{xy}$ when it emits a colour singlet, with $T^a$ taken from
{meth}`~feynlag.groups.gauge.SUN.generators` in the field's rep. No Casimir
is typed in: for SU(3) the four-quark matrix elements $2$
($=\mathrm{Tr}(T^aT^b)\mathrm{Tr}(T^aT^b)$) and $-\tfrac23$
($=\mathrm{Tr}(T^aT^bT^aT^b)$), and the vanishing of a singlet × octet
s channel ($\mathrm{Tr}\,T^a=0$), come out of the index sums, for any SU(N)
and any irrep. The Dirac algebra is untouched: `Amplitude.squared` already
computes each $M_d\bar M_{d'}$ separately and now multiplies it by
$C_{dd'}$; without a colour matrix every earlier result is unchanged.

**Declaring a quark.** Colour is explicit in feynlag's components (a triplet
is three `IndexedBase` components, the gluon eight symbols), and the
extractor reports $(\bar q_r,q_c,G_a)$ with coefficient $g_sT^a_{rc}$. A
coloured species is declared as the decay calculator already does it: a
`DiracParticle` on **one** colour component, with `color=3`. Colour-singlet
vertices are then right as extracted. For the gluon, the calculator extracts
over the gluon components itself and replaces what it finds by **one** vertex
$(\bar q,q,g)$ on a single mediator symbol (`calc.gluon`), with the stripped
coupling $\text{coefficient}/T^a_{rr}$ read from the diagonal generators. It
requires every non-zero diagonal generator to give the same value — that
identifies the component $r$ and checks the vertex really is $g\,T^a$ — and
raises otherwise.

**Refused, not approximated**: a coloured species without `colour_group`; a
declared `color` that disagrees with the field's rep; a fermion line joining
different colour dimensions through a singlet (a leptoquark-like coloured
mediator); an external gluon, and coloured fermions with external bosons
(stage 5b).

**Fixed along the way**: the γ₅ guard refused valid processes with float
masses, because the Gram determinant of four dependent momenta then survives
as a round-off residue instead of an exact zero. Float-valued tables are now
tested for singularity numerically (against Hadamard's bound); exact input
is handled as before.

**Verified** (`tests/test_colour.py`):

- **The colour matrices** against the SU(N) trace identities for
  $N=2,3,4$: $(N^2-1)/4$, $-(N^2-1)/(4N)$, $N^2$, $N$, $(N^2-1)/2$ and $0$.
- **The parton table [ESW96, Table 7.1]**, exactly and symbolically, with
  the couplings extracted from a QCD × QED model (averaged over initial and
  summed over final spins and colours, $g_s^4$ removed):
  $qq'\to qq'$ and $q\bar q'\to q\bar q'$: $\tfrac49\frac{s^2+u^2}{t^2}$;
  $q\bar q\to q'\bar q'$: $\tfrac49\frac{t^2+u^2}{s^2}$;
  $qq\to qq$: $\tfrac49\big(\frac{s^2+u^2}{t^2}+\frac{s^2+t^2}{u^2}\big)-\tfrac{8}{27}\frac{s^2}{tu}$;
  $q\bar q\to q\bar q$: $\tfrac49\big(\frac{s^2+u^2}{t^2}+\frac{t^2+u^2}{s^2}\big)-\tfrac{8}{27}\frac{u^2}{st}$.
  All five matched on the first run.
- **QCD and QED together, massive quarks**: $u\bar u\to u\bar u$ with photon
  and gluon in both channels, against an explicit evaluator that sums the
  amplitude over diagrams for each colour and spin configuration with
  explicit $T^a$ matrices and spinors — no colour matrix and no trace.
- **Colour-singlet limits**: $\sigma(e^+e^-\to q\bar q)=N_cQ_q^2\cdot4\pi\alpha^2/3s$
  and $\sigma(q\bar q\to e^+e^-)=(Q_q^2/N_c)\cdot4\pi\alpha^2/3s$;
  $\sigma(q\bar q\to q'\bar q')=8\pi\alpha_s^2/27s$.

Worked example: `examples/qcd_scattering.py`.

### Stage 5b — external gluons ✅ (delivered)

$q\bar q\to gg$, $gg\to q\bar q$, $qg\to qg$ and the mixed
$q\bar q\to g\gamma$, on Tier 4's line engine. The external state is the
calculator's gluon symbol.

**Physical polarization sums.** For two external gluons $-g_{\mu\nu}$ is
wrong: the amplitude satisfies $k_\mu\mathcal M^{\mu\nu}\propto k'^\nu$, not
$0$, so unphysical polarizations survive (in a covariant gauge the ghosts
cancel them). `VectorLeg(reference=n)` sums over the two physical ones,

$$\sum\varepsilon_\mu\varepsilon^*_\nu=-g_{\mu\nu}+\frac{k_\mu n_\nu+n_\mu k_\nu}{k\cdot n},
\qquad n^2=0 ,$$

and the calculator gives each of two external gluons the other's momentum as
$n$. The extra pieces are ordinary loose tensors, so the contraction code is
unchanged. A single gluon (with a photon) keeps $-g$: there is no three-boson
diagram and the abelian identity holds.

**Colour.** `line_colour_tensor` over (bar-slot colour, field-slot colour,
boson 1, boson 2): $(T^aT^b)_{xy}$ or $(T^bT^a)_{xy}$ for the two
quark-exchange diagrams, in line order, and $\sum_cT^c_{xy}f^{cab}$ for the
gluon exchange. The three-gluon vertex is one `VVV` vertex on the single
gluon symbol with the stripped coupling $-g$ (`adjoint_vvv`); its legs are
ordered (mediator, boson 1, boson 2) for both $f$ and the Lorentz tensor, so
the two antisymmetries stay aligned and $\kappa=-1$ from stage 4b applies
unchanged. `LineAmplitude.squared(colour=)` weights each pair of diagrams.

**A hole in the Ward guard, found and closed.** Stages 4a–4c relied on the
computed Ward check to refuse an invalid $-g$ sum. For two non-abelian
vectors both summed with $-g$ it did not fire: the residue
$k_\mu\mathcal M^{\mu\nu}\propto k'^\nu$ is squared by $-g$ to $k'^2=0$. The
check now also contracts every other $-g$ leg with the probed leg's own
momentum, which does see it. Nothing shipped before had two external
non-abelian massless vectors, and the calculator never uses $-g$ for two
gluons; the fix matters for hand-built amplitudes.

**Verified** (`tests/test_gluons.py`):

- **The parton table [ESW96, Table 7.1]**, exactly and symbolically:
  $q\bar q\to gg=\tfrac{32}{27}\frac{t^2+u^2}{tu}-\tfrac83\frac{t^2+u^2}{s^2}$,
  $gg\to q\bar q=\tfrac16\frac{t^2+u^2}{tu}-\tfrac38\frac{t^2+u^2}{s^2}$,
  $qg\to qg=\frac{s^2+u^2}{t^2}-\tfrac49\frac{s^2+u^2}{su}$.
  All three matched on the first run.
- **Massive quarks** ($q\bar q\to gg$, $gg\to Q\bar Q$) against an explicit
  evaluator — spinors, transverse polarization vectors, $T^a$ and $f^{abc}$,
  the amplitude summed per colour and spin configuration — which first
  checks its own Ward identity.
- **Gauge independence**: quark momenta as reference vectors give the same
  $\sum|\mathcal M|^2$.
- **Controls**: with $\kappa$ flipped the Ward check fails; with $-g$ on both
  gluons it fails too, and forcing it gives a different number.
- **Ingredients**: the stripped $ggg$ coupling against `cubic_couplings` on
  every triple; the polarization sum against explicit vectors; the colour
  matrix ($\tfrac{16}{3}$, $-\tfrac23$, $12$, and
  $K_{ab}-K_{ba}=iK_s$ from $[T^a,T^b]=if^{abc}T^c$).
- **Abelian limits**: $q\bar q\to\gamma\gamma=N_cQ_q^4\cdot8e^4(u/t+t/u)$ and
  $q\bar q\to g\gamma$ with $\mathrm{Tr}(T^aT^a)=4$.

### Stage 5c — $gg\to gg$ ✅ (delivered)

The first amplitude with no fermion line, and the first diagram with a
four-point contact vertex. Scope: four external gluons of the calculator's
colour group. Any other four-boson process ($WW\to WW$,
$\gamma\gamma\to WW$, $VV\to hh$) still raises: those need scalar exchange,
`VVSS` and electroweak `VVVV` contact vertices and charged legs.

**The amplitude** (`pheno/boson.py`). A `BosonTerm` is a coefficient times
loose tensors (metrics and momenta) on the four external indices and on
propagator dummies; there is no Dirac string. `BosonAmplitude.squared`
contracts them with the new trace-free `contraction.contract_tensors`, uses
the same `VectorLeg` polarization sums as the line engine, and runs the same
Ward check (`line.check_ward_identities`, factored out of
`LineAmplitude.squared`).

**Three colour structures, not four diagrams.** The exchange diagrams have
colour $f^{a_ia_je}f^{ea_ka_l}$ for the pairings $(12|34)$, $(13|24)$,
$(14|23)$, and the contact vertex is a sum of three Lorentz structures with
exactly those colour tensors,

$$i g^2\big[f^{12e}f^{34e}(g_{14}g_{23}-g_{13}g_{24})
+f^{13e}f^{24e}(g_{14}g_{23}-g_{12}g_{34})
+f^{14e}f^{23e}(g_{13}g_{24}-g_{12}g_{34})\big]$$

(`adjoint_vvvv`, the coupling already exported to UFO; `vvvv_tensor` writes
one structure). So `enumerate_boson_diagrams` returns three entries, each an
exchange diagram plus its share of the contact vertex, and the colour matrix
is $3\times3$: $N^2(N^2-1)$ on the diagonal, half of it off the diagonal
with the signs of the Jacobi identity (`four_boson_colour_tensor`; $72$ and
$\pm36$ for SU(3)). The cubic vertices have their legs in the order
$(i,j,e)$ and $(e,k,l)$ for both $f$ and the Lorentz tensor, as in stage 5b.

**All four gluons on physical polarization sums**: the two incoming ones
take each other's momentum as reference, and so do the two outgoing ones.

**What this process fixes.** Each exchange diagram has two cubic vertices,
so $gg\to gg$ cannot see the sign of the cubic coupling ($\kappa$ of stage
4b); $q\bar q\to gg$ and $e^+e^-\to W^+W^-$ fix that. It fixes the
**contact** term relative to the exchange diagrams:
$(\kappa\,c_3)^2\cdot(-i)/q^2$ against $i g^2$.

**Making it fast enough.** Term by term the sum is $33^2\times3^4$
contractions, over ten minutes. Two exact reductions bring symbolic
$gg\to gg$ to about ten seconds. Each diagram is rewritten in a few
canonical monomials, with every vector on an external index taken modulo
momentum conservation and modulo the momenta that leg's polarization sum
annihilates ($k$ and the reference $n$ for a physical sum; computed from the
sum actually in use, so the Ward probes are reduced correctly). And the
contraction of two monomials through the polarization sums factorises over
the connected components of its index graph.

**Verified** (`tests/test_four_gluons.py`):

- **The parton table [ESW96, Table 7.1]**, exactly and symbolically:
  $gg\to gg=\tfrac92\big(3-\tfrac{tu}{s^2}-\tfrac{su}{t^2}-\tfrac{st}{u^2}\big)$,
  averaged over $(2\times8)^2$ initial states. It matched on the first run.
- **An explicit evaluator** written from the textbook rules [PS95, Chapter
  16]: transverse polarization vectors and $f^{abc}$, the amplitude summed
  per colour and helicity configuration; it first checks its own Ward
  identity. SU(3) and pure SU(2) Yang–Mills both agree with the engine (the
  SU(2) number also against $16N^2(N^2-1)$ times the same bracket, which is
  what the engine and the evaluator give, not a quoted result).
- **The contact vertex** against a direct fourth derivative of the quartic
  term of $-\tfrac14F^2$, colour component by colour component.
- **Controls**: with the contact term's sign flipped, or the contact term
  left out, the Ward check fails; with $-g$ on the gluons it fails, and
  forcing it gives a different number; with $\kappa$ flipped the result is
  unchanged.
- **Gauge independence**: pairing each incoming gluon with an outgoing one
  as reference gives the same $\sum|\mathcal M|^2$.
- The cross section inside an angular cut against the integrated closed
  form (the $t$- and $u$-channel poles need a cut).

## 17.6 Summary

| tier | channels | new machinery | effort | oracle |
|---|---|---|---|---|
| 1 ✅ | $e^+e^-\to\mu^+\mu^-$ (γ only), $f\bar f\to f\bar f$ via a scalar | `TwoToTwoKinematics`; `SpinorChain`/`Diagram`/`Amplitude` (**done**); amplitude-level propagators; `ExternalState` averaging | small–medium | Peskin QED closed form + explicit-matrix oracle |
| 2 ✅ | $e^+e^-\to\mu^+\mu^-$ (Z only); any single chiral-current diagram | ε (γ₅) algebra (`pheno.epsilon`): $\mathrm{Tr}[\gamma\gamma\gamma\gamma\gamma_5]$, ε·ε → Gram determinant, `forward_backward_asymmetry` | large | $d\sigma/d\cos\theta$ shape + $A_{FB}=\tfrac34A_eA_f$ vs [LEPEWWG06]; γ₅-carrying matrix oracle (massless + massive mediator) |
| 3 ✅ | full $e^+e^-\to f\bar f$ (γ/Z interference); Bhabha, Møller, $\nu_ee$ | multi-diagram double sum (same-pairing product of traces; exchanged-pairing single trace); s/t/u topology enumeration; permutation-parity fermion signs; `ScatteringCalculator` | medium–large | **MadGraph 2.7878 pb** (inside $\lvert\eta\rvert<2.5$); Bhabha/Møller closed forms; explicit-spinor oracle |
| 4 ✅ | 4a: $e^+e^-\to\gamma\gamma$, Compton, $e^+e^-\to ZZ/Z\gamma$; 4b: $e^+e^-\to Zh$, $W^+W^-$; 4c: $e^+e^-\to\phi^+\phi^-$ (`VSS`) | 4a: external bosons, fermion propagator, polarization sums with a Ward check, scalar-level contraction (**done**); 4b: `VVS`/`VVV` vertex tensors, boson exchange (**done**); 4c: `VSS` momentum tags (**done**) | large | 4a: Klein–Nishina and its crossing [PS95], explicit spinor + polarization oracle; 4b: **MadGraph 19.498 pb** (engine: 19.447), exact high-energy cancellation, Higgsstrahlung closed form [Djouadi08]; 4c: scalar-QED closed form [BFIK05], Ward identity |
| 5 ✅ | 5a: four-quark scattering, $e^+e^-\to q\bar q$; 5b: $q\bar q\to gg$, $qg\to qg$; 5c: $gg\to gg$ | 5a: colour tensors and the colour matrix from the group generators (**done**); 5b: external gluons, physical polarization sums, $ggg$ colour (**done**); 5c: pure-boson amplitude, 4-point contact vertex (**done**) | medium–large | 5a: ESW four-quark entries [ESW96], explicit colour + spinor sum; 5b: ESW gluon entries, explicit polarization evaluator; 5c: ESW $gg\to gg$, explicit polarization evaluator (no PDFs) |

The recommended order is the table order: Tier 1 gave the first native cross
section for the cost of an object layer that reuses the existing covariant
engine unchanged; Tier 2 was the genuinely new algebra 2→2 requires that 1→2
never did; Tier 3 is where the first MadGraph benchmark stops being
aspirational and is reproduced (the second, $e^+e^-\to W^+W^-$, by Tier 4's stage 4b); Tier 4 is the hardest single piece (derivative
couplings); Tier 5 generalizes to QCD: colour bookkeeping first (stage 5a),
then external gluons (stage 5b) and the four-gluon amplitude (stage 5c).

See {doc}`decays_roadmap` for the parallel decay-side roadmap and its
`n_momenta`/ε-drop precedent, `tests/test_scattering.py` for what Tiers 1–2
pin, `tests/test_interference.py` for Tier 3 and
`tests/test_line_scattering.py`, `tests/test_ww.py` and `tests/test_vss.py`
for Tier 4's stages 4a, 4b and 4c, and `tests/test_colour.py`, `tests/test_gluons.py`
and `tests/test_four_gluons.py` for Tier 5's stages 5a, 5b and 5c.

## 17.7 References

- **[PS95]** M. E. Peskin and D. V. Schroeder, *An Introduction to Quantum
  Field Theory*, Addison-Wesley (1995), ISBN 0-201-50397-2 — Chapter 5,
  "Elementary Processes of Quantum Electrodynamics," §5.1, the
  $e^+e^-\to\mu^+\mu^-$ closed form Tier 1's cross section reproduces, and
  Problem 5.2, Bhabha scattering, whose massless result Tier 3 reproduces
  (and, crossed $s\leftrightarrow u$, Møller scattering); §5.5, "Compton
  Scattering," the Klein–Nishina $|\mathcal M|^2$ with a massive electron and
  its crossing to $e^+e^-\to\gamma\gamma$, which Tier 4's stage 4a reproduces;
  Chapter 16, "Quantization of Non-Abelian Gauge Theories," the three- and
  four-gauge-boson Feynman rules the explicit $gg\to gg$ evaluator of Tier
  5's stage 5c is written from.
- **[BFIK05]** Yu. M. Bystritskiy, G. V. Fedotovich, F. V. Ignatov and
  E. A. Kuraev, "The cross sections of the muons and charged pions pairs
  production at electron-positron annihilation near the threshold,"
  Phys. Rev. D 72 (2005) 114019,
  [arXiv:hep-ph/0505236](https://arxiv.org/abs/hep-ph/0505236) — the Born
  cross section $\pi\alpha^2\beta^3/(3s)$ of a point-like charged scalar
  pair (their pion formula at $F_\pi=1$), which Tier 4's stage 4c reproduces.
- **[Djouadi08]** A. Djouadi, "The Anatomy of Electro-Weak Symmetry
  Breaking. I: The Higgs boson in the Standard Model," Phys. Rept. 457 (2008)
  1–216, [arXiv:hep-ph/0503172](https://arxiv.org/abs/hep-ph/0503172) —
  Eq. (4.18), the Higgsstrahlung cross section $\sigma(e^+e^-\to ZH)$ Tier 4's
  stage 4b reproduces.
- **[ESW96]** R. K. Ellis, W. J. Stirling and B. R. Webber, *QCD and Collider
  Physics*, Cambridge Monographs on Particle Physics, Nuclear Physics and
  Cosmology 8, Cambridge University Press (1996), ISBN 978-0-521-54589-1 —
  Table 7.1, the parton-level $2\to2$ $|\mathcal M|^2$ table; Tier 5's stage
  5a reproduces its four-quark entries, stage 5b its entries with two
  gluons and stage 5c $gg\to gg$.
- **[LEPEWWG06]** ALEPH, DELPHI, L3, OPAL, SLD Collaborations, LEP
  Electroweak Working Group, SLD Electroweak and Heavy Flavour Groups,
  "Precision Electroweak Measurements on the Z Resonance," Phys. Rept. 427
  (2006) 257–454, [arXiv:hep-ex/0509008](https://arxiv.org/abs/hep-ex/0509008),
  [doi:10.1016/j.physrep.2005.12.006](https://doi.org/10.1016/j.physrep.2005.12.006)
  — Eq. (1.55) the Born-level $e^+e^-\to f\bar f$ angular distribution,
  Eq. (1.56) the coupling asymmetry $A_f$, Eq. (1.66) the forward–backward
  asymmetry $A_{FB}^{0,f}=\tfrac34A_eA_f$ Tier 2's `forward_backward_asymmetry`
  reproduces.
- **[PDG]** Particle Data Group, *Review of Particle Physics* — "Kinematics"
  review (Mandelstam invariants, the two-body $t$-range, $d\sigma/dt$); see
  the current edition at [pdg.lbl.gov](https://pdg.lbl.gov/), cited without a
  year pin since PDG republishes annually and section numbers shift (same
  convention as {doc}`decays_roadmap` §16.2's use of the same review).

# 7. Diagonalization and the Physical Basis

## Physics statement

A weak-basis mass matrix built in {doc}`masses` is generically not diagonal
— its off-diagonal entries *are* the mixing between weak eigenstates.
Finding the physical (mass) basis means finding a rotation that
diagonalizes it, then rewriting every Lagrangian term in terms of the new,
physical fields. `feynlag` keeps these two facts — "here is a rotation" and
"apply it to the Lagrangian" — cleanly separated in the
{class}`~feynlag.vacuum.diagonalize.Rotation` object.

## `Rotation`: the generic substitution machinery

A `Rotation(old_fields, new_fields, matrix, kind)` (`vacuum/diagonalize.py:23`)
stores $\text{new} = R\cdot\text{old}$ and exposes:

- **`substitution()`**: the dict rewriting *old* fields in terms of *new*
  ones, `old_i → Σ_j (R⁻¹)_{ij} new_j` — i.e. the inverse relation, since
  the Lagrangian is written in weak-basis fields and needs those replaced.
  `kind` selects how the inverse is computed cheaply: `R^T` for
  `'orthogonal'`, `R^†` for `'unitary'`, a generic `R.inv()` for `'general'`
  — never a blind numerical inverse when the structure of $R$ already
  guarantees a cheap closed form.
- **`apply(M)`**: $R\,M\,R^{-1}$ (equivalently $RMR^T$ for symmetric $M$,
  orthogonal $R$) — the matrix conjugated *into* the new basis.
- **`check(M)`**: re-verifies that `apply(M)` really is diagonal, returning
  every nonzero off-diagonal residual rather than trusting the
  construction blindly — this is the code-level enforcement of
  CONVENTIONS.md's rule that a rotation angle must be verified against its
  defining condition, not assumed correct because it came from a
  textbook formula.

`Model.rotate(rotation)` registers a `Rotation` in application order;
`Model.physical_lagrangian` applies every registered rotation's
`substitution()` via `xreplace`, in that same order — which is exactly what
lets two rotations be **chained** (e.g. `examples/sm_u1x.py`'s Weinberg
rotation producing an intermediate $Z^0$ symbol, immediately consumed by a
second Z–Z′ rotation registered right after it).

## 7.1 Analytic 2×2 orthogonal diagonalization

### Derivation: the $\tan2\theta$ formula

For a real symmetric $2\times2$ matrix $M$ and
$R(\theta) = \begin{pmatrix}\cos\theta & \sin\theta\\-\sin\theta & \cos\theta\end{pmatrix}$,
demanding the off-diagonal entry of $R M R^T$ vanish is the defining
condition for the mixing angle. Expanding $(RMR^T)_{12}$ directly:

$$
(RMR^T)_{12} \;=\; \cos2\theta\,M_{12} \;-\; \tfrac12\sin2\theta\,(M_{11}-M_{22}),
$$

using $M_{12}=M_{21}$ (symmetry). Setting this to zero gives

$$
\tan2\theta \;=\; \frac{2M_{12}}{M_{11}-M_{22}}.
$$

`solve_mixing_angle_2x2(M)` (`vacuum/diagonalize.py:110`) returns both
$\theta = \tfrac12\arctan(\tan2\theta)$ **and** the `tan2theta` expression
itself — CONVENTIONS.md requires keeping the defining relation around for
verification, not just the solved angle, because a solved `atan` can hide a
branch/sign choice that only the original `tan 2θ` condition makes
unambiguous. `diagonalize_orthogonal_2x2` wraps this into a full `Rotation`
and attaches `.angle_relation` (`Eq(tan(2θ), tan2theta)`) for exactly that
downstream verification.

## 7.2 SVD (Dirac mass matrices)

### Derivation: why $M M^T$ and $M^T M$

A biunitary decomposition seeks $U_L, U_R$ orthogonal with $U_L M U_R^T =
D$ diagonal. If such $U_L, U_R$ exist, then

$$
U_L(MM^T)U_L^T = U_L M U_R^T\,U_R M^T U_L^T = D D^T = D^2, \qquad
U_R(M^TM)U_R^T = U_R M^T U_L^T\,U_L M U_R^T = D^TD = D^2
$$

(using $U_R U_R^T = U_L U_L^T = I$) — i.e. $U_L$ *must* diagonalize $MM^T$
and $U_R$ *must* diagonalize $M^TM$, both to the same $D^2$. This is a
**necessary** condition, not sufficient: diagonalizing $MM^T$ fixes $U_L$
only up to an independent orthogonal transformation within each degenerate
eigenspace (sign flips, or full rotations for repeated eigenvalues), and
`Mᵀ M`'s diagonalization fixes $U_R$ with the same ambiguity —
**independently**. `diagonalize_svd` (`vacuum/diagonalize.py:158`)
therefore computes $U_L$, $U_R$ from `_orthogonal_diagonalizer` separately
and then **re-aligns** them: it forms $D=U_LMU_R^T$, and for any row where
the intended diagonal entry vanishes, swaps rows of $U_R$ to find the
correct partner column, then flips the sign of any row of $U_R$ needed to
make $D$'s diagonal non-negative.

### Derivation: the analytic 2×2 route (`diagonalize_svd_2x2`)

For symbolic $2\times2$ inputs, `Matrix.diagonalize(normalize=True)`
(SymPy's generic path, used inside `_orthogonal_diagonalizer`) returns
unusable nested-radical/`Abs` expressions once $M$'s entries are symbolic
rather than numeric — so `diagonalize_svd_2x2` avoids ever solving $U_R$
independently from $M^TM$ at all. Instead:

1. Get $\theta_L$ the ordinary 2×2 way, from $M M^T$ (§7.1).
2. Form $N = R(\theta_L)\,M$. Since $R(\theta_L)(MM^T)R(\theta_L)^T$ is
   diagonal by construction of $\theta_L$, and that product equals
   $N N^T$, **$N N^T$ is diagonal** — i.e. $N$'s two rows are orthogonal
   vectors (in the ordinary Euclidean sense).
3. A single rotation that zeroes the second entry of row 0 of $N$,
   $\theta_R = \arctan\bigl(N_{01}/N_{00}\bigr)$, **also** zeroes the
   second entry of row 1: two orthogonal 2-vectors, both rotated by the
   *same* angle, either become simultaneously axis-aligned or stay
   simultaneously off-diagonal — there is no intermediate case in two
   dimensions. So the single condition from row 0 is already sufficient to
   diagonalize the whole matrix.

This makes $U_L M U_R^T$ **exactly** diagonal by an algebraic identity
(orthogonality of $N$'s rows), not by a sign convention that happens to
work for "roughly half of parameter space" — which is precisely the
failure mode of solving $\theta_L$ and $\theta_R$ **independently** from
$MM^T$ and $M^TM$ separately: each angle is only fixed up to its own
eigenvector-ordering/sign convention, and those two conventions can
disagree for a generic (non-symmetric) $M$, silently producing an
anti-diagonal (row-swapped) result. Deriving $\theta_R$ *from* $\theta_L$
and $M$ removes the second independent choice entirely.

## 7.3 Takagi factorization (Majorana mass matrices)

A (real) symmetric matrix $M=M^T$ always admits a real orthogonal
diagonalization $M = O^TDO$ (ordinary spectral theorem), but the diagonal
entries $D_{ii}$ can be **negative** — unphysical for a mass. Takagi
factorization instead seeks $M = UD_{\rm abs}U^T$ with $U$ unitary and
$D_{\rm abs}\ge0$, absorbing the sign into a phase. `diagonalize_takagi`
(`vacuum/diagonalize.py`) builds this directly: with `phases` diagonal,
`phases[i,i] = i` (imaginary unit) wherever $D_{ii}<0$ and $1$ otherwise,

$$
U \equiv O^T\cdot\text{phases}, \qquad
U\,D_{\rm abs}\,U^T = O^T\,\text{phases}\,D_{\rm abs}\,\text{phases}\,O
= O^T\bigl(\text{phases}^2 D_{\rm abs}\bigr)O = O^TDO = M,
$$

using $\text{phases}^2_{ii} = i^2 = -1$ exactly where $D_{ii}$ was
negative, so $\text{phases}^2 D_{\rm abs} = D$ recovers the original
(signed) diagonal. $U$ is genuinely complex (not merely orthogonal) only
when at least one eigenvalue was negative — the standard Majorana-phase
convention for absorbing an unphysical negative mass into the field
redefinition.

**Symbolic vs numeric.** The orthogonal $O$ can come from two places
(`method=`). `"symbolic"` uses SymPy's exact `Matrix.diagonalize`, which is
fine for a 2×2 or a block-diagonal matrix. It stalls on a generic $N>2$
matrix, though: a 3+2 seesaw with a generic $3\times2$ $m_D$ has
characteristic polynomial $\lambda\times$(quartic) and does not finish.
`"numeric"` uses mpmath's symmetric eigensolver (`eigsy`, which ships with
SymPy) at `dps` digits (default 50) and orders the columns by increasing
mass. The high precision is needed because a seesaw spectrum spans about 14
orders of magnitude, so double precision would lose the light states. An
eigenvalue below $10^{-\rm dps}$ of the largest counts as an exact zero mode
($D=0$, no phase, since the sign of a round-off-level eigenvalue is noise).
The default `"auto"` picks numeric for a numeric matrix larger than 2×2 and
symbolic otherwise, so every 2×2 and every symbolic call keeps its exact
result. The phase convention is the same on both routes, so the numeric $U$
feeds `MajoranaRotation` unchanged.

## 7.4 Fermion mass-basis rotations

Fermion rotations work through the **same** `Rotation`/`Model.rotate`
machinery as bosons, with one structural difference: `old_fields`/
`new_fields` are `Indexed` — e.g.
`Rotation([eL[i], EL[i]], [e1L[i], e2L[i]], rotation_2x2(θ))` — carrying an
explicit flavor-index symbol `i` that must be the **same** symbol
everywhere the Lagrangian was written, so `xreplace`'s key matching
actually fires.

Two rules make this work correctly:

- **Register two rotations per chirality** — one for the field leg, one for
  the bar leg, both with the *same* rotation matrix (a fermion field and
  its Dirac adjoint must rotate together for the bilinear to remain a
  consistent physical object).
- **`expand_bilinear` is mandatory after rotation**, and is applied
  automatically by `extract_fermion_vertices` and `fermion_mass_matrix`
  ({doc}`vertices`). After `xreplace` substitutes a rotated field, a
  `Bilinear` slot ends up holding an `Add` — e.g.
  `cosθ·ψ₁[i] + sinθ·ψ₂[i]` — but `Bilinear` is an opaque custom
  `Function`, so ordinary `sp.expand()` does **not** distribute it over
  that sum (the same "teach SymPy about a custom operator's linearity"
  problem `PartialMu`/`D_linear` solves for derivatives, {doc}`lagrangian`,
  applied here to two argument slots instead of one). Without
  `expand_bilinear`, vertex extraction would group terms by the *unsplit*
  composite `Add`-valued key instead of by each individual mass-eigenstate
  bilinear — silently wrong, not an error. `examples/sm_vll.py` and its
  tutorial notebook include a standalone demonstration of exactly this
  failure mode.

## 7.5 Guided physical basis (`mass_blocks`, `diagonalize`)

Sections 7.1–7.4 describe the diagonalizers; using them still means deciding
by hand which symbols form each block, building every mass matrix, knowing
which rotation each block needs, creating the new symbols and wiring the
conjugate maps. `Model.mass_blocks()` and `Model.diagonalize()` do that
bookkeeping and leave the user the one physics choice, the names:

```python
model.solve_tadpoles([m11sq, m22sq])
even, odd, charged = model.mass_blocks()
# scalar  Q=0  cp-even  [H10_r, H20_r]  goldstones: 0
# scalar  Q=0  cp-odd   [H10_i, H20_i]  goldstones: 1
# scalar  Q=1           [H1p, H2p]      goldstones: 1
model.diagonalize(even, names=["H", "h"], angle="alpha")
model.diagonalize(odd, names=["G0", "A0"], angle="beta")
model.diagonalize(charged, names=["Gp", "Hp"], conjugate_names=["Gm", "Hm"])
phys = model.physical     # bosons, cmap, conjugates, charges, goldstones, masses
```

### Algorithm: finding the blocks

Nothing is assumed about CP or about a vacuum alignment.

1. **Electric charge** of every component comes from the vacuum
   ({doc}`charges`: the diagonal-generator combination that annihilates every
   VEV), normalised so the first non-abelian Cartan coefficient is $+1$
   ($Q = T^3 + \dots$).
2. **Real neutral sector.** One mass matrix over *all* real fluctuations
   $\phi_r, \phi_i$, then the **connected components of its non-zero pattern**.
   When CP is conserved the CP-even and CP-odd fluctuations do not connect and
   two blocks come out; with a CP-violating potential they come out as one
   block tagged `mixed`. The CP-conserving case is a limit of the general
   one, not an input.
3. **Complex sector.** Components without a VEV expansion, grouped by charge,
   with $M^2_{ij} = \partial^2V/\partial\bar\phi_i\partial\phi_j$ and the same
   connected-component split.
4. **Fermions.** The Dirac mass terms $\bar\psi_L P_R \psi_R$ at the vacuum
   define a bipartite graph between left and right components; its connected
   components are the blocks (colour copies separate on their own).

Two situations are refused with a specific message instead of being guessed:

- a neutral complex component with a $\phi\phi$ mass term (an inert doublet
  with $\lambda_5$): its real and imaginary parts are split, so it must be
  declared with a zero VEV, `expand_vev({comp: 0})`. A zero VEV has no tadpole
  variable; its stationarity condition (the term linear in the fluctuation)
  is reported by `Model.tadpoles()` only if it does not vanish identically;
- components of opposite charge that mix (doublets with $Y=\pm\tfrac12$),
  which need a basis of a field together with a conjugate field.

### Derivation: Goldstone directions

Under an infinitesimal gauge transformation
$\delta\phi = i\alpha^a T^a \phi$. The potential is invariant, so at a
stationary point $\langle\phi\rangle$ the mass matrix annihilates every
direction $i\,T^a\langle\phi\rangle$ — the Goldstone theorem [GSW62]. Each
broken generator therefore *gives* a zero mode, with no diagonalization:

$$
G^a \;\propto\; i\,T^a\langle\phi\rangle, \qquad M^2\,G^a = 0 .
$$

In fluctuation coordinates, with $\phi = (v + \phi_r + i\phi_i)/\sqrt2$,
$\delta\phi_r = \sqrt2\,\mathrm{Re}\,\delta\phi$ and
$\delta\phi_i = \sqrt2\,\mathrm{Im}\,\delta\phi$
(`goldstone_directions`). For two doublets with VEVs $v_1, v_2$ the
directions are $(v_1, v_2)/\sqrt{v_1^2+v_2^2}$ in both the CP-odd and the
charged block, so the rotation is $R(\beta)$ with $\tan\beta = v_2/v_1$ — the
familiar 2HDM result [Branco12] *comes out* of the group theory.

`mass_blocks` checks $M^2 G = 0$ for every Goldstone vector it attaches to a
block. A failure means the vacuum is not a stationary point as given, almost
always because the tadpoles were not solved, and it raises saying so.

### Algorithm: choosing the rotation

| block | route | order of `names` |
|---|---|---|
| $1\times1$ | no rotation (optional rename) | the state |
| $n\times n$ with $n-1$ Goldstones | fixed exactly by the Goldstone vectors: rows $G^\dagger$, then the orthogonal complement | Goldstones, then the physical state |
| real $2\times2$, no Goldstone | $\tan2\theta$ (7.1) | rows of $R(\theta)$ |
| fermion $2\times2$, one flavour | analytic biunitary (7.2), field- and bar-side rotations for both chiralities | as `diagonalize_svd_2x2` |
| any block with `values=` | numeric, at 50 digits | ascending or descending mass (`order=`) |
| symbolic, larger, fewer Goldstones | `NotImplementedError` listing the options | |

Every rotation is checked against the mass matrix before it is registered:
symbolically for the Goldstone-fixed route, at random numeric points for the
$\tan2\theta$ route (whose explicit $\tfrac12\arctan$ angle makes `simplify`
take minutes on an identity that holds by construction). A user-supplied
`rotation=` goes through the same check.

Flavoured fermion blocks ($n_f>1$) and blocks with a Majorana mass term are
*reported* with their matrix and `diagonalizable=False`, and the note names
the existing route (`diagonalize_svd(method="numeric")`, or 7.3).

### Which state is which: the $\tan2\theta$ branch

$\tan2\theta$ fixes $\theta$ only modulo $\pi/2$, and a shift by $\pi/2$
swaps the two eigenstates. The principal value $\theta=\tfrac12\arctan(\dots)$
puts the heavier state first only where $M_{11}>M_{22}$; which state is
heavier cannot be decided symbolically. So:

- without `values=`, the names follow the rows of $R(\theta)$ on the principal
  branch, exactly as `diagonalize_orthogonal_2x2` does, and carry no mass
  claim;
- with `values=` (a benchmark point), the rotation stays symbolic in
  $\theta$ and the numbers only pick the branch
  ($\theta \to \theta \pm \pi/2$ with the sign of $M_{12}$ when needed) so the
  first name is the heavier state (`order="light_first"` for the opposite).
  `Diagonalization.branch_valid_for` records the signs of
  $(M_{11}-M_{22},\,M_{12})$ at that point: the labelling holds in the region
  of parameter space where those signs do, and a point across
  $M_{11}=M_{22}$ needs its own call.

Floats in `values=` are read as exact decimals, so a numeric rotation is
computed at full working precision and a Goldstone comes out as an exact zero.

### `model.physical`

Each `diagonalize` call, and the SM helpers `weinberg_rotation`,
`charged_current_rotation` and `to_physical_basis`, record their states in one
registry: `bosons` (with conjugate partners), `cmap`, `conjugates`,
`goldstones`, `masses`, and `charges` (the vacuum-derived charge propagated
through the registered rotations). These are the arguments `Model.vertices`,
`Model.validate(charges=…, fields=…, conjugate_map=…, conjugates=…)` and the
UFO export ask for.

Gauge bosons are not rotated by `diagonalize`: use `weinberg_rotation` and
`charged_current_rotation` (`feynlag.models`).

## 7.6 References

- **[GSW62]** J. Goldstone, A. Salam and S. Weinberg, *"Broken Symmetries"*,
  Phys. Rev. **127**, 965 (1962).
  [doi:10.1103/PhysRev.127.965](https://doi.org/10.1103/PhysRev.127.965) — the
  Goldstone theorem: a massless mode along each broken-generator direction.
- **[Branco12]** G. C. Branco, P. M. Ferreira, L. Lavoura, M. N. Rebelo,
  M. Sher and J. P. Silva, *"Theory and phenomenology of two-Higgs-doublet
  models"*, Phys. Rept. **516** (2012) 1–102,
  [arXiv:1106.0034](https://arxiv.org/abs/1106.0034),
  [doi:10.1016/j.physrep.2012.02.002](https://doi.org/10.1016/j.physrep.2012.02.002)
  — the 2HDM $\beta$ rotation of the CP-odd and charged sectors that the
  Goldstone-fixed route reproduces.

## Design gotchas

- **Never use generic `Matrix.diagonalize()`/`eigenvects()` on symbolic
  matrices larger than 2×2.** SymPy's closed-form solver produces nested
  radicals that don't simplify — use a `rotation_2x2`-style ansatz per
  2×2 block, or fall back to numeric diagonalization at export time.
- **Verify every rotation against its `tan2θ` (or equivalent) defining
  condition**, not merely `sin²+cos²=1` — `Rotation.check()` exists
  precisely so this isn't left to manual inspection.

## Validation

- `tests/test_physical_basis.py` — the guided basis (7.5): SM blocks, charges
  and Goldstone directions; the 2HDM blocks found without hints, with
  `tan β = v2/v1`, `m_A²` and `m_{H±}²` from the Goldstone-fixed route and the
  CP-even rotation identical to the hand-built one; branch selection on both
  sides of `M11 = M22`; a CP-violating potential merging the neutral block;
  the refusal guards; the vector-like-lepton rotations equal to
  `diagonalize_svd_2x2`'s.

- `tests/test_scalar_pipeline_thdm.py::test_alpha_rotation_diagonalizes_cp_even`,
  `::test_goldstone_beta_rotation` — the analytic 2×2 route, §7.1.
- `tests/test_fermion_sector.py::TestDiagonalization`,
  `::TestRotatedBilinearExtraction` — the biunitary SVD route and the
  `expand_bilinear` requirement, §7.2/§7.4.
- `tests/test_vll.py::test_mixing_angles`, `::test_svd_diagonalizes` — the
  full VLL 2×2 biunitary case end to end, both symbolically and at random
  numeric points.

## Minimal snippet

```python
from feynlag import diagonalize_svd_2x2, rotation_2x2

rot_L, rot_R = diagonalize_svd_2x2(
    M, [eL[i], EL[i]], [eLbar[i], ELbar[i]],
    [e1L[i], e2L[i]], [e1Lbar[i], e2Lbar[i]],
    angle_left=sp.Symbol("theta_L"), angle_right=sp.Symbol("theta_R"))
model.rotate(rot_L)
model.rotate(rot_R)
```

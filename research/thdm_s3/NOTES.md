# 3HDM-S₃ — research log

The S₃-symmetric three-Higgs-doublet model: `(H1, H2)` an S₃ doublet, `HS` the
singlet. The symbolic build lives in `examples/thdm_s3.py` (potential only,
weak-basis CP-even mass matrix) and `examples/THDM_S3_Tutorial.ipynb` (the
pedagogical walk-through). This directory is where the open questions go.

## Layout

| file | what it is |
|---|---|
| `model.py` | the reusable build — algebra only. `build_model()` (with `soft=True` for soft S₃ breaking), the three mass matrices, the geometric rotation, the lambdified spectrum, the quartic potential + its numerical boundedness scan. Lifts the tutorial's notebook-local §6–§8 into importable form. |
| `scan.py` | the parameter sweep at scan size — the cuts of notebook 01 in a chunked loop, sole writer of `results/viable_points.json`. |
| `constraints.py` | numpy-vectorized boundedness-from-below + tree-unitarity conditions, quoted from [DasDey14] — **plus** the corrected neutral-direction condition derived in finding 4 below. |
| `fermions.py` | the S₃ fermion sectors — basis map to [LFVHD], the five Yukawa structures, mass-matrix extraction, the two-stage diagonalization, the $G_k$/$Q_i$ LFV couplings. |
| `paper_lfvhd/LFVHD_3HDMS3.tex` | the draft this work checks against; **patched** (findings 6–8). |
| `01_scalar_parameter_space.ipynb` | scalar parameter space under theory constraints. **Done.** |
| `02_s3_fermion_sector.ipynb` | lepton + quark Yukawa sectors, the draft's derivations, the CKM obstruction, soft breaking. **Done.** |
| `decays.py` | the physical basis registered on the `Model` at general δ, the electroweak kinetic sector, the lepton mass basis, and the benchmark-point loaders. |
| `derive.py` | the by-hand derivation vocabulary — `show`, `coefficients_of`, `rows`, `check_homogeneous/limit/invariant`, `hand_slice`, `falsify`, and the `Ledger` that emits the summary table and the printable LaTeX appendix. Renders sympy as typeset maths under a notebook kernel, aligned text outside one. |
| `derivations_01.tex` | printable appendix emitted by notebook 01 §5.6 — the algebra typeset for checking on paper (`pdflatex derivations_01.tex`). |
| `03_scalar_decays.ipynb` | physical basis, gauge couplings, VSS + loop γγ, and the LFV rates. **Done.** |
| `04_bfb_conditions.ipynb` | the [BotoRomaoSilva22] BFB method read and executed: their $V_N/V_{CB}/V_G$ split, copositivity, the lower-bound strategy — then applied to our $S_3$ potential. **Done.** |
| `derivations_04.tex` | printable appendix emitted by notebook 04 §8. |
| `scan_soft.py` | the parameter sweep in the **soft-broken** vacuum — notebook 01's cuts over λ, (θ, φ) and the three free soft quadratics; sole writer of `results/viable_points_soft.json`. |
| `05_soft_scalar_scan.ipynb` | the soft-broken vacuum: released tadpoles, the φ fundamental domain, the general-vacuum spectrum, and the scan against notebook 01. **Done.** |
| `derivations_05.tex` | printable appendix emitted by notebook 05 §7. |
| `lfv.py` | charged-lepton LFV couplings at any vacuum: per-doublet $G_k$ read off the Lagrangian (`lepton_g_function`), the draft's analytic inversion at the alignment, and the exact arrowhead-cubic enumeration of every lepton-fit branch off it (`lepton_branches`, `branch_kind`). Shared by notebooks 03 and 06. |
| `soft_decays.py` | per-point numeric decay machinery for the soft-broken vacuum: rotations rebuilt from the stored inputs, the $hH^+H^-$ cubic tensor, VSS overlaps and widths, $R_{\gamma\gamma}$, LFV over every lepton branch, the S₃-image map. |
| `06_soft_decays.ipynb` | decays and LFV on the 2021 soft points, validated against the extractor, the exact-S₃ limit and the S₃ image. **Done.** |
| `derivations_06.tex` | printable appendix emitted by notebook 06. |
| `report_scalar_sector.tex` | advisor/collaborator report on notebook 01 (the corrected BFB condition, the scan with the hVV coupling cut). |
| `report_fermion_sector.tex` | report on notebook 02: the three draft defects, $V_{us}=0$ with exact S₃, the soft-breaking CKM fit. |
| `report_scalar_decays.tex` | report on notebook 03: $h_0$ gauge-phobic for any δ, δ = −ψ, LFV rates against [CMS21]. |
| `report_soft_vacuum.tex` | report on notebook 05: the soft-broken scan and its decoupling-like spectrum. |
| `report_soft_decays.tex` | report on notebook 06: the arrowhead lepton fit, SM-like LFV from the non-vacuum admixture, $R_{\gamma\gamma}$ and VSS in the soft vacuum. |
| `build_reports.sh` | rebuilds the five `report_*.pdf` (figures first, then pdflatex ×2); the PDFs are committed so they render on GitHub. |
| `README.md` | the folder's GitHub landing page: the report index. |
| `extract_figures.py` | copies the stored notebook figures to `figures/*.png` for the reports; each entry is guarded by a marker string in its source cell. |
| `results/viable_points.json` | every point surviving the scan's cuts (1964 from 60M samples), each tagged with its CP-even states' $hVV$ coupling² (`hvv_squared`) and its δ; written by `scan.py`. |
| `results/quark_soft_fit.json` | soft-breaking quark benchmark (masses + Cabibbo angle). |
| `results/viable_points_soft.json` | 2021 soft-broken points from 110M samples, each with its (θ, φ), both draft-basis ratios, all four soft terms, the mass-ordered spectrum, hVV couplings² and the 3×3 CP-even mixing; written by `scan_soft.py`. |
| `results/decay_benchmarks.json` | per-point δ and VV couplings, plus the LFV branching-ratio distribution per (state, channel, $\mu_3$) and a 200-entry sample. |
| `results/soft_decay_benchmarks.json` | per soft point: SM-like index and $\kappa_V$, $R_{\gamma\gamma}$, $hH^+H^-$ couplings, VSS widths, the least-constrained lepton configuration; plus LFV summaries by branch kind and misalignment. Written by notebook 06. |

Notebooks import `model.py` from their own directory; from the repo root use
`sys.path.insert(0, "research/thdm_s3")`.

**Kernel note.** These notebooks are executed with the repo's `lagrangian`
kernel. They used to run under the plain `python3` kernel because the `lagrangian`
conda env had no `numpy`/`matplotlib` (the scan and the figures need both); that
env now carries the dev extras — `numpy`, `matplotlib` and `scipy` are all present
— so the exception no longer applies and all four notebooks declare `lagrangian`.

`language_info.version` records the interpreter that produced the stored outputs,
so it is set by re-executing, never by hand. VS Code re-stamps it whenever it
attaches a kernel, which is why a notebook can go dirty with nothing but that one
line changed.

**The switch moved `results/quark_soft_fit.json`.** The soft-breaking quark fit
(notebook 02 §11) is seeded — `np.random.default_rng(7)` over 60 restarts — but
that does not make it environment-stable: the `lagrangian` env carries numpy 2.5.1
/ scipy 1.18.0 against base python3's 2.2.4 / 1.16.1, and the best-of-60
`least_squares` lands on a different solution. It is not a disagreement about
physics. The fit has **9 free parameters against 7 targets** (six quark masses plus
$V_{us}$), so its exact solutions form a two-parameter family; both the old and the
new point have `cost` ~1e-19 and reproduce all six masses and $V_{us}=0.2243$.
`r` is therefore *not a prediction* — it is wherever the optimiser stopped ($1.674$
before, $1.446$ now, against the exact-S₃ $\sqrt3=1.732$), and §11's claim is only
that soft breaking frees $r$ from $\sqrt3$ and switches the Cabibbo angle on, which
holds at either point. Quote `r` or `V_abs` from this file as an existence proof,
never as a determination. (This described the original 7-target fit — six masses
plus $V_{us}$. Finding 9 below extended it to all three CKM magnitudes, nine
targets against nine parameters, which is exactly determined rather than a free
family; the same "quote `r` as an existence proof" rule applies there too, for a
different reason — see finding 9.)

Re-execute with the **miniconda** nbconvert, not the one first on `PATH`
(`~/.local/bin/jupyter-nbconvert` runs under a different Python and dies on a
missing `packaging`/`dateutil`):

```bash
/home/moises/miniconda3/bin/jupyter-nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=lagrangian --ExecutePreprocessor.timeout=1800 \
  research/thdm_s3/01_scalar_parameter_space.ipynb
```

**Memory note.** The box has ~7 GB and often <2 GB free; a 2 M-point scan that
materializes full-length temporaries gets OOM-killed (silently — nbconvert dies
with no message and leaves the notebook untouched). `constraints.py` therefore
applies every heavy array function in blocks (`_chunked`, `CHUNK = 250_000`),
which holds peak RSS to ~285 MB for all masks over 2 M points.

## How these notebooks are written: derive, then check

The notebooks were originally written **state → assert**: the markdown announced a
result, the code confirmed it, and the object you would have written on paper was
computed and discarded. That establishes correctness and teaches nothing — a
passing `assert` only says the machine agrees with a result you were handed.

Headline derivations now use four moves (`derive.py`), first applied in §5.4 of
notebook 01:

1. **set up** — build the object and `show` it *raw*;
2. **collect** — `coefficients_of` so the structure **emerges**; never type the
   grouped form in from the answer and assert equality to it;
3. **recognise** — name what appeared and why it had to (the symmetry, the
   covariant);
4. **check** — from *physics*, not from the same algebra: `check_homogeneous`
   (scaling), `check_limit` (does a known theory come back?), `check_invariant`
   (a symmetry it must respect), `hand_slice` (small enough to finish on paper).

Plus `falsify` — a check that cannot fail is not a check, so break the input and
confirm the check fires. The *predict* step is a **markdown blockquote** placed
immediately before the cell that settles the question, asking the reader to commit
to an expectation first; it was briefly a `predict()` function, but prose typesets
and a print statement does not.

Each step registers with a `Ledger`, which prints *claim / derived here vs
imported vs assumed / independent checks / what would falsify it* and emits
`derivations_0N.tex` for checking the algebra away from the screen. Every sympy
object `derive.py` shows is rendered as typeset maths under a notebook kernel
(`show`, `rows`, `coefficients_of`, the residual inside a failed check, and each
`Ledger` step's expression) and degrades to aligned text outside one — the
coefficient table is the thing you are meant to *stare at*, so it must not arrive
as `(r_1**2 + r_2**2)**2`.

### Where the idiom is applied

| notebook | § | what it replaced |
|---|---|---|
| 01 | 4.2 | trying `e = ±λ₄` and keeping the match → **solving** an over-determined 20-equation system for $(d,e,f,g,h)$; the sign is an output |
| 01 | 5.4 | the asserted grouped BFB form → `coefficients_of` |
| 02 | 4 / 4.1 | three hand-typed μ dictionaries → read off `M_ℓ` |
| 02 | 5.1 | comparing to two typed closed forms → building $(\sigma_1-\sigma_2)^2$ from $\mathrm{tr}(A^{\mathsf T}A)$ and $\det A$ |
| 02 | 10 | a hand-written eigenvector → `eigenvects()`, and decoupling recast as orthogonality to the vacuum direction |
| 03 | 2 | `sm_WW` typed in → read off the $\delta\to0$ limit, with $h_0$'s zero shown to be $R_S^{\mathsf T}\hat n=(\cos\delta,0,\sin\delta)$ |
| 03 | 7 | `lepton_mus()` never checked → its output verified against the physical masses as singular values of $M_\ell$ |

**§5.4 of notebook 01 is the reference implementation.** The payoff there was concrete: the
`coefficients_of` table shows at a glance that λ₁,λ₃ share a coefficient (so only
λ₁+λ₃ appears), that λ₅,λ₆,2λ₇ do likewise, and that **λ₂ multiplies zero** — it
needs a relative phase, so the real neutral slice cannot see it. None of that was
visible when the grouped form was asserted. The λ₄→0 limit check then shows *why*
[DasDey14] Eq. (4g) looked right: with λ₄ off the potential is a genuine quadratic
form in (x,y) and Eq. (4g) really is the copositivity cross-term condition. The
half-integer power λ₄ introduces is exactly what invalidates it.

## Findings

### 1. The tutorial benchmark sits at the wrong electroweak scale

`examples/THDM_S3_Tutorial.ipynb` picks $(v_1,v_2,v_S)=(200,115,80)$ GeV
precisely so that $\sqrt{\sum v_i^2}\approx246$ GeV ([GomezBock21] Eq. 8,
stated in its md cell 5). It then imposes the S₃ alignment as the substitution
`align = {v1: v2/sqrt(3)}`, which **replaces** $v_1=200$ by $115/\sqrt3=66.4$
and never re-checks the scale. The aligned vacuum is $v=155.03$ GeV.

Consequence: at fixed λ and θ every mass² is homogeneous of degree 2 in the
VEVs, so the §9 spectrum is uniformly low by $(246/155.03)^2 = 2.518$ in mass²,
i.e. **every mass in GeV is low by a factor 1.587**. The *qualitative*
conclusion there ("5 of 7 states tachyonic") is unaffected.

Fix: impose both conditions at once. They leave one free parameter, the same
angle θ the geometric rotation uses ([GomezBock21] Eq. 24): $v_{12}=v\sin θ$,
$v_S=v\cos θ$, and the alignment then fixes $v_1=v_{12}/2$,
$v_2=\sqrt3\,v_{12}/2$ exactly. Implemented as `model.aligned_vevs`, pinned by
`tests/test_thdm_s3.py::test_aligned_vacuum_can_meet_the_electroweak_scale`.

**Not yet propagated to the tutorial.** Correcting `examples/THDM_S3_Tutorial.ipynb`
§9 is a separate, small follow-up (re-execute + commit).

### 2. feynlag's masses reproduce the published closed forms — and fix the dictionary

The λ basis is *not* trivially the literature's. The λ₄ invariant transcribed
literally from [DasDey14] into feynlag's component labels **fails**
`check_discrete_invariance` against S₃ — the doublet irrep bases differ.

The dictionary is established in two steps, neither of them assumed.

**Seven entries by basis-independence.** Under a general $SO(2)$ rotation of the
S₃ doublet, the singlet contraction $x_S=(x_{S1},x_{S2})$ rotates by α while $d_2$
rotates by 2α. Testing all eight quartic structures symbolically: **seven are
invariant** and only $x_S\cdot d_2$ (the λ₄ structure) can change. Invariant
structures are the same object in either basis, so their coefficients transfer by
term matching: $a=2\lambda_8$, $b=\lambda_5$, $c=2\lambda_1$, $d=2\lambda_2$,
$f=\lambda_6$, $g=2\lambda_3$, $h=2\lambda_7$.

**The last entry — λ₄'s sign — from physics.** The only S₃-preserving
transformation that changes $x_S\cdot d_2$ is $\alpha=\pi$, i.e. $H_{1,2}\to-H_{1,2}$,
which flips it and leaves the other seven alone. Which sign applies is settled by
requiring feynlag's independently-computed pseudoscalar and charged masses to equal
[GomezBock21]'s closed forms Eqs. (30)–(33). All four match **exactly**, and only
for $e=-\lambda_4$.

Composing with the [GomezBock21]↔[DasDey14] correspondence gives
$\lambda_k^{\rm DD}=\lambda_k^{\rm FL}$ for all $k\neq4$ and
$\lambda_4^{\rm DD}=-\lambda_4^{\rm FL}$. Since the flip is a *field
redefinition*, no physical condition may depend on it — and indeed λ₄ enters the
theory conditions only as $|\lambda_4|$ or $\lambda_4^2$, so they transfer
unchanged. That is a consequence, not a coincidence.

This doubles as an end-to-end validation of feynlag's 3HDM-S₃ chain — potential,
tadpoles, all three mass matrices and the rotation — against published closed
forms. Pinned by `tests/test_thdm_s3.py::test_masses_match_gomezbock_closed_forms`.

### 3. The tutorial's benchmark λ's are not an admissible theory

At $\lambda_k=0.05k$ the potential is **not bounded from below** ([DasDey14]
Eq. 4 fails). "5 of 7 tachyonic" was the symptom; the point was never a valid
theory. Tree unitarity is comfortably satisfied there (largest eigenvalue 3.63
against the 16π ≈ 50.3 bound).

### 4. The published boundedness conditions are not sufficient

**The most substantive finding.** Boundedness from below is directly testable —
$V_4$ is homogeneous of degree 4, so the potential is bounded iff $V_4\ge0$ on the
unit sphere in field space. Testing [DasDey14] Eq. (4) against feynlag's own $V_4$
(12 real dof, extracted by `model.quartic_potential`) produces **explicit
counterexamples**: λ points Eq. (4) accepts that have a real *neutral* direction
along which $V_4<0$. Over the 2 M-point scan the corrected condition removes
**9.35%** of everything Eq. (4) admits.

The cause is exact. On the real neutral slice $(H_1^0,H_2^0,H_S^0)=(r_1,r_2,r_S)$,

$$V_4=(\lambda_1+\lambda_3)x^2+(\lambda_5+\lambda_6+2\lambda_7)xy+\lambda_8y^2
+2\lambda_4 r_S r_1(r_1^2-3r_2^2),\quad x=r_1^2+r_2^2,\ y=r_S^2,$$

and the last term is the S₃ cubic covariant $2\lambda_4\sqrt y\,x^{3/2}\cos3\psi$,
worst case $-2|\lambda_4|x^{3/2}\sqrt y$. With $x=1$, $t=\sqrt y$ the **exact**
condition on this slice is

$$f(t)=\lambda_8t^4+(\lambda_5+\lambda_6+2\lambda_7)t^2-2|\lambda_4|t+(\lambda_1+\lambda_3)\ \ge 0
\quad\text{for all } t\ge0,$$

and **[DasDey14] Eq. (4g) is precisely $f(1)>0$** — one point of that curve.
Necessary, not sufficient. Implemented as `constraints.neutral_real_bfb_min`
(closed-form minimisation via batched companion matrices) and
`strict_bfb_mask`; validated against brute-force direction sampling on the same
slice (0 disagreements). Reproduced in `01_scalar_parameter_space.ipynb` §5.3–§5.5.

Still only *necessary* overall — it covers the real neutral slice, not
complex-neutral or charged directions. So the notebook also runs
`model.numeric_bfb_min` over the full 12-dimensional space as an empirical
backstop on the quoted benchmark points; it removes a further 3 of 322 (§8),
confirming the residual gap is real but small once the other cuts are applied.

**Resolved.** The [DasDey14] Erratum (Phys. Rev. D 91, 039905, 2015 — never posted
to arXiv, but freely readable at APS) corrects **Eqs. (9a)–(9c)**, the tadpole
conditions, and nothing else; it states that the changes "do not affect any of our
conclusions". Eq. (4) and Eqs. (36)–(37) are untouched, so this is an insufficiency
of the *final published* conditions, not of a superseded pre-erratum form. Notebook 01
§2.2 checks the corrected Eq. (9) against our own tadpole derivation: setting
(9a) = (9b) leaves 3λ₄v₃(v₁²−3v₂²)/v₂ and forces v₁ = √3 v₂, which is our alignment
under the documented v₁↔v₂ swap. The one equation Das & Dey had to correct is the one
equation we never imported.

### 5. Boundedness, not unitarity, is the binding constraint

Over $\lambda\in U(-2,2)^8$: boundedness keeps ~5.7% of points (and the corrected
condition removes ~10% of those), unitarity then removes **none**. Tree unitarity
only bites at $|\lambda|\gtrsim$ several. Within boundedness, Eq. (4g) and
Eq. (4d)/(4f) do most of the rejecting.

Viable spectra cluster sub-TeV, consistent with [DasDey14]'s own conclusion
that "many new scalars must be lurking below 1 TeV". Current cut-flow numbers
live in `results/viable_points.json` (`cut_flow`), regenerated by the notebook.

### 6. The draft's S₃ basis differs from feynlag's — but the couplings map 1:1

`LFVHD_3HDMS3.tex` uses $a=R(-2\pi/3)$, $b=$ reflection about 30°; feynlag uses
$\rho(a)=R(2\pi/3)$, $\rho(b)=\mathrm{diag}(1,-1)$. They are conjugate by a unique
O(2) **reflection at $\pi/6$** (`fermions.tex_basis_map`, verified not assumed);
$\det=-1$, which is why the draft's order-3 generator is feynlag's inverse. The
visible symptom is the alignment: $v_1=\sqrt3v_2$ in the draft, $v_2=\sqrt3v_1$
in feynlag.

Four of the five Yukawa structures are *dot products* of two S₃ doublets and so
are invariant under any orthogonal basis change; only the triple-doublet
structure ($Y_2$) is basis-sensitive — the fermionic analogue of $\lambda_4$
(finding 2). Under the reflection the draft's $Y_2$ maps onto feynlag's with
coefficient $+1$, so **the coupling dictionary is the identity** (unlike the
scalar case, where $\lambda_4$ flipped sign).

### 7. The draft's charged-lepton Yukawa sector is invariant and complete

All five terms pass gauge and S₃ invariance in feynlag. `suggest_yukawa`,
enumerating independently, returns exactly five structures — matching
$Y_1^\ell\ldots Y_5^\ell$. Character theory agrees: the trivial rep appears once
in $\mathbf2^{\otimes3}$, and over the eight $(\bar L,H,e_R)$ irrep assignments
exactly five invariants exist.

**Gap:** the draft declares $\nu_{1R},\nu_{2R}$ (a $\mathbf2$) and $\nu_{SR}$ (a
$\mathbf1$) and never uses them. Those fields permit further S₃- and
gauge-invariant operators (Dirac structures on $\tilde H$, plus bare Majorana
masses, the $\nu_R$ being gauge singlets). Enumerated in
`02_s3_fermion_sector.ipynb` §8; not built.

### 8. Three defects in the draft — found, corrected, and patched into the `.tex`

1. **The $\mu\leftrightarrow Y$ dictionary is a uniform factor $\sqrt2$ too
   large.** With the draft's own $\langle H_i^0\rangle=v_i/\sqrt2$ and its
   explicit prefactors, the Lagrangian gives $\mu_1^\ell=Y_1^\ell v_3/2$,
   $\mu_2^\ell=Y_2^\ell v_2/2$, $\mu_3^\ell=Y_3^\ell v_3/\sqrt2$,
   $\mu_4^\ell=Y_4^\ell v_2/2$, $\mu_5^\ell=Y_5^\ell v_2/2$. The printed values
   are those of the convention $\langle H_i^0\rangle=v_i$, which contradicts
   $v=246$ GeV in $m_W$. **Harmless downstream** — the $\mu$'s are eliminated
   for physical masses, so $M_\ell$, $G_k$ and $Q_i$ are unaffected; it matters
   only if one wants the Yukawa couplings themselves.
2. **Eq. (muil_equations) line 3 mixed eigenvalues and singular values.** The
   singular-value relation is
   $(m_\tau-m_\mu)^2=(2\mu_3-m_\mu-m_\tau)^2+4(\mu_4+\mu_5)^2$; the printed
   $+16\mu_4\mu_5$ is the *eigenvalue* gap $(\mathrm{tr}A)^2-4\det A$ (patched,
   along with Eq. muil_sols line 3 and the $\mu_4\mu_5\ge0$ remark). **Caveat
   (audit, 2026-09-15):** the patched derivation of $\mu_5=\mu_4$ still uses
   $m_\mu+m_\tau=\mathrm{tr}A$, i.e. eigenvalue matching. Together with the
   singular-value relations that is *equivalent* to $\mu_4=\mu_5$
   ($\mathrm{tr}A^{\mathsf T}A-[(\mathrm{tr}A)^2-2\det A]=4(\mu_4-\mu_5)^2$), so
   $\mu_5=\mu_4$ is the ansatz $O_L=O_R$, not a consequence of the masses. The
   draft should say so; it has not been changed there yet (notebook 02 §5.1).
3. **A coefficient slip:** $\mathrm{tr}(N)=4\mu_4^2+m_\mu^2+m_\tau^2$, not
   $2\mu_4^2+\ldots$. The conclusion $\mu_4=0$ is unaffected.

Everything else checks out: $R_S=R_AR_H$ identically, the $O_{12}$
block-diagonalization with $m_e=\mu_1-2\mu_2$, $\tan2\theta_\ell$ and the
$p_{1,2}$ closed forms, and $Q_1(A)=\mathrm{diag}(m)/v$. The last is an *exact
identity* that holds for any rotation (the first column of $R_A$ is the vacuum
direction), so it does not test $O$; the $G_k$ split is checked separately, with
$v_1$ kept explicit so that $G_1\neq0$.

The basis map sends feynlag's aligned vacuum $(v_1,\sqrt3v_1)$ to
$(\sqrt3v_1,-v_1)$ in the draft basis, i.e. $r=v_1/v_2=-\sqrt3$. That sign is
harmless: $M(-r)=P\,M(r)\,P$ with $P=\mathrm{diag}(-1,1,1)$, a sign flip of the
first-generation fields (notebook 02 §4.2).

### 9. Exact S₃ forces $V_{us}=0$; soft breaking is what turns it on

The quark sector (not treated in the draft) has 10 Yukawa couplings — 5 down via
$H$, 5 up via $\tilde H$, which is the same S₃ doublet because the S₃ matrices
are real. $M_u$ and $M_d$ therefore have **identical structure**.

The 1–2 block-diagonalizing rotation satisfies $\tan2\psi=-r$ with $r=v_1/v_2$:
it depends on the **vacuum alone**, never on the Yukawas, so the same $O_{12}$
acts in every sector. The first generation decouples **iff**
$r\,(2-\sqrt{r^2+1})=0$, i.e. $|r|=\sqrt3$ — exactly the
vacuum that preserves the residual $\mathbb Z_2$. Hence with exact S₃

$$V_{\rm CKM}=O_{23}(\theta_u)^{\mathsf T}O_{23}(\theta_d)\ \Longrightarrow\
V_{us}=V_{ub}=V_{cd}=V_{td}=0,$$

against $|V_{us}|=0.2243$. The mechanism is [DasDeyPal16]'s ("unbroken
$\mathbb Z_2$ ⟹ approximate CKM block structure"), **but the block differs**.
There the Cabibbo block stays free and soft breaking generates only the small
(Wolfenstein $\mathcal O(\lambda^2)$ and higher) elements. Here the draft's
assignment gives a 2–3 block, so $V_{us}$ itself vanishes. ($V_{us}=0$ does not
depend on the $\mu_5=\mu_4$ ansatz: at $|r|=\sqrt3$ both entries decouple.)

**Soft breaking fixes it, and costs nothing elsewhere.** Being dimension-2 by
definition it touches no quartic, so *every* boundedness/unitarity result of
finding 4 and notebook 01 carries over unchanged. What it does is release the
vacuum from $r=\sqrt3$: `build_model(soft=True)` adds the four CP-conserving
S₃-breaking quadratics (the $\mathbf2$ pair and the $H_S^\dagger H_{1,2}$ pair;
the hermitian quadratic space is 6-dimensional in the real symmetric case, of
which 2 are invariant), and the tadpole system stops being over-constrained.
A bounded fit now targets all three CKM magnitudes as well as the six masses, and
reproduces all nine simultaneously to numerical precision (`cost`$\sim10^{-26}$),
at a misalignment of $r-\sqrt3\approx-0.38$ (`results/quark_soft_fit.json`).

**Caveat, smaller than it was, but still real.** With 9 parameters against 9
targeted observables the system is exactly determined rather than
under-determined, and the least-squares scan turns up several *distinct* exact
(`cost`$\sim0$) solutions at different $r$ depending on the restart seed — and,
as the switch note above documents, on the numpy/scipy version too. So $r$ is
still an *existence proof*, not a *prediction*, though now for a different reason
(a discrete set of exact branches, not a continuous free family). What the fit
does newly establish is that the two-stage structure does **not** prevent
$|V_{us}|$, $|V_{cb}|$ and $|V_{ub}|$ from being matched simultaneously — the
concern in the previous version of this note (that tying the 2–3 rotation to the
same few mass-fixing parameters might make that impossible) does not hold. What
remains structurally out of reach is the CP phase: these Yukawas are real, so the
CKM matrix built from them is real and the Jarlskog invariant $J$ vanishes
identically, independent of the couplings. Whether a complex-Yukawa extension of
this sector can additionally reproduce $\delta$ (equivalently $J\neq0$) is open.

### 10. The physical basis, the gauge sector, and LFV rates (notebook 03)

`decays.py` registers the physical basis on the `Model` — the blocker this file
recorded for notebook 03 — at **general δ**, with the draft's Scenarios A and B
appearing only as the $\delta\to0,\pi/2$ limits.

**$h_0$ is gauge-phobic, exactly.** $R_H(\delta)$ rotates only in the 1–3 plane,
so $h_0$'s column of $R_S=R_AR_H(\delta)$ is $R_A$'s middle column, orthogonal to
the vacuum direction. Hence for **any** δ

$$g_{h_0VV}=0,\qquad g_{h_1VV}=\cos\delta\,g^{\rm SM},\qquad
g_{h_2VV}=\sin\delta\,g^{\rm SM},\qquad \sum_i g_{h_iVV}^2=(g^{\rm SM})^2 .$$

This also settles the old *"the 125 GeV cut is a mass condition only"* entry:
with the electroweak kinetic terms in place the alignment scenarios are testable,
and $\delta$ is **predicted** at each viable point, giving a coupling-based
SM-likeness cut that notebook 01 now applies (finding 11).

**δ is $-\psi$, not $\alpha-\theta_v$** (corrected; the first version of
`decays.delta_of_point` used the latter). $R_A(\phi,\theta_v)R_H(\delta)=
R_A(\phi,\theta_v+\delta)$ identically, so δ is the rotation *on top of* the
geometric basis and the draft's total angle is $\alpha=\theta_v+\delta$ — there is
no θ_v to subtract from a block angle already measured from that basis. The sign
then comes from the transpose: $R_H$'s $(0,2)$ block is $\mathrm{rot}(\delta)$ and
the physical basis needs $R_H^{\mathsf T}BR_H$ diagonal, while
`solve_mixing_angle_2x2` returns ψ with $RBR^{\mathsf T}$ diagonal. The old formula
left $R_S^{\mathsf T}M_SR_S$ off-diagonal at the $10^5$ level and mis-assigned every
coupling. Since $\psi=\tfrac12\arctan(\cdot)$, **$|\delta|<\pi/4$ always**, so
$\cos^2\delta\ge\tfrac12$ and $h_1$ is *by convention* the larger-coupling state;
scenario B ($\delta\to\pi/2$) is a symbolic limit, not a reachable point. Over the
viable set δ lands in $[-0.32,\,0.32]$ rad — the model sits near, but not at,
scenario A. Pinned in `tests/test_thdm_s3.py::test_delta_is_minus_the_block_angle`,
which checks the two wrong candidates genuinely fail.

**The draft's Scenario-C relations are exact**, not approximate: $Q_i$ is linear
in the columns of $R_S$, so $Q_1(C)=\cos\delta\,Q_1(A)-\sin\delta\,Q_3(A)$ etc.
hold for all δ, and $Q_2$ is entirely δ-independent. Combined with the above,
$h_0$ has no $VV$ coupling *and* δ-independent lepton couplings.

**LFV: each CP-even state has its own channels** (corrected 2026-10-01). At the
exact-S₃ vacuum the electron decouples under $O_{12}$ only along the vacuum
direction of doublet space. The SM-like $h_1$ and $h_2$ therefore never couple to the
electron and give $\tau\mu$ only. The gauge-phobic $h_0$ points orthogonal to the
vacuum, has an electron on **every** coupling, and gives $e\tau$ and $e\mu$ only, with
$\tau\mu=0$ exactly (derived in notebook 03 §7.2, checked at every point and every
$\mu_3$). Results:
- **SM-like state:** $\mathcal B(h\to\tau\mu)$ exceeds [CMS21]'s $0.15\%$ in **18%** of
  entries. This bounds $\mu_3^\ell$, not the scalar sector: the fraction runs
  0.3% → 6.4% → 48% across $\mu_3=0.5,\,0.9,\,1.4$, and the median point sits a
  factor ~9 below the limit.
- **$h_0$:** its $e\tau$ and $e\mu$ rate ratios to $\Gamma_h^{\rm SM}$ are large (medians
  $5\times10^{-3}$, $5\times10^{-2}$). The 125 GeV limits ([CMS21], [CMS23]) do not apply to
  a gauge-phobic non-125 state. [CMS23]'s 110–160 GeV $e\mu$ resonance search is the
  relevant comparison, and it needs $h_0$'s production cross section, which this build
  does not have.

**Superseded readings.**
- "$h_0$ exceeds the $\tau\mu$ limit in 81%; $\mathcal B(h\to\tau e)$ is zero to
  machine precision for every state". Notebook 03 built $M_\ell$ with the alignment's
  $\sqrt3$ typed in, so `g_matrices` put every doublet Yukawa on $G_1$ ($G_2=0$, the trap
  notebook 02 §7 warns about), and it paired draft-basis $G_k$ with a feynlag-basis $R_S$.
  $Q_1$ and $Q_3$ only see $\sum_kv_kG_k=M$ and were right; $Q_2$ was wrong. The fix,
  `lfv.lepton_g_function`, reads $G_k$ off the Lagrangian with all three VEVs symbolic and
  agrees with an independent draft-basis route to $10^{-17}$. It also reproduces the
  draft's **printed** $Q_2(A)$ exactly. The draft had $h_0$ right all along, and only
  this notebook's first version disagreed with it.
- Earlier still: "generically overshoots, a majority of entries". That used the wrong δ
  and was computed before the coupling cut.

So the exact-S₃ vacuum ties the quark and lepton flavour structures together: findings
9 and 10 are the same first-generation decoupling.

**Library work this required** (both in `src/`, pinned by the main suite):
`VSS` decays ($A\to Zh$, $H^\pm\to W^\pm h$ — open at every benchmark point and
gauge-strength, so omitting them would have made heavy-state BRs wrong rather
than merely incomplete), and the spin-0 form factor `A_zero` plus a general
`higgs_diphoton_amplitude` so a charged Higgs can run in the $\gamma\gamma$ loop.

### 11. The soft-broken vacuum: a decoupling-like spectrum (notebook 05)

**Soft means dimension 2, not small.** The S₃-breaking spurions renormalize only in
proportion to themselves, so small values are technically natural ['tHooft80] but
not required. The three free soft terms were sampled as signed uniform variables up
to (500 GeV)², and the size of the breaking was tagged per point instead of imposed.

**Two structural results, both pinned in `tests/test_thdm_s3.py`.**
- Solving the tadpoles for $m_{D2}^2$ (the `build_model(soft=True)` default,
  which notebook 02 was run with) divides by $v_1^2-v_2^2$, a coordinate pole at
  φ = 45°. Solving for $m_{D1}^2$ divides only by $v_1v_2$. `scan_soft.py` uses
  `soft_solve_for="mD1sq"`.
- φ ∈ (0, π/3) is a fundamental domain. The S₃ reflection about 60° maps φ to
  2π/3 − φ, with the same matrix acting on *both* spurion doublets, and gives an
  identical spectrum. Both edges are residual-Z₂ directions, and φ = π/3 is
  notebook 01's alignment. One consequence for notebook 02: each point stands
  for two draft-basis ratios, $r=-\cot(\varphi-\pi/6)$ and $-\tan\varphi$. The
  quark-fit $|r|=1.349$ is φ = 53.44° in the domain.

**The spectrum machinery generalizes.** The geometric rotation still isolates
both Goldstones, with exact zeros at a rational point. The CP-even sector becomes a
full 3×3 with no gauge-phobic state (hVV sum rule to 1e-15). Brute-force mpmath
diagonalization agrees in all three sectors, and the aligned zero-soft slice
reproduces notebook 01 to 1e-10.

**Scan: 2021 points from 1.1×10⁸** (`results/viable_points_soft.json`, ~5 min).
- The quartic cuts are identical to notebook 01's (5.15%).
- No tachyons passes 2.5× more often.
- The 125 GeV coupling cut is where the soft space dies.

**Physics:**
- **Decoupling-like.** The SM-like state is the lightest CP-even state in 99.7% of
  points. The heaviest scalar has a median of 1.0 TeV, against 0.44 TeV in the
  exact scan, where unitarity caps masses at λv². The heavy CP-even, CP-odd and
  charged states are near-degenerate: median spread 4%, and 1.5% above 1 TeV. Non-SM
  CP-even states couple to VV at the 1e-4 level.
- **The exact scan's light scalars do not survive generic soft breaking.** A non-SM
  scalar below 100 GeV appears in 16% of exact points and 0.7% of soft ones. The
  light states are a property of the measure-zero exact slice, not of the S₃
  quartics. λ₄ > 0 is likewise exact-scan-only (100% against 51%).
- **Survivors pile up at both residual-Z₂ edges of φ.** At φ → 0 this is the
  decoupling tail: the solved $m_{D1}^2\propto1/(v_1v_2)$ blows up, and all 34
  points with a state above 5 TeV sit at φ < 7°. At φ → π/3 the breaking is
  moderate (median 3.5v² within 10°). The quark-fit vacuum lies in the populated
  upper range (205 points within ±2°).

**Caveats.** The near-S₃ corner is barely populated under a uniform prior in $m^2$
(15 points with every soft term below v²). Only a *local* minimum is required.
Notebook 03's decays/LFV are not redone, because its δ assumes the aligned 2×2
block. All three are open items.

### 12. Decays and LFV in the soft-broken vacuum (notebook 06)

**Validated against the library, not assumed.** `soft_decays.py` rebuilds each point's
rotations from the stored inputs and computes its couplings numerically:
- the $hH^+H^-$ trilinears come from a lambdified cubic tensor of the quartics (soft terms
  are quadratic and never enter a cubic);
- the VSS couplings are $g/(2c_W)R_{\rm odd}^{\mathsf T}R_{\rm even}$ and
  $(g/2)R_C^{\mathsf T}R_{\rm even}$.

At a non-aligned point all 27 trilinears match feynlag's extractor to $2\times10^{-13}$ GeV,
the 18 VSS couplings match to $10^{-17}$, and the closed-form VSS width equals the engine's.
The φ = π/3, zero-soft slice reproduces corrected notebook 03. The S₃ image vacuum gives
identical observables to $10^{-12}$, once its $m_{D1}^2$ is re-solved rather than read
rounded from the JSON.

**The lepton fit is an arrowhead cubic, and the vacuum restricts it.** In feynlag's basis
the doublet block of $M_\ell$ is a reflection with φ-only eigenvectors, and the third
generation couples to them as $D(\cos\tfrac{3\varphi}2,\sin\tfrac{3\varphi}2)$. Matching the
characteristic polynomial leaves one cubic per eigenvalue-sign pattern, so every branch is
found (random-start least squares never finds another; pinned in
`tests/test_thdm_s3.py::test_soft_lepton_mass_matrix_is_an_arrowhead_in_cos_3phi`).
- Existence depends on $\cos3\varphi$ only, not θ.
- At φ = π/3 every $\mu_3\in(-m_\tau,m_\tau)$ works. Away from it the draft-type branch
  (decoupled entry → electron) survives on ~5% of the $\mu_3$ axis, near $|\mu_3|\approx
  m_\tau$. Soft breaking turns the exact model's free dial into a near-prediction: the
  third generation is essentially the singlet.

**SM-like LFV is pure admixture.** The vacuum direction couples through
$\sum_kv_kG_k/v=M_\ell/v$, which is diagonal in the mass basis at *any* vacuum. So all of the
SM-like state's flavour violation sits in its $1-\kappa_V^2$ admixture of the other two
directions (median $2.4\times10^{-3}$). The total LFV rate correlates with it at 0.88 in log.
This is the alignment limit, and it holds however large the soft breaking is.

**Rates.** All three channels are on in the soft vacuum. Over every (point, $\mu_3$, branch)
the SM-like medians are about $10^{-5}$ (τμ, τe) and $3\times10^{-6}$ (eμ). The fractions
over the limits are 2.9%, 1.9% and **18.3%**: $e\mu$ against [CMS23] is the binding channel.
Only **5 of 2021** scalar points have no lepton configuration under all three limits, so
LFV again bounds the lepton dial, not the scalar sector.

**γγ and VSS.**
- $R_{\gamma\gamma}$ has median 0.97 (90% of points in [0.85, 1.00]). The 48 points below
  0.8 all have a light $H^\pm$ (< 310 GeV) interfering destructively with the $W$, and the
  charged loop decouples to $\kappa_V^2$ above 1.5 TeV. The top coupling is assumed equal to
  $\kappa_V$.
- $A_{1,2}\to Zh$ and $H^\pm_{1,2}\to W^\pm h$ are open almost everywhere (median 20–80 MeV).
  Their couplings to the SM-like state obey $\sum_a|O_{ak}|^2=1-\kappa_V^2$ exactly.

## Open questions / next

- ~~Get the [DasDey14] erratum.~~ **Closed** — see finding 4. It corrects only the
  tadpole equations, so Eq. (4g)'s insufficiency stands against the final published
  record. What remains worth doing is an **independent recomputation of the unitarity
  eigenvalues**: Eq. (37) is now known to be final, but the notebook still only checks
  it for internal consistency (typeset↔code identity, discriminants manifestly ≥ 0).
  [BentoRomaoSilva22] recomputes them for all symmetry-constrained 3HDMs.
- **Complete the boundedness conditions.** The derived condition is exact only on
  the real neutral slice; complex-neutral and charged directions are covered
  numerically, not analytically. [BotoRomaoSilva22] does this properly for
  U(1)×U(1), U(1)×Z₂ and Z₂×Z₂ — not S₃ — and is the methodological model.
  `04_bfb_conditions.ipynb` now pins down exactly what is missing: our $V_N$,
  $V_{CB}$ and $V_G$ all sit inside their framework, and **only the $\lambda_4$
  remainder does not** — its monomials are degree 1 in one off-diagonal bilinear (or
  degree 1 in each of two different ones), shapes their three symmetry classes never
  produce. Bounding it gives a quartic form in $\sqrt{r_i}$ with odd exponents, i.e.
  not a polynomial in $r_i$, so copositivity does not apply. That is the same
  obstruction as notebook 01's $t=\sqrt y$, and the natural construction is a hybrid:
  their machinery for the charged/complex directions, ours for $\lambda_4$. **Not
  implemented — `constraints.py` is untouched.**
- **A global quark fit** — see the caveat in finding 9. Whether the soft-broken
  model can reproduce all four CKM parameters *and* the six quark masses at once
  is open, and the $|V_{cb}|$ result is mild evidence against it without further
  structure. CP violation would additionally need complex Yukawas; everything in
  notebook 02 is real.
- ~~Soft breaking and the scalar spectrum.~~ **Done** — finding 11,
  `05_soft_scalar_scan.ipynb`, `scan_soft.py`.
- **The near-S₃ corner of the soft scan.** A uniform prior in $m^2$ leaves only 15
  of 2021 points with every soft term below $v^2$ (finding 11). Whether the exact
  scan's light gauge-phobic scalars reappear continuously as the breaking shrinks
  needs a targeted scan: log-uniform soft terms and φ near π/3.
- **Global minimum in the soft-broken vacuum.** `scan_soft.py` requires only a
  local minimum (no tachyons). With soft terms the potential can have several
  minima, so a multi-start minimization of the full potential on the survivors is
  the missing vacuum-stability cut.
- ~~Decays and LFV on the soft-broken points.~~ **Done** — finding 12,
  `06_soft_decays.ipynb`, `soft_decays.py`, `lfv.py`. (Doing it surfaced the
  notebook-03 $G_2=0$ split, corrected in finding 10.)
- **Quark sector per soft point.** Heavy-state total widths and the top coupling
  in $R_{\gamma\gamma}$ (set to $\kappa_V$ in notebook 06) both need it.
- **$R_{\gamma\gamma}$ as a cut.** 48 soft points have $R_{\gamma\gamma}<0.8$, all with a
  light $H^\pm$. Turning that into an exclusion needs the production side and a
  cited $\mu_{\gamma\gamma}$.
- **$\mu\to e\gamma$.** Off the exact vacuum the $e\mu$ and $e\tau$ couplings are
  switched on, so the radiative decays will likely constrain more than $h\to e\mu$.
  They need a loop calculation.
- **The neutrino sector** (finding 7) is enumerated but not built: Dirac Yukawas
  on $\tilde H$ plus S₃-allowed Majorana $\nu_R$ masses. feynlag has the pieces
  (`seesaw_mass_matrix`, `MajoranaRotation`, Takagi) — see `examples/sm_seesaw.py`.
- **Quark channels are absent from notebook 03**, so it quotes the *measured*
  SM-like total width rather than computing one, and reports no $b\bar b$ /
  $c\bar c$ BRs. Deliberate: notebook 02 finding 9 showed the exact-S₃ quark
  sector is already excluded ($V_{us}=0$), so fitting it would mean quoting
  rates from a sector known to be wrong. The soft-broken vacuum is where to
  redo it — which also makes the heavy-state ($A_{1,2}$, $H^\pm$) BRs
  incomplete, even though the VSS machinery they need now exists.
- **$Z\gamma$ with a charged scalar** would need its own two-argument form
  factor, the analogue of what `A_zero` did for $\gamma\gamma$. Untouched.
- The LFV rates depend on $\mu_3^\ell$, the one dial the lepton mass relations
  leave free; notebook 03 scans it rather than fixing it from anything.
- ~~A neutral direct-search bound is missing from the scalar scan.~~ **Closed.**
  `model.hvv_function` gives every CP-even state's $hVV$ coupling² alongside its
  mass (validated against brute-force diagonalization of the un-rotated matrix),
  so `constraints.sm_like_cut`/`lep_neutral_mask` cut on the **coupling**: the
  state carrying $g_{hVV}$ must sit at $125\pm3$ GeV with $\kappa^2\ge0.9$, and
  nothing below 114.4 GeV may have $\xi^2\ge0.05$ ([LEP2003]). A light $h_0$ passes
  on the physics — $g_{h_0VV}=0$ exactly — not on an exemption. Because the two
  non-phobic states share the SM strength, the $\kappa^2$ threshold already implies
  the LEP one, so the result is unchanged for any floor in $[0.01,0.10]$: the step
  approximation of [LEP2003] Fig. 10 carries no weight. 2M → 322 under the old
  mass-only cut, 64 under this one.
- The scan samples λ uniformly, which is inefficient given the shape of the
  viable region. A targeted sampler would resolve its boundary far better.
  Mitigated for now by brute force: `scan.py` sweeps 60M points in ~3.5 min
  (30 chunks of 2M) for ~2000 survivors, and chunk 0 reproduces notebook 01's own
  sample exactly, which is what the notebook asserts against the cached file.

## References

- **[GomezBock21]** M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez, *"Scalar and
  gauge sectors in the 3-Higgs Doublet Model under the S₃-symmetry"*,
  Eur. Phys. J. C **81**, 942 (2021),
  [arXiv:2102.02800](https://arxiv.org/abs/2102.02800),
  [doi:10.1140/epjc/s10052-021-09731-3](https://doi.org/10.1140/epjc/s10052-021-09731-3).
  Eq. (2)/(6) the $(a,\ldots,h)$ potential; Eq. (8) the $v=246$ GeV constraint;
  Eq. (13) the √3 alignment; Eq. (24) the θ parametrization; Eq. (29) the
  geometric rotation; Eqs. (30)–(33) the pseudoscalar/charged closed-form
  masses; Eqs. (54)–(55) the alignment scenarios A/B. Also cited in
  `docs/manual/ssb.md`.
- **[DasDey14]** D. Das, U. K. Dey, *"Analysis of an extended scalar sector with
  S₃ symmetry"*, Phys. Rev. D **89**, 095025 (2014),
  [arXiv:1404.2491](https://arxiv.org/abs/1404.2491) (v2, May 2014),
  [doi:10.1103/PhysRevD.89.095025](https://doi.org/10.1103/PhysRevD.89.095025).
  Eq. (3c) the λ₁…λ₈ potential; Eq. (4a)–(4g) boundedness-from-below;
  Eq. (36) + (37a)–(37l) the tree-unitarity eigenvalues. This is the actual
  source of the constraints — [GomezBock21] §2.2 delegates to it.
  **Erratum: Phys. Rev. D 91, 039905 (2015)** — *not* posted to arXiv (the arXiv
  record stops at v2), but freely readable at APS (bronze OA; APS returns 403 to
  automated fetches, which is what made it look inaccessible). **Obtained and read.**
  It corrects Eqs. (9a)–(9c), the tadpole conditions, and nothing else, stating that
  the changes "do not affect any of our conclusions". Eq. (4) and Eqs. (36)–(37) —
  what `constraints.py` transcribes — are unaffected. See finding 4 and notebook 01
  §2.2.
- **[BentoRomaoSilva22]** M. P. Bento, J. C. Romão, J. P. Silva, *"Unitarity
  bounds for all symmetry-constrained 3HDMs"*, JHEP **08** (2022) 273,
  [arXiv:2204.13130](https://arxiv.org/abs/2204.13130),
  [doi:10.1007/JHEP08(2022)273](https://doi.org/10.1007/JHEP08(2022)273).
  Post-dates the erratum; the cross-check target for the unitarity eigenvalues.
- **[BotoRomaoSilva22]** R. Boto, J. C. Romão, J. P. Silva, *"Bounded from below
  conditions on a class of symmetry constrained 3HDM"*, Phys. Rev. D **106**,
  115010 (2022), [arXiv:2208.01068](https://arxiv.org/abs/2208.01068),
  [doi:10.1103/PhysRevD.106.115010](https://doi.org/10.1103/PhysRevD.106.115010).
  Covers U(1)×U(1), U(1)×Z₂, Z₂×Z₂ — not S₃ — but is the methodological model for
  doing the boundedness analysis properly (BFB-n and BFB-c directions separately).
  **Read and executed in `04_bfb_conditions.ipynb`**: the $V_4=V_N+V_{CB}+V_G$ split,
  $0\le z_{ij}\le r_ir_j$, BFB-n as copositivity of a 3×3 matrix, and their
  lower-bound strategy (bound $V_{CB}$ and $V_G$ by their worst cases, then one
  copositivity test) giving *sufficient* conditions. Note their general-3HDM couplings
  are also called $\lambda_1\ldots\lambda_9$ and are **not** ours — their $\lambda_4$
  is $(\phi_1^\dagger\phi_1)(\phi_2^\dagger\phi_2)$.
- **[LFVHD]** M. Zeleny-Mora, M. Mondragón, T. A. Valencia-Pérez, *"Exploring LFV
  Higgs decays in the Three Higgs Doublet Model"*, draft —
  `paper_lfvhd/LFVHD_3HDMS3.tex`. Supplies the S₃ lepton assignment, the
  Yukawa Lagrangian, the charged-lepton mass matrix and its diagonalization,
  and the $Q_i$ LFV couplings. Checked (and patched) in findings 6–8.
- **[LEP2003]** ALEPH, DELPHI, L3, OPAL Collaborations and the LEP Working Group
  for Higgs Boson Searches, *"Search for the Standard Model Higgs boson at LEP"*,
  Phys. Lett. B **565** (2003) 61, arXiv:hep-ex/0306033,
  doi:10.1016/S0370-2693(03)00614-2. Fig. 10 is the 95% CL bound on
  $\xi^2=(g_{HZZ}/g^{\rm SM}_{HZZ})^2$; $\xi^2=1$ is excluded below 114.4 GeV.
- **[CMS21]** CMS Collaboration, *"Search for lepton-flavor violating decays of the
  Higgs boson in the μτ and eτ final states in proton-proton collisions at
  √s = 13 TeV"*, Phys. Rev. D **104**, 032013 (2021),
  [arXiv:2105.03007](https://arxiv.org/abs/2105.03007). B(H→μτ) < 0.15%,
  B(H→eτ) < 0.22% at 95% CL.
- **[CMS23]** CMS Collaboration, *"Search for the lepton-flavor violating decay of the
  Higgs boson and additional Higgs bosons in the eμ final state in proton-proton
  collisions at √s = 13 TeV"*, Phys. Rev. D **108**, 072004 (2023),
  [arXiv:2305.18106](https://arxiv.org/abs/2305.18106),
  [doi:10.1103/PhysRevD.108.072004](https://doi.org/10.1103/PhysRevD.108.072004).
  B(H→eμ) < 4.4×10⁻⁵ at 95% CL; also a search for additional scalars at 110–160 GeV.
- **[DasDeyPal16]** D. Das, U. K. Dey, P. B. Pal, *"S₃ symmetry and the quark
  mixing matrix"*, Phys. Lett. B **753**, 315 (2016),
  [arXiv:1507.06509](https://arxiv.org/abs/1507.06509),
  [doi:10.1016/j.physletb.2015.12.038](https://doi.org/10.1016/j.physletb.2015.12.038).
  The quark-sector companion: an unbroken Z₂ leaves the CKM matrix approximately
  block diagonal, and soft S₃ breaking in the scalar sector generates the small
  elements. Independently re-derived in finding 9. (This is very likely the
  `Das:2015sca` key cited by the draft, though its `.bib` is not in the repo.)
- **[BabuWuXu23]** K. S. Babu, Y. Wu, S. Xu, *"Fermion Masses, Neutrino Mixing
  and Higgs-Mediated Flavor Violation in 3HDM with $S_3$ Permutation
  Symmetry"*, [arXiv:2312.15828](https://arxiv.org/abs/2312.15828) (2023).
  An alternative published S₃-3HDM Yukawa treatment; not used — the draft
  supplied the assignment instead.
- **[PDG]** S. Navas *et al.* (Particle Data Group), *Review of Particle
  Physics*, Phys. Rev. D **110**, 030001 (2024),
  [doi:10.1103/PhysRevD.110.030001](https://doi.org/10.1103/PhysRevD.110.030001).
  Quark masses and CKM elements used in the notebook-02 fit.
- **['tHooft80]** G. 't Hooft, *"Naturalness, chiral symmetry, and spontaneous
  chiral symmetry breaking"*, NATO Sci. Ser. B **59**, 135 (1980),
  [doi:10.1007/978-1-4684-7571-5_9](https://doi.org/10.1007/978-1-4684-7571-5_9).
  Technical naturalness: a parameter may be small if setting it to zero enlarges
  the symmetry. Used in finding 11 for why soft terms need not be small, but may be.
- **[Yildirim26]** E. Yildirim, *"Double SM-like Higgs Production at future
  $e^+e^-$ colliders in the 3-Higgs Doublet Model under the $S_3$ symmetry"*,
  [arXiv:2604.24421](https://arxiv.org/abs/2604.24421) (2026). Applies
  perturbative unitarity, vacuum stability and LHC/Tevatron data to this same
  model; a cross-check target for the constraint implementation.

# Validated against MadGraph

The strongest correctness statement a Lagrangian→Feynman-rule tool can make is
that its output computes the *right numbers* in a downstream generator.
feynlag's Standard-Model UFO is round-tripped through **MadGraph5_aMC@NLO**: the
exported model is imported, real cross sections are computed, and they are
compared to the standard `sm` model shipped with MadGraph at the *same*
parameter point.

Reproduce it with:

```bash
python scripts/madgraph_roundtrip.py
```

which downloads MadGraph if needed, exports the SM UFO
(`scripts/export_sm_ufo.py`), and runs both models on both processes.

## Parameter point

The export uses MadGraph's stock electroweak input scheme
$(\alpha_{\rm EW}^{-1}, G_F, M_Z) = (132.50698,\ 1.16639\times10^{-5},\
91.1876)$, from which the same tree-level relations the stock `sm` uses give

$$M_W = 80.4185\ \text{GeV},\quad M_Z = 91.1876\ \text{GeV},\quad v = 246.22\ \text{GeV},$$

so the two models have numerically identical couplings by construction.
Lepton-collider beams: $\sqrt s = 200$ GeV, no PDF (`lpp=0`).

## Cross sections

| Process | feynlag UFO | stock `sm` | agreement |
|---|---|---|---|
| $e^+e^-\to\mu^+\mu^-$ | $2.7878 \pm 0.0027$ pb | $2.7878 \pm 0.0027$ pb | 0.00% (0.0σ) |
| $e^+e^-\to W^+W^-$ | $19.498 \pm 0.058$ pb | $19.497 \pm 0.058$ pb | 0.01% (0.0σ) |

Both agree with the stock model to within Monte-Carlo error.

## What each process validates

- **$e^+e^-\to\mu^+\mu^-$** exercises the photon and $Z$ FFV couplings and the
  $Z$ propagator (γ and $Z$ $s$-channel interference).
- **$e^+e^-\to W^+W^-$** is the **gauge-cancellation acid test**. The $\nu$
  $t$-channel diagram grows with energy and is tamed only by a precise
  cancellation against the $\gamma/Z$ $s$-channel diagrams — which requires the
  relative signs across the $W\bar\nu e$ FFV vertex and the $WW\gamma/WWZ$
  triple-gauge vertices to be exactly right. A wrong relative sign gives
  ≈98 pb instead of ≈19.5 pb; getting 19.5 pb confirms the cancellation holds.

## Bugs the round-trip caught

Building this benchmark surfaced two real UFO-export bugs (both fixed, now
regression-tested) and one convention subtlety — exactly the kind of thing that
is invisible to unit tests but breaks a real generator run:

1. **Relative imports** in the generated UFO (`from .object_library …`) broke
   MadGraph's param-card generation, which runs the modules as standalone
   scripts. Real UFO models use absolute imports; feynlag now does too
   (`test_ufo_export.py::test_ufo_uses_absolute_imports`).
2. **The FFV export dropped the Feynman-rule $i$** that the UFO coupling
   convention carries (the bosonic and VVV couplings already had it). This is
   invisible in $e^+e^-\to\mu^+\mu^-$ (an overall phase cancels in $|\mathcal
   M|^2$) but breaks the FFV↔VVV interference in $e^+e^-\to W^+W^-$. Fixed in
   `add_fermion_vertex`.
3. **The triple-gauge coupling** needs a sign flip for the electroweak
   vertices but not for the gluon. This looked for a long time like an
   unresolved MadGraph convention mismatch; it is neither a mismatch nor a
   bug. feynlag's symbols label **fields**, a UFO leg labels a **particle**,
   and the field $W^+$ annihilates a $W^+$ but *creates* a $W^-$ — so the leg
   carrying the symbol `Wp` is UFO's `W-` leg. Emitting legs under naive
   names transposes each conjugate pair, and `VVV1` is totally antisymmetric:
   one conjugate pair $\Rightarrow -1$ (the electroweak cubics), none
   $\Rightarrow +1$ ($ggg$). `gauge_basis.ufo_leg_sign` computes it, and the
   export no longer applies anything by hand.

   The same rule explains why VVS/VVSS/VVVV never needed a flip — their
   structures are *invariant* under that transposition — and it reproduces
   MadGraph's `GC_4`, `GC_53`, `GC_10`, `GC_3` and `GC_61` with no other
   adjustment. It is also behaviour-preserving here: the old export emitted
   $(\gamma,W^+,W^-)$ with $-ie$, the same vertex as the $(\gamma,W^-,W^+)$
   with $+ie$ emitted now, so the 19.50 pb below is unchanged.

This is the payoff of feynlag's verification-first design: the round-trip is a
harness that turns "the model looks right" into "the model computes the right
cross section."

## Quartic gauge couplings: the $\gamma\gamma\to W^+W^-$ Ward identity

Cross sections are not the only oracle MadGraph offers. Its `check` command
evaluates a process's matrix element at random phase-space points and runs
three model-internal consistency tests — Lorentz invariance, the gauge/BRS
(Ward) identity, and leg-permutation symmetry — with **no beams and no Fortran
cross-section run**, so it is fast enough to use as a routine acceptance test.

$\gamma\gamma\to W^+W^-$ is the ideal probe for the quartic: its only diagrams
are $t$- and $u$-channel $W$ exchange plus the $\gamma\gamma W^+W^-$ contact
term, and gauge invariance holds *only* if the quartic's normalization relative
to the exchange diagrams is exactly right. A wrong quartic cannot hide.

```bash
mg5_aMC <<< 'import model /path/to/FEYNLAG_SM
check a a > w+ w-'
```

| Model | Lorentz invariance | Gauge (BRS) ratio | Result |
|---|---|---|---|
| stock `sm` | $3.1\times10^{-15}$ | $5.0\times10^{-28}$ | Passed |
| feynlag SM UFO | $2.6\times10^{-15}$ | $1.0\times10^{-27}$ | Passed |
| feynlag, quartic $\times 3$ | $3.7\times10^{-1}$ | $2.6\times10^{-2}$ | **Failed** |

The third row is the point. Before this was validated, `assemble_vvvv`
reconstructed **3× the true vertex** (the three UFO `VVVV1/2/3` structures are
linearly dependent, so the decomposition is a 1-parameter family and the
symmetric representative carries a $1/3$ that the hard-coded normalization
omitted). Re-exporting with the old value makes both checks fail
catastrophically and inflates the matrix element by ~500×. The unit tests could
not see it, because the "independent" ground truth had been built in the same
over-complete convention and compared coefficient *lists*, which cannot detect
a common factor.

The exported couplings also match stock `sm` entry by entry at the shared
parameter point, compared in the convention-free metric-pair basis since
MadGraph's `VVVV` basis differs from feynlag's
(`tests/test_ufo_sm_bosonic.py`, in CI):

| Vertex | feynlag | stock `sm` |
|---|---|---|
| $\gamma\gamma W^+W^-$ | $ie^2$ | `GC_5` |
| $W^+W^-W^+W^-$ | $-ig^2$ | `GC_35` |
| $W^+W^-ZZ$ | $ig^2c_w^2$ | `GC_36` |
| $\gamma W^+W^-Z$ | matches on MG's two-structure `VVVV5` shape | `GC_57` |
| $hW^+W^-$, $hZZ$, $hhW^+W^-$, $hhZZ$ | extractor output, unrescaled | `GC_72`, `GC_81`, `GC_34`, `GC_65` |

## QCD: the $u\bar u\to gg$ gauge check

Until this, feynlag's **QCD sector had never been exercised in any process**.
The cross-section round-trip covers $e^+e^-\to\mu^+\mu^-$ and
$e^+e^-\to W^+W^-$, both QCD-free, and the $\gamma\gamma\to W^+W^-$ check
above is electroweak. Every QCD result in the repo was a symbolic pin of
feynlag's own output — which is exactly how three separate colour defects
survived in it.

$u\bar u\to gg$ is the right probe: $t$- and $u$-channel quark exchange (no
$ggg$) interferes with the $s$-channel gluon (one $ggg$), so the amplitude is
**linear** in the triple-gluon coupling and its sign is observable.
$gg\to gg$ will **not** do — it goes as $ggg^2$ and is sign-blind.

```bash
python scripts/madgraph_qcd.py     # exports the UFO and runs all three models
```

| Model | Lorentz invariance | Gauge (BRS) ratio | Result |
|---|---|---|---|
| stock `sm` | $1.6\times10^{-16}$ | $2.3\times10^{-30}$ | Passed |
| feynlag QCD UFO | $3.0\times10^{-15}$ | $2.0\times10^{-30}$ | Passed |
| feynlag, $ggg$ sign flipped | $3.6\times10^{-1}$ | $4.0$ | **Failed** |

The third row is the point, as before: a pass alone would only mean "nothing
crashed". `scripts/madgraph_qcd.py` exports the broken variant itself and
exits non-zero unless it fails.

### Two defects this caught, both invisible to everything symbolic

**1. Every coupling was tagged `QED`.** `writer.py` assigned
`order = {'QED': max(n_legs-2, 1)}` unconditionally, so the gluon vertices
carried a QED order. MadGraph rejects that outright —
`CRITICAL: Model with non QCD emission of gluon (found 6 of those)` — and then
builds the wrong diagram set, which made the first run of this check
*inconclusive* rather than failing: the broken control passed too, because the
$ggg$ diagram was not in the amplitude at all. The order is now QCD whenever
the colour tensor is not the singlet `'1'`; the power was already right
($ggg\to1$, $gggg\to2$, $qqg\to1$).

**2. The $qqg$ colour tensor was transposed.** The writer emitted
`T(3,1,2)` with its `[bar, field, boson]` leg order, reasoning from
`fermion_gauge_current`'s `T[r,c]` Lagrangian index order. But UFO's
`T(a,i,j)` takes `i` as the **fundamental** index, so it must be `T(3,2,1)` —
what MadGraph's stock `sm` emits for `[u~, u, g]`. Transposing a hermitian
$T^a$ conjugates it, which flips the sign of the $ggg$ interference: with
`T(3,1,2)` the model **failed** both the Lorentz and gauge checks
(rel. diff $0.50$, BRS ratio $3.4$).

That second one is worth dwelling on. The failure was initially masked twice
over — once by the coupling-order bug, and once by a regex in the check parser
that dropped failing sections (MadGraph indents `JAMP` rows under a failure),
so a hard failure first appeared as a *missing* section. And the naive reading
of the surviving numbers was that flipping $ggg$ "fixed" it — which would have
been precisely the wrong conclusion, since $ggg=-g_s$ matches stock `GC_10`.
The colour index order was the real defect, and only the exported couplings
matching stock ($ggg=-g_s$, $qqg=ig_s$, $gggg=ig_s^2$) made that legible.

## Four-fermion operators: the muon-decay width

The four-fermion (FFFF) export is validated the same way, against an analytic
result. `scripts/madgraph_fermi.py` exports a Fermi-theory UFO for

$$\mathcal L \supset -\tfrac{4G_F}{\sqrt2}(\bar\nu_\mu\gamma^\mu P_L\mu)(\bar e\,\gamma_\mu P_L\nu_e) + \text{h.c.}$$

and asks MadGraph for the muon partial width, comparing to the textbook

$$\Gamma(\mu^-\to e^-\bar\nu_e\nu_\mu) = \frac{G_F^2 m_\mu^5}{192\pi^3} = 3.009\times10^{-19}\ \text{GeV}\quad(\tau_\mu = 2.19\ \mu\text{s}).$$

| Quantity | feynlag UFO (MadGraph) | analytic | agreement |
|---|---|---|---|
| $\Gamma(\mu\to e\nu\nu)$ | $3.007\times10^{-19}$ GeV | $3.009\times10^{-19}$ GeV | 0.07% |

This confirms the exported `FFFF*` Lorentz structure, the fermion-flow leg
assignment, and the coupling normalization are all physically correct — the
part unit tests (symbolic extraction + coupling round-trip) cannot see. It also
surfaced a UFO-export requirement specific to contact interactions: **both the
operator vertex and its Hermitian conjugate must be exported**, since MadGraph
needs the conjugate pair to establish a consistent fermion-number flow through
the four-fermion vertex (a UFO carrying only one fails diagram generation). Not
in CI (each launch compiles Fortran).

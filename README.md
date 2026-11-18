# feynlag

Tree-level Feynman rules from Beyond-Standard-Model Lagrangians, in pure
[SymPy](https://sympy.org).

You declare particle fields with their gauge and discrete-symmetry
representations, write the Lagrangian explicitly with library building blocks
(`Dmu`, `dag`, `Bilinear`), and `feynlag` takes it from there:

- gauge / discrete invariance and hermiticity checks,
- electroweak symmetry breaking: VEV expansion, tadpole conditions,
- mass-matrix extraction and diagonalization (orthogonal, unitary, SVD for
  Dirac fermions, Takagi for Majorana),
- rotation from the weak basis to the physical (mass) basis,
- vertex extraction (SSS, SSSS, VSS, VVS, VVSS, VVV, VVVV, FFS, FFV, plus
  four-fermion and Majorana/Weinberg operators) with derivative couplings
  taken to momentum space,
- model consistency in one call (`Model.validate()`): invariance,
  hermiticity, anomaly cancellation, charge conservation,
- export: LaTeX vertex tables and UFO (MadGraph et al.),
- phenomenology (`feynlag.pheno`): decay widths and branching ratios
  (tree-level 1→2, off-shell `h→WW*/ZZ*`, loop-induced `h→gg/γγ/Zγ`) and
  tree-level 2→2 cross sections from every diagram with its interference:
  four-fermion processes, external photons/Z/W/Higgs (`e⁺e⁻→W⁺W⁻`, `Zh`,
  Compton), and coloured processes at parton level, external gluons and
  `gg→gg` included. Calculators are built from the model
  (`ScatteringCalculator.from_model`), processes are named
  (`"e- e+ > w+ w-"`), and results come in the unit you ask for.

Parameters are split into **external** (fixed by experiment, e.g. `v`, `m_h`,
`g`) and **internal** (derived: tadpole solutions, mixing angles, inverted
quartics), forming a dependency chain that closes the UFO parameter card.

## Documentation

Full docs, including an **Algorithms Manual** deriving the physics and
design of every pipeline stage (invariance checking, EWSB/tadpoles, mass
matrices, diagonalization, vertex extraction, export), tutorial notebooks,
an examples gallery, and the API reference:
**https://moiseszeleny.github.io/feynlag/**

## Install

```bash
pip install feynlag            # add [numeric] for SciPy-backed integration
```

For development:

```bash
pip install -e .[dev]
pytest
```

## Quick tour

```python
import sympy as sp
from feynlag import (ExternalParameter, InternalParameter, SU2, U1, Scalar,
                     Lagrangian, Model, Dmu, dag)

gw  = ExternalParameter("gw", 0.6535, positive=True)
g1  = ExternalParameter("g1", 0.3580, positive=True)
SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

v   = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
lam = ExternalParameter("lam", 0.129)
mu2 = InternalParameter("mu2", unit_dim=2)      # defined by the tadpole

H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
           component_names=["Gp", "H0"])
H.expand_vev({H.components[1]: v})              # H0 -> (v + h + i G0)/√2

HdH = (dag(H) * H.mat)[0]
DH  = Dmu(H)
L = Lagrangian()
L.add((dag(DH) * DH)[0], sector="kinetic")
L.add(mu2.s * HdH - lam.s * HdH**2, sector="potential")

m = Model("SM", gauge_groups=[SU2L, U1Y],
          fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
          parameters=[gw, g1, v, lam, mu2], lagrangian=L)

m.check_invariance()          # gauge invariance, hermiticity, dim ≤ 4
m.solve_tadpoles([mu2])       # {mu2: lam v²}, registered as internal

h = sp.Symbol("H0_r", real=True)
m.mass_matrix([h])            # Matrix([[2 lam v²]])
m.feynman_rules([h])          # {(h,h,h): -6i lam v, (h,h,h,h): -6i lam}
```

See `examples/` (index: [`examples/README.md`](examples/README.md)) for full runs: `sm_scalar_gauge.py` (complete SM: Higgs +
electroweak gauge + leptons + quark/QCD sector), `sm_vll.py` (SM + a
vector-like lepton doublet, biunitary mass-matrix diagonalization),
`sm_u1x.py` (SM × U(1)_X with a Z′, symbolic charges, chained rotations),
`thdm.py` (2HDM with the α rotation), `thdm_s3.py` (3HDM+S₃, where the
tadpole conditions force the √3 vacuum alignment), `sm_ckm.py` (CKM quark
mixing), `fermi_theory.py` (four-fermion muon decay), `sm_weinberg.py` and
`sm_seesaw.py` (Majorana neutrino masses), `sm_decays.py` and
`sm_higgs_decays.py` (widths and the full Higgs branching-ratio table), and
2→2 scattering: `ee_to_ff.py`, `ee_scattering.py` (MadGraph's γ+Z
`e+e-→μ+μ-` and Bhabha), `ee_to_ww.py` (`e+e-→W+W-` with its gauge
cancellation, and `e+e-→Zh`), `scalar_qed.py` (charged scalars through the
`VSS` vertex) and `qcd_scattering.py` (four-quark, external-gluon and
`gg→gg` parton scattering). The
[docs site](https://moiseszeleny.github.io/feynlag/) walks these models
stage by stage in ten executed tutorial notebooks.

### Phenomenology

```python
from feynlag import standard_model
from feynlag.pheno import DecayCalculator, DiracParticle, ScatteringCalculator

sm = standard_model()                       # SM electroweak + one lepton generation
(L, eR), = sm.leptons
e  = DiracParticle("e", L.components[1], eR.components[0], 0)
nu = DiracParticle("nu", L.components[0], None, 0)      # left-handed only

decays = DecayCalculator.from_model(sm.model, particles=[e, nu],
                                    fermion_sectors=("gauge",))
print(decays.decay_table("z", unit="MeV"))  # Γ(Z→νν̄), Γ(Z→e⁺e⁻), BRs

calc = ScatteringCalculator.from_model(
    sm.model, [e, nu], widths={sm.physical.Z: 2.4952},
    aliases={"e-": e.particle, "e+": e.antiparticle})
ww = calc.process("e- e+ > w+ w-")          # ν exchange + γ/Z s channel
print(ww.summary(sqrt_s=200, unit="pb"))    # diagrams, σ, A_FB
```

The bosons, their masses (read off the physical Lagrangian), the conjugate
pairs and the parameter point all come from the model. `feynlag.pheno` holds
this user API, and `feynlag.pheno.engine` the building blocks (traces,
diagrams, kinematics, colour tensors).

## Validation

The test suite pins the physics, not just the code (dual verification:
symbolic difference **and** random-point numeric checks):

- SM Higgs: `μ² = λv²`, `m_h² = 2λv²`, `h³ = −3i m_h²/v`, `h⁴ = −3i m_h²/v²`
- SM gauge: `m_W = gv/2`, Weinberg rotation, `hWW = i g m_W g^{μν}`,
  `γW⁺W⁻ = e`, `ZW⁺W⁻ = g cosθ_W`, scalar-QED Goldstone vertices
- SM leptons: `hℓℓ = −i m_ℓ/v`, `Wℓν = i g/√2 γ^μ P_L`,
  Z couplings ∝ `T³ − Q sin²θ_W`
- 2HDM: tadpoles, all three mass matrices and rotation angles vs the
  Gunion–Haber/Branco expressions
- 3HDM+S₃: invariant potential from the library's CG products; the tadpole
  system forces the `√3` alignment
- UFO: generated model imports cleanly; parameters resolve in dependency
  order; `hWW` coupling pinned numerically
- 2→2 scattering, straight from the extracted couplings:
  - MadGraph's `e+e-→μ+μ-` (2.7876 pb vs 2.7878 ± 0.0027 inside `|η|<2.5`)
    and `e+e-→W+W-` (19.447 pb vs 19.498 ± 0.058);
  - the exact high-energy gauge cancellation in `e+e-→W+W-`;
  - the Ellis–Stirling–Webber parton table (four-quark, gluon and `gg→gg`
    entries), exactly and symbolically;
  - every amplitude against an independent explicit-spinor/polarization
    evaluator.

## Status / roadmap

feynlag 0.5 is a **beta**: the tree-level pipeline above is complete and
tested (746 tests pinning physical results). Both the exported SM UFO and the
built-in 2→2 engine are **cross-checked against MadGraph**: `e+e-→μ+μ-` and
the gauge-cancelling `e+e-→W+W-` reproduce the stock `sm` cross sections to MC
precision, and a four-fermion UFO reproduces the muon width (see
[`docs/benchmark.md`](https://github.com/moiseszeleny/feynlag/blob/main/docs/benchmark.md)).

Known limitations (planned, see [`docs/roadmap.md`](https://github.com/moiseszeleny/feynlag/blob/main/docs/roadmap.md)):

- no R_ξ gauge fixing or ghosts (Goldstone bosons are kept, but no
  gauge-fixing terms, ghost vertices or ξ dependence);
- Majorana vertices are symbolic-only, not yet exported to UFO;
- 2→2 scattering does not yet cover electroweak four-boson processes
  (`WW→WW`, `γγ→WW`, `VV→hh`: `gg→gg` is the only four-boson process);
  coloured cross sections are parton level, with no PDFs;
- in 1→2 decays the derivative couplings (`VSS`, `VVV`) are not yet
  squared;
- no NLO / UFO 2.0 extensions.

Unsupported cases raise `NotImplementedError` rather than returning a
plausible-looking wrong answer.

## License

MIT — see [`LICENSE`](https://github.com/moiseszeleny/feynlag/blob/main/LICENSE).

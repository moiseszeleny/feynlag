"""Full tree-level ``e⁺e⁻ → μ⁺μ⁻`` and Bhabha from the Lagrangian — Tier 3.

Builds the SM electroweak sector with massless electrons and muons at
MadGraph's benchmark point (``docs/benchmark.md``: α⁻¹=132.50698, G_F, M_Z),
hands it to :class:`~feynlag.pheno.ScatteringCalculator`, and lets the
topology enumerator find every diagram:

- ``e⁺e⁻→μ⁺μ⁻``: the γ and Z s-channel diagrams, **with their interference**
  — 2.844 pb over the full angle, 2.788 pb inside MadGraph's default lepton
  acceptance ``|η|<2.5`` (MadGraph: 2.7878 ± 0.0027 pb), and the large
  ``A_FB ≈ 0.57`` that only the γ–Z interference produces at √s = 200 GeV;
- Bhabha ``e⁺e⁻→e⁺e⁻``: four diagrams, (s, t) × (γ, Z), the s channel with
  the relative fermion sign −1.  The t-channel photon makes the full-angle
  cross section diverge, so it is quoted inside an angular cut.

Run with::

    python examples/ee_scattering.py
"""

import math

import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_scaffold,
    fermion_gauge_current, to_physical_basis,
)
from feynlag.pheno import DiracParticle, ScatteringCalculator

# MadGraph's stock electroweak input scheme (docs/benchmark.md)
AEWM1, GF, MZ, WZ = 132.50698, 1.16639e-5, 91.1876, 2.4952
AEW = 1 / AEWM1
EE = math.sqrt(4 * math.pi * AEW)
MW = math.sqrt(MZ**2 / 2 + math.sqrt(MZ**4 / 4 - AEW * math.pi * MZ**2
                                     / (GF * math.sqrt(2))))
SW2 = 1 - MW**2 / MZ**2
GW, G1 = EE / math.sqrt(SW2), EE / math.sqrt(1 - SW2)
VEV = 2 * MW * math.sqrt(SW2) / EE
GEV2_TO_PB = 0.3893794e9
SQRT_S = 200.0


def build():
    """SM electroweak gauge sector + two massless lepton generations."""
    ew = electroweak_scaffold(gw=GW, g1=G1, v=VEV, mh=125.0)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)

    def doublet(name, comps):
        return WeylFermion(name, reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                           chirality="L", nflavors=1, component_names=comps)

    def singlet(name, comp):
        return WeylFermion(name, reps={U1Y: -1}, chirality="R", nflavors=1,
                           component_names=[comp])

    Le, eR = doublet("Le", ["nueL", "eL"]), singlet("eR", "eR")
    Lmu, muR = doublet("Lmu", ["numuL", "muL"]), singlet("muR", "muR")
    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR, Lmu, muR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("SM_ee", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR, Lmu, muR],
                  parameters=ew.parameters, lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)

    e = DiracParticle("e", Le.components[1], eR.components[0], 0)
    mu = DiracParticle("mu", Lmu.components[1], muR.components[0], 0)
    calc = ScatteringCalculator(
        model, [e, mu], masses={ph.A: 0, ph.Z: MZ}, widths={ph.Z: WZ},
        boson_fields=[ph.Z, ph.A], conjugate_map=ph.cmap)
    return calc, e, mu, {ew.gw.s: GW, ew.g1.s: G1}


def pb(sigma, proc):
    return float(sigma.subs(proc.kinematics.s, SQRT_S**2)) * GEV2_TO_PB


def main():
    calc, e, mu, values = build()
    eta_cut = math.tanh(2.5)

    print("=" * 68)
    print(f"e+e- -> mu+mu- at sqrt(s) = {SQRT_S:.0f} GeV (feynlag.pheno, Tier 3)")
    print("=" * 68)
    mumu = calc.process((e.particle, e.antiparticle),
                        (mu.particle, mu.antiparticle),
                        numeric=True, extra=values)
    for d in mumu.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator}, fermion sign {d.sign:+d}")
    full = pb(mumu.cross_section(), mumu)
    cut = pb(mumu.cross_section(cos_range=(-eta_cut, eta_cut)), mumu)
    afb = float(mumu.forward_backward_asymmetry().subs(mumu.kinematics.s,
                                                       SQRT_S**2))
    print(f"  sigma (full angle)        = {full:.4f} pb")
    print(f"  sigma (|eta| < 2.5)       = {cut:.4f} pb   "
          f"[MadGraph: 2.7878 +- 0.0027 pb]")
    print(f"  A_FB                      = {afb:.4f}")

    print("\n" + "=" * 68)
    print("Bhabha e+e- -> e+e-")
    print("=" * 68)
    bhabha = calc.process((e.particle, e.antiparticle),
                          (e.particle, e.antiparticle),
                          numeric=True, extra=values)
    for d in bhabha.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator}, fermion sign {d.sign:+d}")
    sigma = bhabha.numeric_cross_section(SQRT_S**2, cos_range=(-eta_cut, eta_cut))
    sigma *= GEV2_TO_PB
    print(f"  sigma (|eta| < 2.5)       = {sigma:.2f} pb")

    print("\nThe interference terms (gamma-Z; s-t) are what Tiers 1-2 could not "
          "compute.")


if __name__ == "__main__":
    main()

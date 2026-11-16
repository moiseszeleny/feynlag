"""``e⁺e⁻ → W⁺W⁻`` and ``e⁺e⁻ → Zh`` from the Lagrangian — Tier 4.

``e⁺e⁻ → W⁺W⁻`` is the gauge-cancellation acid test.  It has three diagrams:
the neutrino exchange, and the photon and Z in the s channel through the
triple-gauge vertices.  The neutrino diagram alone grows with energy and
violates unitarity; the sum does not, but only if the relative sign between
the ``W ℓ ν`` current and the triple-gauge couplings is exactly right.  Here
every coupling is extracted from the model — the fermion currents by the
vertex extractor, the ``AWW``/``ZWW`` couplings by ``Model.gauge_vertices`` —
and the cross section is compared with MadGraph's 19.498 ± 0.058 pb at
√s = 200 GeV (``docs/benchmark.md``).

Run:  python examples/ee_to_ww.py
"""

import math

import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_scaffold,
    fermion_gauge_current, to_physical_basis,
)
from feynlag.pheno import DiracParticle, ScatteringCalculator, line_amplitude

# MadGraph's stock electroweak input scheme (docs/benchmark.md)
AEWM1, GF, MZ, WZ, MH = 132.50698, 1.16639e-5, 91.1876, 2.4952, 125.0
AEW = 1 / AEWM1
EE = math.sqrt(4 * math.pi * AEW)
MW = math.sqrt(MZ**2 / 2 + math.sqrt(MZ**4 / 4 - AEW * math.pi * MZ**2
                                     / (GF * math.sqrt(2))))
SW2 = 1 - MW**2 / MZ**2
GW, G1 = EE / math.sqrt(SW2), EE / math.sqrt(1 - SW2)
VEV = 2 * MW * math.sqrt(SW2) / EE
GEV2_TO_PB = 0.3893794e9


def build():
    """SM electroweak gauge sector + one massless lepton generation."""
    ew = electroweak_scaffold(gw=GW, g1=G1, v=VEV, mh=MH)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)
    Le = WeylFermion("Le", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nueL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("SM_ww", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)

    e = DiracParticle("e", Le.components[1], eR.components[0], 0)
    # the neutrino is left-handed only: right=None
    nu = DiracParticle("nu_e", Le.components[0], None, 0)
    calc = ScatteringCalculator(
        model, [e, nu],
        masses={ph.A: 0, ph.Z: MZ, ph.Wp: MW, ph.h: MH}, widths={ph.Z: WZ},
        boson_fields=[ph.Z, ph.A, ph.Wp, ph.Wm, ph.h],   # unitary gauge: no Goldstones
        conjugate_map=ph.cmap, conjugates={ph.Wp: ph.Wm, ph.Wm: ph.Wp},
        gauge_groups=[SU2L])                             # collect AWW / ZWW
    values = {ew.gw.s: GW, ew.g1.s: G1, ew.v.s: VEV}
    return calc, ph, e, values


def main():
    calc, ph, e, values = build()
    ee = (e.particle, e.antiparticle)

    print("=" * 68)
    print("e+e- -> W+W-  (feynlag.pheno, Tier 4)")
    print("=" * 68)
    ww = calc.process(ee, (ph.Wp, ph.Wm), numeric=True, extra=values)
    for d in ww.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator} ({d.kind} exchange)")
    sigma = ww.numeric_cross_section(200.0**2) * GEV2_TO_PB
    print(f"  sigma(sqrt s = 200 GeV) = {sigma:.3f} pb   "
          f"[MadGraph: 19.498 +- 0.058 pb]")

    kin = ww.kinematics
    nu_only = line_amplitude(
        ww.externals, [d for d in ww.diagrams if d.kind == "fermion"],
        kin).squared(kin)
    print("\n  Sum|M|^2 at cos(theta) = 0.3 — all three diagrams vs the "
          "neutrino exchange alone:")
    for rs in (200.0, 1e3, 1e4, 1e5):
        at = {kin.s: rs**2, kin.t: kin.t_of_cos(0.3).subs(kin.s, rs**2)}
        print(f"    sqrt s = {rs:>8.0f} GeV   full = {complex(ww.squared().subs(at)).real:10.4f}"
              f"   nu only = {complex(nu_only.subs(at)).real:14.4e}")
    print("  The full amplitude tends to a constant; the neutrino diagram "
          "grows like s^2.")

    print("\n" + "=" * 68)
    print("e+e- -> Z h  (Higgsstrahlung)")
    print("=" * 68)
    zh = calc.process(ee, (ph.Z, ph.h), numeric=True, extra=values)
    for d in zh.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator} ({d.kind} exchange)")
    for rs in (240.0, 250.0, 500.0):
        print(f"  sigma(sqrt s = {rs:.0f} GeV) = "
              f"{zh.numeric_cross_section(rs**2) * GEV2_TO_PB * 1e3:.1f} fb")


if __name__ == "__main__":
    main()

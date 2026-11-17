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

import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_inputs, electroweak_scaffold,
    fermion_gauge_current, to_physical_basis,
)
from feynlag.pheno import DiracParticle, ScatteringCalculator, line_amplitude

# MadGraph's stock (α, G_F, M_Z) input scheme (docs/benchmark.md): the
# couplings and VEV follow at tree level, and the model derives M_W, M_Z
# from them, so no boson mass is typed below.
INPUTS = electroweak_inputs(alpha_inv=132.50698, GF=1.16639e-5, MZ=91.1876)
WZ, MH = 2.4952, 125.0


def build():
    """SM electroweak gauge sector + one massless lepton generation."""
    ew = electroweak_scaffold(**INPUTS.couplings, mh=MH)
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
    # Everything else comes from the model: the unitary-gauge bosons (no
    # Goldstones), the W± conjugate pair, the masses read off the physical
    # Lagrangian, the broken SU(2) whose AWW/ZWW vertices are collected, and
    # the parameter point (so process() is numeric by default).
    # The bosons' MadGraph names (a, z, w+, w-, h) come with the physical
    # basis; the electron's are ours to choose.
    calc = ScatteringCalculator.from_model(
        model, [e, nu], widths={ph.Z: WZ},
        aliases={"e-": e.particle, "e+": e.antiparticle})
    return calc


def main():
    calc = build()

    print("=" * 68)
    print("e+e- -> W+W-  (feynlag.pheno, Tier 4)")
    print("=" * 68)
    ww = calc.process("e- e+ > w+ w-")
    print(ww.summary(sqrt_s=200.0, unit="pb"))
    print("  [MadGraph: 19.498 +- 0.058 pb]")

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
    zh = calc.process("e- e+ > z h")
    print(zh.summary())
    for rs in (240.0, 250.0, 500.0):
        print(f"  sigma(sqrt s = {rs:.0f} GeV) = "
              f"{zh.numeric_cross_section(sqrt_s=rs, unit='fb'):.1f} fb")


if __name__ == "__main__":
    main()

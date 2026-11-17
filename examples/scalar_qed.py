"""Scalar QED with a Yukawa: the ``VSS`` derivative coupling — Tier 4.

A U(1) gauge theory with an electron, a neutral left-handed fermion ``χ`` and
a charged scalar ``φ``.  The ``γφφ`` vertex comes out of ``|D_μφ|²`` carrying
momentum tags, ``i e (p(φ⁻) − p(φ⁺))``; the scattering engine resolves each
tag to the momentum of the diagram leg that field sits on.

Two processes:

- ``e⁺e⁻ → φ⁺φ⁻``.  With the Yukawa switched off it is the photon alone and
  reproduces the point-like scalar Born cross section ``πα²β³/(3s)``
  (Bystritskiy, Fedotovich, Ignatov, Kuraev, Phys. Rev. D 72, 114019 (2005),
  arXiv:hep-ph/0505236).  With it on, the ``χ`` exchange interferes.
- ``e⁻γ → χφ⁻``.  Electron exchange plus ``φ`` exchange through ``γφφ``.
  The photon's Ward identity ``k·M = 0`` holds only if the two diagrams have
  the right relative sign; the engine checks it before summing polarizations,
  so this process *is* the sign test of the derivative coupling.

The model has no VEV: it is unbroken, and its physical Lagrangian is the one
written.

Run:  python examples/scalar_qed.py
"""

import math

import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, Lagrangian, Model, Scalar, U1, WeylFermion,
    conjugate_pair, diracPR,
)
from feynlag.pheno import DiracParticle, ScatteringCalculator, cross_section_in

E_EM, Y_CHI, M_PHI = 0.303, 0.5, 30.0


def build():
    e_em = ExternalParameter("e_em", E_EM, positive=True)
    y = ExternalParameter("y_chi", Y_CHI, positive=True)
    m_phi = ExternalParameter("m_phi", M_PHI, positive=True)
    U1em = U1("U1em", coupling=e_em)

    phi = Scalar("phi", reps={U1em: -1}, component_names=["phim"])
    eL = WeylFermion("eL", reps={U1em: -1}, chirality="L", nflavors=1,
                     component_names=["eL"])
    eR = WeylFermion("eR", reps={U1em: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    chiL = WeylFermion("chiL", reps={U1em: 0}, chirality="L", nflavors=1,
                       component_names=["chiL"])
    A = U1em.bosons("A")

    i = sp.Symbol("i", integer=True)
    phim = phi.components[0]            # the field φ: annihilates φ⁻
    L = Lagrangian()
    L.add_kinetic(phi, eL, eR, chiL, index=i)
    L.add(-m_phi.s**2 * sp.conjugate(phim) * phim, sector="potential")
    yukawa = (-y.s * sp.conjugate(phim)
              * Bilinear(chiL.bar_components[0][i], diracPR, eR.components[0][i]))
    L.add(yukawa + sp.conjugate(yukawa), sector="yukawa")

    model = Model("ScalarQED", gauge_groups=[U1em],
                  fields=[phi, eL, eR, chiL, A],
                  parameters=[e_em, y, m_phi], lagrangian=L)
    model.check_invariance().raise_on_failure()

    phip, cmap = conjugate_pair(phim, "phip")       # φ† as its own symbol
    photon = A.components[0]
    e = DiracParticle("e", eL.components[0], eR.components[0], 0)
    chi = DiracParticle("chi", chiL.components[0], None, 0)
    calc = ScatteringCalculator(
        model, [e, chi], masses={photon: 0, phim: m_phi.s},
        boson_fields=[photon, phim, phip], fermion_sectors=("yukawa",),
        conjugate_map=cmap, conjugates={phim: phip, phip: phim})
    values = {e_em.s: E_EM, y.s: Y_CHI, m_phi.s: M_PHI}
    return calc, dict(e=e, chi=chi, A=photon, phim=phim, phip=phip, y=y), values


def main():
    calc, p, values = build()
    e, chi = p["e"], p["chi"]

    print("=" * 68)
    print("Three-leg vertices")
    print("=" * 68)
    for vertex in calc.vertices():
        print("  ", vertex)

    print("\n" + "=" * 68)
    print("e+e- -> phi+ phi-  (photon only: Yukawa off)")
    print("=" * 68)
    pair = calc.process((e.particle, e.antiparticle), (p["phip"], p["phim"]),
                        numeric=True, extra={**values, p["y"].s: 0})
    alpha = E_EM**2 / (4 * math.pi)
    for rs in (70.0, 200.0, 1000.0):
        s = rs**2
        beta = math.sqrt(1 - 4 * M_PHI**2 / s)
        closed = math.pi * alpha**2 * beta**3 / (3 * s)
        sigma = pair.numeric_cross_section(s, unit="pb")
        print(f"  sqrt s = {rs:>6.0f} GeV   sigma = {sigma:10.5f} pb"
              f"   pi alpha^2 beta^3/(3s) = {cross_section_in(closed, 'pb'):10.5f} pb")

    print("\n" + "=" * 68)
    print("e+e- -> phi+ phi-  (photon + chi exchange)")
    print("=" * 68)
    full = calc.process((e.particle, e.antiparticle), (p["phip"], p["phim"]),
                        numeric=True, extra=values)
    for d in full.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator} ({d.kind} exchange)")
    print(f"  sigma(sqrt s = 200 GeV) = "
          f"{full.numeric_cross_section(sqrt_s=200.0, unit='pb'):.4f} pb")

    print("\n" + "=" * 68)
    print("e- gamma -> chi phi-  (the Ward identity is checked)")
    print("=" * 68)
    photo = calc.process((e.particle, p["A"]), (chi.particle, p["phim"]),
                         numeric=True, extra=values)
    for d in photo.diagrams:
        print(f"  diagram: {d.channel}-channel {d.mediator} ({d.kind} exchange)")
    photo.squared()         # raises if k.M != 0 for the photon
    print(f"  sigma(sqrt s = 200 GeV) = "
          f"{photo.numeric_cross_section(sqrt_s=200.0, unit='pb'):.4f} pb")


if __name__ == "__main__":
    main()

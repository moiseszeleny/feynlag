"""SM + type-I seesaw with feynlag, end to end.

The Standard Model extended by right-handed neutrinos ``ν_R`` (total gauge
singlets) with a Dirac Yukawa and a large right-handed Majorana mass:

    L ⊃ −Y_ν L̄ H̃ ν_R  −  ½ M_R ν_Rᵀ C ν_R  +  h.c.

After EWSB the Dirac mass is ``m_D = Y_ν v/√2``.  In the left-handed basis
``n = (ν_L, ν_R^c)`` the neutral-fermion Majorana mass matrix is

    M_ν = [[ 0 , m_D ], [ m_Dᵀ, M_R ]]   (Takagi-diagonalised).

For ``M_R ≫ m_D`` this is the **seesaw**: a naturally tiny light neutrino
``m_ν ≈ −m_D M_R⁻¹ m_Dᵀ`` and a heavy state ``≈ M_R``, mixed by
``V ≈ m_D M_R⁻¹``.  The light–heavy mixing ``V`` controls every heavy-neutrino
coupling, extracted below through feynlag's vertex machinery via the
``MajoranaRotation`` (the physical neutrinos are Majorana, mixing ``ν_L`` with
the charge-conjugate ``ν_R^c``).

Headline physics (pinned in tests/test_seesaw.py):

- ``m_ν = −m_D²/M_R`` (1 generation), matching the exact Takagi light mass;
- the heavy-neutrino production coupling ``W ℓ̄ N = (g/√2)·V``, with
  ``V ≈ m_D/M_R`` — vanishing as ``M_R → ∞`` (decoupling);
- the neutral current ``Z ν̄ N ∝ (g_Z/2)·V``.

Run:  python examples/sm_seesaw.py
"""

import sympy as sp

from feynlag import (
    FL_I, FL_J, DiracGamma, Lagrangian, MajoranaRotation, Model,
    ParameterSet, WeylFermion, diagonalize_takagi, diracC, diracPL, diracPR,
    electroweak_scaffold, extract_fermion_vertices, fermion_mass_matrix,
    majorana_mass, majorana_mass_matrix, seesaw_light_mass,
    seesaw_mass_matrix, tilde, yukawa,
)


def _num(x):
    return complex(sp.N(x))


def main():
    # --- symmetries, parameters, fields (1 generation) ------------------
    ew = electroweak_scaffold(lam=0.129)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    gw, g1, v, mu2 = ew.gw, ew.g1, ew.v, ew.mu2
    P = ParameterSet(*ew.parameters)
    yv = P.external("yv", 0.01, positive=True)                  # Dirac Yukawa
    MR = P.external("MR", 1.0e3, positive=True, unit_dim=1)     # seesaw scale

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=1,
                      component_names=["nuR"])

    # field level: the constructors do the H̃ = iσ₂H* contraction, the
    # flavour index and the + h.c.
    LYukD = yukawa(yv, Ll, tilde(H), nuR)     # −Y_ν L̄ H̃ ν_R + h.c.
    LMaj = majorana_mass(MR, nuR)             # −½ M_R ν_Rᵀ C ν_R + h.c.

    L = Lagrangian()
    ew.add_higgs(L)                                    # kinetic + potential
    L.add(LYukD, sector="yukawa")
    L.add_kinetic(Ll, eR)                              # the gauge currents
    L.add(LMaj, sector="other")     # kept separate: the Majorana mass, not a vertex

    model = Model("SM-seesaw", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, eR, nuR],
                  parameters=P, lagrangian=L)

    print("invariance:", model.check_invariance())
    model.check_invariance().raise_on_failure()
    print("tadpole:   ", model.solve_tadpoles([mu2]))

    # gauge rotations: the model finds the blocks (W¹,W² → W±; W³,B → Z,γ)
    charged, neutral = model.mass_blocks(kind="gauge")
    Z, A = model.diagonalize(neutral, names=["Z", "A"]).new_fields
    Wp, Wm = model.diagonalize(charged, names=["Wp"],
                               conjugate_names=["Wm"]).rotation.new_fields

    i, j = FL_I, FL_J
    nuL, nuLbar, eLbar = Ll.nuL, Ll.nuLbar, Ll.eLbar
    nR, nRbar = nuR.nuR, nuR.nuRbar
    CPL = diracC * diracPL

    # --- the seesaw mass matrix -----------------------------------------
    mD = fermion_mass_matrix(LYukD, nuLbar, nR, model.vacuum, 1, (i, j),
                             gamma=diracPR)
    MRmat = majorana_mass_matrix(LMaj, nR, model.vacuum, 1, (i, j), gamma=CPL)
    Mnu = seesaw_mass_matrix(mD, MRmat)
    print("\nseesaw mass matrix M_ν (basis ν_L, ν_R^c):")
    sp.pprint(Mnu)
    print("seesaw light mass  m_ν = −m_D²/M_R =",
          sp.simplify(seesaw_light_mass(mD, MRmat)[0, 0]))

    # --- Takagi diagonalisation at a benchmark --------------------------
    bench = {yv.s: 0.01, v.s: 246.0, MR.s: 1000.0, gw.s: 0.6535, g1.s: 0.3580}
    Mn = sp.Matrix(2, 2, lambda a, b: sp.nsimplify(Mnu[a, b].subs(bench)))
    U, D = diagonalize_takagi(Mn)
    masses = [abs(_num(D[k, k])) for k in range(2)]
    light = 0 if masses[0] < masses[1] else 1
    heavy = 1 - light
    mixing = abs(_num(U.conjugate()[0, heavy]))
    print(f"\nbenchmark (y_ν=0.01, v=246, M_R=1000 GeV):")
    print(f"   m_light = {masses[light]:.4e} GeV,  m_heavy = {masses[heavy]:.1f} GeV")
    print(f"   light–heavy mixing V = |U*[0,heavy]| = {mixing:.5f}  (≈ m_D/M_R)")

    # --- rotate the weak neutrinos → physical Majorana χ, extract couplings
    chiL, chiR = sp.IndexedBase("chiL"), sp.IndexedBase("chiR")
    chiLbar, chiRbar = sp.IndexedBase("chiLbar"), sp.IndexedBase("chiRbar")
    rot = MajoranaRotation(U, nuL, nR, nuLbar, nRbar,
                           chiL, chiR, chiLbar, chiRbar, n_L=1)
    Lchi = rot.apply(model.physical_lagrangian(sector="yukawa"), (i, j), 1)
    tab = extract_fermion_vertices(Lchi, [Wp, Wm, Z, A])

    mu = sp.Symbol("mu", integer=True)
    gL = DiracGamma(mu) * diracPL
    gsq = float((gw.s / sp.sqrt(2)).subs(bench))
    gz = float(sp.sqrt(gw.s**2 + g1.s**2).subs(bench))

    def cval(c):
        return abs(_num(sp.simplify(c).subs(bench))) if c != 0 else 0.0

    print("\nheavy-neutrino couplings (feynlag-extracted, physical basis):")
    for k, tag in ((light, "light ν"), (heavy, "heavy N")):
        cW = cval(tab.get((eLbar[0], gL, chiL[k]), {}).get(1, {}).get((Wm,), 0))
        print(f"   W⁻ e {tag}:  |coupling|/(g/√2) = {cW / gsq:.5f}")
    for k, tag in ((light, "light ν"), (heavy, "heavy N")):
        cZ = cval(tab.get((chiLbar[light], gL, chiL[k]), {}).get(1, {}).get((Z,), 0))
        print(f"   Z ν {tag}:   |coupling|/g_Z    = {cZ / gz:.5f}")
    print("→ the heavy-N couplings are suppressed by the mixing V ≈ m_D/M_R;")
    print("  they vanish as M_R → ∞ (the seesaw decoupling).")


if __name__ == "__main__":
    main()

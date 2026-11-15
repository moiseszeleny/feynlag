"""SM + the dim-5 Weinberg operator: Majorana neutrino masses, end to end.

The Weinberg operator is the unique dimension-5 gauge invariant you can add to
the Standard Model,

    L ⊃ (c_ij / Λ) (L_iᵀ C ε L_j)(Hᵀ ε H) + h.c.,

and it is *the* effective operator behind neutrino mass: after EWSB it becomes a
**Majorana** mass ``m_ν = −c v²/Λ`` for the left-handed neutrinos (a symmetric
matrix, diagonalized by Takagi factorization), naturally tiny because it is
suppressed by the high scale Λ (the seesaw).

This exercises feynlag's Majorana machinery (Phase D.2):

- the charge-conjugation matrix ``diracC`` and the ``MajoranaBilinear`` atom
  ``ψ₁ᵀ C Γ ψ₂`` (same-chirality, unlike the Dirac ``Bilinear``);
- ``suggest_yukawa(..., max_dim=5)``, which *enumerates* the Weinberg operator
  once you allow dimension-5 terms;
- ``majorana_mass_matrix`` + ``diagonalize_takagi`` for the physical masses;
- ``extract_majorana_vertices`` for the ``ν̄νh`` / ``ν̄νhh`` couplings.

UFO export of Majorana vertices is not yet supported (MadGraph Majorana-fermion
export is a separate follow-up), so no UFO directory is written here.

Run:  python examples/sm_weinberg.py
"""

import sympy as sp

from feynlag import (
    FL_I, FL_J, Lagrangian, Model, ParameterSet, Scalar, SU2, U1,
    WeylFermion, diagonalize_takagi, diracC, diracPL,
    extract_majorana_vertices, inner, majorana_feynman_rule,
    majorana_mass_matrix, suggest_yukawa, weinberg,
)


def main():
    # --- symmetries, parameters, fields ---------------------------------
    P = ParameterSet()
    gw = P.external("gw", 0.6535, positive=True)
    g1 = P.external("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = P.external("v", 246.0, positive=True, unit_dim=1)
    Lam = P.external("Lam", 1.0e14, positive=True, unit_dim=1)  # seesaw scale
    lam = P.external("lam", 0.129)
    mu2 = P.internal("mu2", unit_dim=2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.H0: v})
    # two lepton generations (enough to show the mixing / Takagi)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=2, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=2,
                     component_names=["eR"])

    # --- let `suggest` enumerate the dim-5 operator basis ----------------
    print("suggest_yukawa(max_dim=4):")
    for t in suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=4):
        print("   ", t)
    print("suggest_yukawa(max_dim=5) adds the Weinberg operator:")
    for t in suggest_yukawa([Ll, eR], [H], [SU2L, U1Y], max_dim=5):
        print("   ", t)

    # --- the Weinberg operator at field level (symmetric c_ij) ---------
    # weinberg() does the (L ε H)(L ε H) contraction, the flavour indices
    # (c[fl_i, fl_j]) and the + h.c.; it enters with +c.  The component-level
    # form it expands to is pinned in tests/test_terms.py.
    c = sp.IndexedBase("c")            # symmetric Weinberg Wilson coefficient
    i, j = FL_I, FL_J
    O = weinberg(c, Ll, H) / Lam.s     # c/Λ (L ε H)(L ε H) + h.c.

    HdH = inner(H, H)
    V = -mu2 * HdH + lam * HdH**2
    L = Lagrangian()
    L.add_kinetic(H)
    L.add(-V, sector="potential")
    L.add(O, sector="other")

    model = Model("SM-Weinberg", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=P, lagrangian=L)

    # which allowed operators up to dim 5 does this Lagrangian contain?  The
    # Weinberg operator is matched; the charged-lepton Yukawa is reported
    # missing (this example leaves e_R massless on purpose).
    cov = model.term_coverage(max_dim=5)
    print("\nterm coverage (max_dim=5):", cov)

    # --- validate (dim-5 needs max_dim=5) --------------------------------
    report = model.check_invariance(max_dim=5)
    print("\ninvariance:", report)
    report.raise_on_failure()
    print("tadpole:", model.solve_tadpoles([mu2]))

    # --- Majorana neutrino mass + Takagi diagonalization -----------------
    nuL, CPL = Ll.nuL, diracC * diracPL
    Mnu = majorana_mass_matrix(O, nuL, model.vacuum, 2, (i, j), gamma=CPL)
    print("\nMajorana mass matrix  m_ν = −c v²/Λ:")
    sp.pprint(Mnu)

    bench = {c[0, 0]: 0.8, c[0, 1]: 0.3, c[1, 0]: 0.3, c[1, 1]: 1.0,
             v.s: 246.0, Lam.s: 1.0e14}
    Mnum = sp.Matrix(2, 2, lambda a, b: sp.nsimplify(Mnu[a, b].subs(bench)))
    U, D = diagonalize_takagi(Mnum)
    masses = [complex(D[k, k]).real for k in range(2)]
    print(f"\nphysical Majorana masses at Λ=10¹⁴ GeV, v=246 GeV:")
    print(f"   m_ν1 = {masses[0]:.3e} GeV,  m_ν2 = {masses[1]:.3e} GeV "
          f"(sub-eV, the seesaw)")
    ok = sp.simplify(U * D * U.T - Mnum) == sp.zeros(2, 2)
    print("   Takagi check  M = U D Uᵀ:", ok)

    # --- ν̄νh coupling ----------------------------------------------------
    Lphys = sp.expand(model.physical_lagrangian(sector="other"))
    h, G0 = H.H0_r, H.H0_i
    table = extract_majorana_vertices(Lphys, [h, G0])
    key = (nuL[i], CPL, nuL[j])
    c_h = table.get(key, {}).get(1, {}).get((h,), 0)
    rule = sp.simplify(majorana_feynman_rule(c_h, CPL, (h,)))
    print("\nν̄_i ν_j h coupling =", rule, " (∝ c v/Λ — the seesaw-suppressed "
          "Higgs–neutrino coupling)")


if __name__ == "__main__":
    main()

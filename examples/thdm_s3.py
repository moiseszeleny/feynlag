"""3HDM with S₃ symmetry: the feynlag stress test, end to end.

(H1, H2) form an S₃ doublet, HS the singlet.  The potential is built from
the library's own Clebsch–Gordan products, checked for gauge AND S₃
invariance, and the tadpole conditions are shown to force the √3 vacuum
alignment (v₁ = √3 v₂ in the literature basis; components swap in feynlag's
real-orthogonal irrep basis).

Run:  python examples/thdm_s3.py
"""

import sympy as sp

from feynlag import (
    Lagrangian, Model, ParameterSet, S3, SU2, Scalar, U1, inner,
)


def main():
    P = ParameterSet()
    gw = P.external("gw", 0.6535, positive=True)
    g1 = P.external("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    s3 = S3()

    v1 = P.external("v1", 200.0, positive=True, unit_dim=1)
    v2 = P.external("v2", 115.0, positive=True, unit_dim=1)
    vS = P.external("vS", 80.0, positive=True, unit_dim=1)
    lams = {k: P.external(f"lm{k}", 0.05 * k) for k in range(1, 9)}
    mu0sq = P.internal("mu0sq", unit_dim=2)
    mu1sq = P.internal("mu1sq", unit_dim=2)

    def doublet(name):
        return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                      component_names=[f"{name}p", f"{name}0"])

    H1, H2, HS = doublet("H1"), doublet("H2"), doublet("HS")
    s3.assign("2", H1, H2)
    s3.assign("1", HS)
    H1.expand_vev({H1.H10: v1})
    H2.expand_vev({H2.H20: v2})
    HS.expand_vev({HS.HS0: vS})

    # the SU(2)×U(1) invariants A†B; the S3 structure is built from them
    x11, x22 = inner(H1, H1), inner(H2, H2)
    x12, x21 = inner(H1, H2), inner(H2, H1)
    s1, s2 = inner(HS, H1), inner(HS, H2)
    sss = inner(HS, HS)

    # S3 CG contractions (2⊗2 = 1 ⊕ 1' ⊕ 2 in the real-orthogonal basis)
    inv1 = x11 + x22
    inv1p = x12 - x21
    d2 = (x11 - x22, -(x12 + x21))

    lam4_term = s1 * d2[0] + s2 * d2[1]
    lam4_term += sp.conjugate(lam4_term)
    lam7_term = s1**2 + s2**2
    lam7_term += sp.conjugate(lam7_term)

    l = lams
    V = (mu1sq * inv1 + mu0sq * sss
         + l[1] * inv1**2 + l[2] * inv1p**2
         + l[3] * (d2[0]**2 + d2[1]**2)
         + l[4] * lam4_term
         + l[5] * sss * inv1
         + l[6] * (s1 * inner(H1, HS) + s2 * inner(H2, HS))
         + l[7] * lam7_term
         + l[8] * sss**2)

    L = Lagrangian().add(-V, sector="potential")
    model = Model("3HDM-S3", gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS],
                  parameters=P, lagrangian=L)

    print("invariance (gauge + S3):", model.check_invariance())

    # --- vacuum alignment from the tadpoles ------------------------------
    tadpoles = model.tadpoles()
    sol = sp.solve([sp.Eq(tadpoles[v2.s], 0), sp.Eq(tadpoles[vS.s], 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]
    residual = sp.factor(sp.expand(tadpoles[v1.s].subs(sol)))
    print("\nthird tadpole after solving the other two:")
    print("  ", residual, "= 0")
    print("solutions for v1:", sp.solve(sp.Eq(residual, 0), v1.s),
          " (the S3 √3 alignment)")

    # --- CP-even mass matrix on the aligned vacuum -----------------------
    align = {v1.s: v2.s / sp.sqrt(3)}
    sol = sp.solve([sp.Eq(tadpoles[v2.s].subs(align), 0),
                    sp.Eq(tadpoles[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]
    h = [H1.H10_r, H2.H20_r, HS.HS0_r]
    M = model.mass_matrix(h).subs(sol).subs(align)
    M = M.applyfunc(lambda e: sp.simplify(sp.expand(e)))
    print("\nCP-even mass matrix (aligned vacuum):")
    sp.pprint(M)


if __name__ == "__main__":
    main()

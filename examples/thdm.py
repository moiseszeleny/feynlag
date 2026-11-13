"""Softly-broken Z2 2HDM scalar sector with feynlag.

The potential is written at field level (``inner``), and the physical basis is
*found* by the model: ``mass_blocks`` reports which fluctuations mix, with
their electric charge (derived from the vacuum) and their Goldstone directions
(derived from the broken generators), and ``diagonalize`` picks the rotation
for each block.  The user only names the states.

- CP-even block: no Goldstone, the ``tan 2α`` route.
- CP-odd and charged blocks: one Goldstone each, so the rotation is fixed
  exactly by the Goldstone vector — ``tan β = v2/v1`` comes out, it is not
  put in.

Run:  python examples/thdm.py
"""

import sympy as sp

from feynlag import (
    Lagrangian, Model, ParameterSet, SU2, Scalar, U1, ZN, hc, inner,
)


def main():
    P = ParameterSet()
    gw = P.external("gw", 0.6535, positive=True)
    g1 = P.external("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v1 = P.external("v1", 246 * 0.95, positive=True, unit_dim=1)
    v2 = P.external("v2", 246 * 0.31, positive=True, unit_dim=1)
    m12sq = P.external("m12sq", 2000.0, unit_dim=2)
    l1, l2, l3, l4, l5 = (P.external(f"lam{i}", 0.1 * i) for i in range(1, 6))
    m11sq = P.internal("m11sq", unit_dim=2)
    m22sq = P.internal("m22sq", unit_dim=2)

    H1 = Scalar("H1", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                component_names=["H2p", "H20"])
    H1.expand_vev({H1.H10: v1})
    H2.expand_vev({H2.H20: v2})

    # soft Z2 (bookkeeping: broken only by m12²)
    Z2 = ZN("Z2", 2)
    Z2.assign(0, H1)
    Z2.assign(1, H2)

    a, b, x = inner(H1, H1), inner(H2, H2), inner(H1, H2)
    V = (m11sq * a + m22sq * b - m12sq * (x + hc(x))
         + l1 / 2 * a**2 + l2 / 2 * b**2 + l3 * a * b + l4 * x * hc(x)
         + l5 / 2 * (x**2 + hc(x)**2))

    L = Lagrangian().add(-V, sector="potential")
    model = Model("THDM", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                  parameters=P, lagrangian=L)

    print("invariance:", model.check_invariance())
    print("tadpoles solved for m11², m22²:")
    for k, s in model.solve_tadpoles([m11sq, m22sq]).items():
        print("  ", k, "=", s)

    # --- the model finds its own mass blocks ---------------------------
    print("\nmass blocks:")
    blocks = model.mass_blocks()
    for block in blocks:
        print("  ", block)
    even, odd, charged = blocks

    print("\nCP-even M²:")
    sp.pprint(even.matrix)

    # --- and the rotation each one needs --------------------------------
    d_even = model.diagonalize(even, names=["Hheavy", "hlight"], angle="alpha")
    print("\ntan(2α) =", sp.simplify(d_even.angle_relation.rhs))

    d_odd = model.diagonalize(odd, names=["G0", "A0"], angle="beta")
    d_ch = model.diagonalize(charged, names=["Gp", "Hp"],
                             conjugate_names=["Gm", "Hm"])
    print("\nGoldstone-fixed rotation:", d_odd.angle_relation)
    print("m_A²  =", sp.factor(d_odd.masses_squared[1]))
    print("m_H±² =", sp.factor(d_ch.masses_squared[1]))

    phys = model.physical
    print("\nphysical scalars:", phys.bosons)
    print("Goldstones:      ", phys.goldstones)
    print("charges:         ", phys.charges)

    Hh, hl = d_even.new_fields
    rules = model.feynman_rules([Hh, hl], simplifier=sp.simplify)
    print(f"\n{len(rules)} neutral-CP-even vertices, e.g.:")
    for key in list(rules)[:4]:
        print(" ", key, "->", sp.simplify(rules[key]))


if __name__ == "__main__":
    main()

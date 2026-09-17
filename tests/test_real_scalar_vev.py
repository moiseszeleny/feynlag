"""Mass matrices with a real VEV'd scalar (SM + Z2 real singlet).

For ``Scalar(real=True)`` the fluctuation symbol *is* the component
(``S → v_S + S``), so shifting twice evaluates at ``S = 2 v_S``. Pinned:
``Model.mass_matrix`` shifts exactly once.

- CP-even block [[2λv², λ_HS v v_S], [λ_HS v v_S, 2λ_S v_S²]]
- G0 and G± massless
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, SU2, Scalar, U1,
    build_mass_matrix, dag, numeric_equal,
)


@pytest.fixture
def xsm():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)

    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    vS = ExternalParameter("vS", 100.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    lamS = ExternalParameter("lamS", 0.2)
    lamHS = ExternalParameter("lamHS", 0.1)
    mu2 = InternalParameter("mu2", unit_dim=2)
    muS2 = InternalParameter("muS2", unit_dim=2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    S = Scalar("S", reps={}, component_names=["S"], real=True)
    s0 = S.components[0]
    S.expand_vev({s0: vS})

    HdH = (dag(H) * H.mat)[0]
    V = (-mu2.s * HdH + lam.s * HdH**2 + muS2.s / 2 * s0**2
         + lamS.s / 4 * s0**4 + lamHS.s / 2 * HdH * s0**2)
    L = Lagrangian().add(-V, sector="potential")
    model = Model("xSM", gauge_groups=[SU2L, U1Y], fields=[H, S],
                  parameters=[gw, g1, v, vS, lam, lamS, lamHS, mu2, muS2],
                  lagrangian=L)
    model.solve_tadpoles([mu2, muS2])
    syms = dict(v=v.s, vS=vS.s, lam=lam.s, lamS=lamS.s, lamHS=lamHS.s)
    return model, H, s0, V, syms


def _single_shift_reference(model, V, fields):
    """Independent route: shift once, differentiate, zero fluctuations."""
    vac = model.vacuum
    M = build_mass_matrix(vac.shift(V), fields)
    zero = {f: 0 for f in vac.fluctuations}
    zero[model.fields[0].components[0]] = 0
    zero[sp.conjugate(model.fields[0].components[0])] = 0
    M = M.applyfunc(lambda e: sp.expand(e.xreplace(zero)))
    return M.applyfunc(lambda e: sp.expand(e.subs(model._tadpole_solutions)))


def test_cp_even_block_single_shift(xsm):
    model, H, s0, V, p = xsm
    h = sp.Symbol("H0_r", real=True)
    M = model.mass_matrix([h, s0])
    expected = sp.Matrix([
        [2 * p["lam"] * p["v"]**2, p["lamHS"] * p["v"] * p["vS"]],
        [p["lamHS"] * p["v"] * p["vS"], 2 * p["lamS"] * p["vS"]**2],
    ])
    assert sp.simplify(M - expected) == sp.zeros(2, 2)

    ref = _single_shift_reference(model, V, [h, s0])
    syms = sorted(p.values(), key=str)
    for i in range(2):
        for j in range(2):
            ok, _ = numeric_equal(M[i, j], ref[i, j], syms, seed=i + j)
            assert ok


def test_goldstones_massless(xsm):
    model, H, s0, V, p = xsm
    G0 = sp.Symbol("H0_i", real=True)
    assert sp.simplify(model.mass_matrix([G0])[0, 0]) == 0
    Gp = H.components[0]
    assert sp.simplify(model.mass_matrix([Gp], charged=True)[0, 0]) == 0


def test_at_vacuum_is_zero_fluctuations_after_shift(xsm):
    model, H, s0, V, p = xsm
    vac = model.vacuum
    assert sp.expand(vac.at_vacuum(V)
                     - vac.zero_fluctuations(vac.shift(V))) == 0
    # the pitfall: at_vacuum on a shifted expression re-shifts a real scalar
    assert vac.at_vacuum(vac.shift(s0)) == 2 * p["vS"]
    assert vac.zero_fluctuations(vac.shift(s0)) == p["vS"]

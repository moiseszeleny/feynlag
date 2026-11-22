"""Guided physical basis: ``Model.mass_blocks`` / ``Model.diagonalize``.

The blocks, Goldstone directions and rotations are *found* by the model; the
tests pin them against the hand-built results of the existing pipeline tests
(``test_scalar_pipeline_thdm.py``, ``test_vll.py``) and the SM values.
"""

import pytest
import sympy as sp

from feynlag import (
    FL_I, InternalParameter, Lagrangian, Model, ParameterSet, Rotation, SU2,
    Scalar, U1, WeylFermion, diagonalize_orthogonal_2x2, diagonalize_svd_2x2,
    dirac_mass, electroweak_scaffold, hc, inner, numeric_equal, rotation_2x2,
    standard_model, yukawa,
)
from feynlag import (
    SU3, charged_current_rotation, weinberg_rotation,
)
from feynlag.vacuum.physical import goldstone_directions

half = sp.Rational(1, 2)


def _ew(P):
    gw = P.external("gw", 0.6535, positive=True)
    g1 = P.external("g1", 0.3580, positive=True)
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)


def _thdm(solve=True, complex_lam5=False):
    """Softly-broken-Z2 2HDM (the potential of examples/thdm.py).  With
    ``complex_lam5`` the potential violates CP explicitly: Im(m12²) is fixed
    by the third stationarity condition, Im(m12²) = Im(λ5) v1 v2 / 2."""
    P = ParameterSet()
    SU2L, U1Y = _ew(P)
    v1 = P.external("v1", 233.7, positive=True, unit_dim=1)
    v2 = P.external("v2", 76.3, positive=True, unit_dim=1)
    m12sq = P.external("m12sq", 2.0e4, unit_dim=2)
    lam = [P.external(f"lam{k}", 0.1 * k + 0.2) for k in range(1, 6)]
    m11sq = P.internal("m11sq", unit_dim=2)
    m22sq = P.internal("m22sq", unit_dim=2)
    H1 = Scalar("H1", reps={SU2L: 2, U1Y: half}, component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: half}, component_names=["H2p", "H20"])
    H1.expand_vev({H1.H10: v1})
    H2.expand_vev({H2.H20: v2})
    l1, l2, l3, l4, l5 = (q.s for q in lam)
    a, b, x = inner(H1, H1), inner(H2, H2), inner(H1, H2)
    m12, lam5 = m12sq.s, l5
    if complex_lam5:
        l5i = P.external("lam5i", 0.15)
        lam5 = l5 + sp.I * l5i.s
        m12 = m12sq.s + sp.I * l5i.s * v1.s * v2.s / 2
    V = (m11sq.s * a + m22sq.s * b - (m12 * x + sp.conjugate(m12) * hc(x))
         + l1 / 2 * a**2 + l2 / 2 * b**2 + l3 * a * b + l4 * x * hc(x)
         + (lam5 * x**2 + sp.conjugate(lam5) * hc(x)**2) / 2)
    L = Lagrangian().add(-V, sector="potential")
    model = Model("THDM", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                  parameters=P, lagrangian=L)
    if solve:
        model.solve_tadpoles([m11sq, m22sq])
    return model, H1, H2, P


def _values(P):
    return {p.s: p.value for p in P.externals}


# ------------------------------------------------------------------- SM

def test_sm_blocks_charges_and_goldstones():
    ew = electroweak_scaffold(lam=0.129)
    L = Lagrangian()
    ew.add_higgs(L)
    model = Model("SM", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ew.parameters, lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    H = ew.H
    even, odd, charged = model.mass_blocks(kind="scalar")
    assert (even.fields, even.cp, even.charge) == ([H.H0_r], "even", 0)
    assert (odd.fields, odd.cp, len(odd.goldstones)) == ([H.H0_i], "odd", 1)
    assert (charged.fields, charged.charge, charged.complex) == ([H.Gp], 1, True)
    assert len(even.goldstones) == 0 and len(charged.goldstones) == 1
    assert "cp-even" in repr(even) and "goldstones: 1" in repr(charged)

    d = model.diagonalize(even)                       # 1×1: nothing to rotate
    assert d.rotations == [] and d.new_fields == [H.H0_r]
    assert sp.simplify(d.masses_squared[0] - 2 * ew.lam.s * ew.v.s**2) == 0
    model.diagonalize(odd)
    dc = model.diagonalize(charged, conjugate_names=["Gm"])
    assert d.goldstones == [] and dc.goldstones == [H.Gp]

    phys = model.physical
    Gm = phys.conjugates[H.Gp]
    assert str(Gm) == "Gm" and phys.cmap == {sp.conjugate(H.Gp): Gm}
    assert phys.bosons == [H.H0_r, H.H0_i, H.Gp, Gm]
    assert phys.goldstones == [H.H0_i, H.Gp, Gm]
    assert phys.charges == {H.H0_r: 0, H.H0_i: 0, H.Gp: 1, Gm: -1}
    assert phys.masses[H.H0_i] == 0 and phys.masses[H.Gp] == 0

    # a 1×1 block can also be renamed
    model2 = standard_model(generations=1, physical_basis=False).model
    model2.solve_tadpoles([p for p in model2.parameters if p.name == "mu2"])
    d2 = model2.diagonalize(model2.mass_blocks(kind="scalar")[0], names=["h"])
    assert str(d2.new_fields[0]) == "h" and len(d2.rotations) == 1
    assert d2.new_fields[0] in model2.physical_lagrangian("potential").free_symbols


def test_sm_helpers_fill_the_registry():
    """to_physical_basis records the same registry the guided route builds."""
    sm = standard_model(generations=1)
    phys, pb = sm.model.physical, sm.physical
    assert phys.bosons == [pb.Z, pb.A, pb.Wp, pb.Wm, pb.h, pb.G0, pb.Gp, pb.Gm]
    assert phys.goldstones == [pb.G0, pb.Gp, pb.Gm]
    assert phys.cmap == pb.cmap
    assert phys.conjugates[pb.Wp] == pb.Wm and phys.conjugates[pb.Gm] == pb.Gp
    assert phys.charges == {pb.Z: 0, pb.A: 0, pb.Wp: 1, pb.Wm: -1, pb.h: 0,
                            pb.G0: 0, pb.Gp: 1, pb.Gm: -1}


def test_goldstone_directions_of_the_sm_vacuum():
    """i T^a <H> for the four generators: three independent broken directions
    (G0 and the complex G+), and Q = T3 + Y annihilates the vacuum."""
    ew = electroweak_scaffold()
    model = Model("SM", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ew.parameters)
    H, v = ew.H, ew.v.s
    dirs = goldstone_directions(model)
    assert len(dirs) == 4                       # T1, T2, T3, Y all move <H>
    t1, t2, t3, y = dirs
    assert t1 == {H.Gp: sp.I * v / (2 * sp.sqrt(2))}
    assert t2 == {H.Gp: v / (2 * sp.sqrt(2))}
    assert t3 == {H.H0_i: -v / 2} and y == {H.H0_i: v / 2}   # T3 + Y: zero


# ----------------------------------------------------------------- 2HDM

def test_thdm_blocks_are_found_without_hints():
    model, H1, H2, P = _thdm()
    even, odd, charged = model.mass_blocks()
    assert even.fields == [H1.H10_r, H2.H20_r] and even.cp == "even"
    assert odd.fields == [H1.H10_i, H2.H20_i] and odd.cp == "odd"
    assert charged.fields == [H1.H1p, H2.H2p] and charged.charge == 1
    assert [len(b.goldstones) for b in (even, odd, charged)] == [0, 1, 1]

    # the Goldstone vector is (v1, v2)/v — derived from T^a<φ>, not assumed
    v1, v2 = P["v1"].s, P["v2"].s
    g = sp.Matrix([v1, v2]) / sp.sqrt(v1**2 + v2**2)
    for block in (odd, charged):
        assert (block.goldstones[0] - g).applyfunc(sp.simplify) == sp.zeros(2, 1)

    # the matrices are the ones Model.mass_matrix gives for those fields
    assert even.matrix == model.mass_matrix(even.fields)
    assert charged.matrix == model.mass_matrix(charged.fields, charged=True)


def test_thdm_goldstone_fixed_rotations_and_masses():
    model, H1, H2, P = _thdm()
    v1, v2, m12sq = P["v1"].s, P["v2"].s, P["m12sq"].s
    lam4, lam5 = P["lam4"].s, P["lam5"].s
    _, odd, charged = model.mass_blocks()

    d_odd = model.diagonalize(odd, names=["G0", "A0"], angle="beta")
    beta = sp.Symbol("beta", real=True)
    assert d_odd.rotation.matrix == rotation_2x2(beta)
    assert d_odd.angle_relation == sp.Eq(sp.tan(beta), v2 / v1)
    assert "beta" in model.parameters                  # registered, defined
    assert model.parameters["beta"].expr == sp.atan(v2 / v1)
    mA2 = (m12sq / (v1 * v2) - lam5) * (v1**2 + v2**2)
    assert d_odd.masses_squared[0] == 0
    assert sp.simplify(d_odd.masses_squared[1] - mA2) == 0
    ok, diff = numeric_equal(d_odd.masses_squared[1], mA2,
                             [v1, v2, m12sq, lam5], seed=3)
    assert ok, diff
    assert [str(s) for s in d_odd.goldstones] == ["G0"]

    # without angle=: the same rotation written with the VEVs
    d_ch = model.diagonalize(charged, names=["Gp", "Hp"],
                             conjugate_names=["Gm", "Hm"])
    R = d_ch.rotation.matrix
    expected = rotation_2x2(sp.atan(v2 / v1))
    assert (R - expected).applyfunc(sp.simplify) == sp.zeros(2, 2)
    mHp2 = (m12sq / (v1 * v2) - (lam4 + lam5) / 2) * (v1**2 + v2**2)
    assert d_ch.masses_squared[0] == 0
    assert sp.simplify(d_ch.masses_squared[1] - mHp2) == 0

    phys = model.physical
    assert [str(s) for s in phys.bosons] == ["G0", "A0", "Gp", "Gm", "Hp", "Hm"]
    assert [str(s) for s in phys.goldstones] == ["G0", "Gp", "Gm"]
    assert {str(k): q for k, q in phys.charges.items()} == {
        "G0": 0, "A0": 0, "Gp": 1, "Gm": -1, "Hp": 1, "Hm": -1}
    Gp, Hp = d_ch.new_fields
    assert phys.cmap == {sp.conjugate(Gp): phys.conjugates[Gp],
                         sp.conjugate(Hp): phys.conjugates[Hp]}


def test_thdm_cp_even_matches_the_handbuilt_tan2alpha_rotation():
    model, H1, H2, P = _thdm()
    even = model.mass_blocks(kind="scalar")[0]
    d = model.diagonalize(even, names=["H", "h"], angle="alpha")
    alpha = sp.Symbol("alpha", real=True)
    Hh, h = d.new_fields
    ref = diagonalize_orthogonal_2x2(even.matrix, even.fields, [Hh, h],
                                     angle=alpha)
    assert d.rotation.matrix == ref.matrix == rotation_2x2(alpha)
    assert d.angle_relation == ref.angle_relation
    assert d.angle_solution == ref.angle_solution       # principal branch
    assert d.branch_valid_for is None
    assert model.rotations[-1] is d.rotation
    # same registered rotation ⟹ same physical Lagrangian as the manual route
    model_ref, H1r, H2r, _ = _thdm()
    model_ref.rotate(Rotation([H1r.H10_r, H2r.H20_r], [Hh, h],
                              rotation_2x2(alpha)))
    assert sp.expand(model.physical_lagrangian("potential")
                     - model_ref.physical_lagrangian("potential")) == 0


@pytest.mark.parametrize("v1,v2,m11_gt_m22", [(233.7, 76.3, False),
                                               (76.3, 233.7, True)])
def test_values_choose_the_branch_with_the_heavier_state_first(
        v1, v2, m11_gt_m22):
    """With values=, names[0] is the heavier state on either side of
    M11 = M22 (the principal ½·atan only does that for M11 > M22)."""
    model, H1, H2, P = _thdm()
    values = _values(P)
    values[P["v1"].s], values[P["v2"].s] = v1, v2
    values[P["lam1"].s], values[P["lam2"].s] = 0.3, 0.3
    even = model.mass_blocks(kind="scalar")[0]
    Mn = even.matrix.subs(values)
    assert (Mn[0, 0] > Mn[1, 1]) == m11_gt_m22

    d = model.diagonalize(even, names=["H", "h"], values=values,
                          register=False)
    th = float(d.angle_solution.subs(values))
    D = rotation_2x2(th) * Mn * rotation_2x2(th).T
    assert abs(D[0, 1]) < 1e-8 * abs(D[0, 0])
    assert D[0, 0] > D[1, 1]                          # "H" is the heavier
    assert d.branch_valid_for == (1 if m11_gt_m22 else -1,
                                  1 if Mn[0, 1] > 0 else -1)
    # the symbolic tan(2θ) relation does not depend on the branch
    assert d.angle_relation.rhs == sp.cancel(
        2 * even.matrix[0, 1] / (even.matrix[0, 0] - even.matrix[1, 1]))

    light = model.diagonalize(even, names=["h", "H"], values=values,
                              order="light_first", register=False)
    th2 = float(light.angle_solution.subs(values))
    D2 = rotation_2x2(th2) * Mn * rotation_2x2(th2).T
    assert D2[0, 0] < D2[1, 1]


def test_cp_violating_potential_merges_the_neutral_block():
    """General case first: with a complex λ5 the CP-even and CP-odd
    fluctuations mix, so the model reports one 4×4 neutral block."""
    model, H1, H2, P = _thdm(complex_lam5=True)
    neutral, charged = model.mass_blocks()
    assert neutral.size == 4 and neutral.cp == "mixed"
    assert len(neutral.goldstones) == 1 and charged.size == 2

    with pytest.raises(NotImplementedError, match="values="):
        model.diagonalize(neutral, names=["G0", "h1", "h2", "h3"])

    d = model.diagonalize(neutral, names=["G0", "h1", "h2", "h3"],
                          values=_values(P), order="light_first")
    m2 = d.masses_squared
    assert m2[0] == 0 and all(m2[k] < m2[k + 1] for k in range(3))
    R = d.rotation.matrix
    assert (R * R.T - sp.eye(4)).norm() < 1e-30
    # the numeric zero mode is the group-theory Goldstone direction
    g = neutral.goldstones[0].subs(_values(P))
    assert abs(abs((R.row(0) * g)[0]) - 1) < 1e-12

    # CP-conserving limit: the block splits back into even ⊕ odd
    cp_model, *_ = _thdm()
    assert [b.size for b in cp_model.mass_blocks()] == [2, 2, 2]


def test_singlet_extension_uses_the_tan2theta_route():
    """SM + real singlet with a portal: one 2×2 CP-even block, no Goldstone."""
    P = ParameterSet()
    SU2L, U1Y = _ew(P)
    v = P.external("v", 246.0, positive=True, unit_dim=1)
    vs = P.external("vs", 500.0, positive=True, unit_dim=1)
    lam, lamS, lamHS = (P.external(n, x) for n, x in
                        (("lam", 0.13), ("lamS", 0.2), ("lamHS", 0.05)))
    mu2 = P.internal("mu2", unit_dim=2)
    muS2 = P.internal("muS2", unit_dim=2)
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    S = Scalar("S", reps={}, real=True)
    H.expand_vev({H.H0: v})
    S.expand_vev({S.S: vs})
    HdH, s = inner(H, H), S.S
    V = (-mu2 * HdH + lam * HdH**2 - muS2 * s**2 / 2 + lamS * s**4 / 4
         + lamHS * HdH * s**2 / 2)
    model = Model("xSM", gauge_groups=[SU2L, U1Y], fields=[H, S], parameters=P,
                  lagrangian=Lagrangian().add(-V, sector="potential"))
    model.solve_tadpoles([mu2, muS2])
    blocks = model.mass_blocks()
    assert [b.fields for b in blocks] == [[H.H0_r, S.S], [H.H0_i], [H.Gp]]
    even = blocks[0]
    assert even.cp == "even" and not even.goldstones
    expected = sp.Matrix([[2 * lam.s * v.s**2, lamHS.s * v.s * vs.s],
                          [lamHS.s * v.s * vs.s, 2 * lamS.s * vs.s**2]])
    assert (even.matrix - expected).applyfunc(sp.simplify) == sp.zeros(2, 2)

    d = model.diagonalize(even, names=["h1", "h2"], angle="theta")
    tan2 = lamHS.s * v.s * vs.s / (lam.s * v.s**2 - lamS.s * vs.s**2)
    assert sp.simplify(d.angle_relation.rhs - tan2) == 0
    assert d.rotation.matrix == rotation_2x2(sp.Symbol("theta", real=True))


# ---------------------------------------------------------------- guards

def test_unsolved_tadpoles_are_reported():
    model, *_ = _thdm(solve=False)
    with pytest.raises(ValueError, match="solve the tadpoles first"):
        model.mass_blocks()


def test_inert_doublet_needs_a_zero_vev_expansion():
    """A VEV-less doublet with a λ5 term: its neutral component splits into
    re/im, which needs expand_vev({comp: 0})."""

    def build(expand):
        P = ParameterSet()
        SU2L, U1Y = _ew(P)
        v = P.external("v", 246.0, positive=True, unit_dim=1)
        lam, l3, l5 = (P.external(n, x) for n, x in
                       (("lam", 0.13), ("lam3", 0.3), ("lam5", -0.1)))
        m2sq = P.external("m2sq", 4.0e4, unit_dim=2)
        mu2 = P.internal("mu2", unit_dim=2)
        H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
        Phi = Scalar("Phi", reps={SU2L: 2, U1Y: half},
                     component_names=["Cp", "N0"])
        H.expand_vev({H.H0: v})
        if expand:
            Phi.expand_vev({Phi.N0: 0})
        a, b, x = inner(H, H), inner(Phi, Phi), inner(H, Phi)
        V = (-mu2 * a + lam * a**2 + m2sq * b + l3 * a * b
             + l5 / 2 * (x**2 + hc(x)**2))
        model = Model("IDM", gauge_groups=[SU2L, U1Y], fields=[H, Phi],
                      parameters=P,
                      lagrangian=Lagrangian().add(-V, sector="potential"))
        model.solve_tadpoles([mu2])
        return model, Phi, P

    model, Phi, _ = build(expand=False)
    with pytest.raises(NotImplementedError,
                       match=r"expand_vev\(\{N0: 0\}\)"):
        model.mass_blocks()

    model, Phi, P = build(expand=True)
    blocks = model.mass_blocks()
    assert [b.size for b in blocks] == [1, 1, 1, 1, 1, 1]   # nothing mixes
    by_field = {b.fields[0]: b for b in blocks}
    v, m2sq, l3, l5 = (P[n].s for n in ("v", "m2sq", "lam3", "lam5"))
    # inert masses: m² = m2² + (λ3 ± λ5) v²/2 for the CP-even / CP-odd state
    assert sp.simplify(by_field[Phi.N0_r].matrix[0, 0]
                       - (m2sq + (l3 + l5) * v**2 / 2)) == 0
    assert sp.simplify(by_field[Phi.N0_i].matrix[0, 0]
                       - (m2sq + (l3 - l5) * v**2 / 2)) == 0
    assert sp.simplify(by_field[Phi.Cp].matrix[0, 0]
                       - (m2sq + l3 * v**2 / 2)) == 0
    assert not by_field[Phi.N0_i].goldstones and not by_field[Phi.Cp].goldstones


def test_opposite_charge_mixing_is_refused():
    """Doublets with Y = +1/2 and Y = −1/2: H1+ mixes with conj(H2−)."""
    P = ParameterSet()
    SU2L, U1Y = _ew(P)
    v1 = P.external("v1", 200.0, positive=True, unit_dim=1)
    v2 = P.external("v2", 143.0, positive=True, unit_dim=1)
    b = P.external("b", 1.0e4, unit_dim=2)
    m1 = P.internal("m1sq", unit_dim=2)
    m2 = P.internal("m2sq", unit_dim=2)
    Hu = Scalar("Hu", reps={SU2L: 2, U1Y: half}, component_names=["Hup", "Hu0"])
    Hd = Scalar("Hd", reps={SU2L: 2, U1Y: -half}, component_names=["Hd0", "Hdm"])
    Hu.expand_vev({Hu.Hu0: v1})
    Hd.expand_vev({Hd.Hd0: v2})
    from feynlag import eps
    e = eps(Hu, Hd)
    V = m1 * inner(Hu, Hu) + m2 * inner(Hd, Hd) - b * (e + hc(e))
    model = Model("HuHd", gauge_groups=[SU2L, U1Y], fields=[Hu, Hd],
                  parameters=P,
                  lagrangian=Lagrangian().add(-V, sector="potential"))
    model.solve_tadpoles([m1, m2])
    with pytest.raises(NotImplementedError, match="opposite charge"):
        model.mass_blocks()


def test_symbolic_block_without_route_and_explicit_rotation():
    model, H1, H2, P = _thdm()
    even, odd, _ = model.mass_blocks()
    with pytest.raises(ValueError, match="need 2 names"):
        model.diagonalize(even, names=["h"])
    # an explicit rotation is accepted only if it diagonalizes the block
    v1, v2 = P["v1"].s, P["v2"].s
    good = rotation_2x2(sp.atan(v2 / v1))
    d = model.diagonalize(odd, names=["G0", "A0"], rotation=good,
                          register=False)
    assert d.rotation.matrix == good
    with pytest.raises(ValueError, match="does not diagonalize"):
        model.diagonalize(odd, names=["G0", "A0"],
                          rotation=rotation_2x2(sp.pi / 7), register=False)
    assert model.rotations == [] and model.physical.bosons == []


# -------------------------------------------------------------- fermions

def _vll():
    ew = electroweak_scaffold(lam=0.129)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    P = ParameterSet(*ew.parameters)
    ye = P.external("ye", 0.01, positive=True)
    lamE = P.external("lamE", 0.4, positive=True)
    MPsi = P.external("MPsi", 1000.0, positive=True, unit_dim=1)
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -half}, chirality="L",
                     component_names=["nuL", "eL"])
    eRf = WeylFermion("eRf", reps={U1Y: -1}, chirality="R",
                      component_names=["eR"])
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -half}, chirality="L",
                       component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -half}, chirality="R",
                       component_names=["NR", "ER"])
    L = Lagrangian()
    ew.add_higgs(L)
    L.add(yukawa(ye, Ll, H, eRf) + yukawa(lamE, PsiL, H, eRf)
          + dirac_mass(MPsi, PsiL, PsiR), sector="yukawa")
    L.add_kinetic(Ll, eRf, PsiL, PsiR)
    model = Model("VLL", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, eRf, PsiL, PsiR], parameters=P,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    return model, (Ll, eRf, PsiL, PsiR), P, ew


def test_vll_fermion_blocks():
    model, (Ll, eRf, PsiL, PsiR), P, ew = _vll()
    nu, lep, heavy_n = model.mass_blocks(kind="fermion")
    assert nu.fields == [Ll.nuL] and nu.right == [] and not nu.diagonalizable
    assert "massless" in nu.note and nu.charge == 0

    assert lep.fields == [Ll.eL, PsiL.EL] and lep.right == [eRf.eR, PsiR.ER]
    assert lep.charge == -1 and lep.diagonalizable
    v, ye, lamE, MPsi = (P[n].s for n in ("v", "ye", "lamE", "MPsi"))
    s2 = sp.sqrt(2)
    # the charged-lepton matrix of examples/sm_vll.py (rows bar-L, cols R)
    assert lep.matrix == sp.Matrix([[ye * v / s2, 0], [lamE * v / s2, MPsi]])

    assert heavy_n.fields == [PsiL.NL] and heavy_n.right == [PsiR.NR]
    assert heavy_n.matrix == sp.Matrix([[MPsi]])
    d = model.diagonalize(heavy_n)
    assert d.rotations == [] and d.masses == [MPsi]
    with pytest.raises(NotImplementedError, match="massless"):
        model.diagonalize(nu)


def test_vll_rotations_equal_the_handbuilt_ones():
    model, (Ll, eRf, PsiL, PsiR), P, ew = _vll()
    lep = model.mass_blocks(kind="fermion")[1]
    d = model.diagonalize(lep, names=["e1", "e2"], angle=("thL", "thR"))
    thL, thR = sp.symbols("thL thR", real=True)
    i = FL_I
    e1L, e2L = d.new_fields
    e1R, e2R = d.new_right
    assert [str(b) for b in (e1L, e2L, e1R, e2R)] == ["e1L", "e2L", "e1R", "e2R"]

    refL, refR = diagonalize_svd_2x2(
        lep.matrix, [Ll.eL[i], PsiL.EL[i]], [eRf.eR[i], PsiR.ER[i]],
        [e1L[i], e2L[i]], [e1R[i], e2R[i]], angle_left=thL, angle_right=thR)
    rotL, rotLbar, rotR, rotRbar = d.rotations
    assert model.rotations[-4:] == d.rotations
    assert rotL.matrix == refL.matrix == rotation_2x2(thL)
    assert rotR.matrix == refR.matrix == rotation_2x2(thR)
    assert rotL.old_fields == refL.old_fields and rotL.new_fields == refL.new_fields
    assert rotL.angle_relation == refL.angle_relation
    assert rotR.angle_solution == refR.angle_solution
    # bar legs: same real matrix, on the bar components, same flavour index
    assert rotLbar.old_fields == [Ll.eLbar[i], PsiL.ELbar[i]]
    assert rotLbar.matrix == rotL.matrix and rotRbar.matrix == rotR.matrix
    assert [str(f.base) for f in rotRbar.new_fields] == ["e1Rbar", "e2Rbar"]
    assert {"thL", "thR"} <= {p.name for p in model.parameters}

    # numerically: diagonal, light state first (m_e ≈ ye v/√2, m_E ≈ MPsi)
    bench = {p.s: p.value for p in P.externals}
    tl = float(rotL.angle_solution.subs(bench))
    tr = float(rotR.angle_solution.subs(bench))
    D = rotation_2x2(tl) * lep.matrix.subs(bench) * rotation_2x2(tr).T
    assert abs(D[0, 1]) < 1e-9 and abs(D[1, 0]) < 1e-9
    assert abs(D[0, 0]) < abs(D[1, 1])
    masses = [float(m.subs({thL: tl, thR: tr}).subs(bench)) for m in d.masses]
    assert abs(masses[0] - float(D[0, 0])) < 1e-9
    assert abs(masses[1] - float(D[1, 1])) < 1e-6


def test_flavoured_and_majorana_blocks_are_reported_not_rotated():
    from feynlag import majorana_mass, tilde

    ew = electroweak_scaffold()
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -half}, chirality="L",
                     nflavors=3, component_names=["nuL", "eL"])
    eRf = WeylFermion("eRf", reps={U1Y: -1}, chirality="R", nflavors=3,
                      component_names=["eR"])
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=3,
                      component_names=["nR"])
    Ye, Yv, MR = sp.IndexedBase("Ye"), sp.IndexedBase("Yv"), sp.IndexedBase("MR")
    L = Lagrangian()
    ew.add_higgs(L)
    L.add(yukawa(Ye, Ll, H, eRf) + yukawa(Yv, Ll, tilde(H), nuR),
          sector="yukawa")
    L.add(majorana_mass(MR, nuR), sector="other")
    model = Model("seesaw", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, eRf, nuR], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    nu, lep = model.mass_blocks(kind="fermion")
    v = ew.v.s
    assert lep.fields == [Ll.eL] and lep.nflavors == 3
    assert lep.matrix == sp.Matrix(3, 3, lambda a, b: Ye[a, b] * v / sp.sqrt(2))
    assert not lep.diagonalizable and "numeric" in lep.note
    assert nu.fields == [Ll.nuL] and nu.right == [nuR.nR]
    assert nu.matrix == sp.Matrix(3, 3, lambda a, b: Yv[a, b] * v / sp.sqrt(2))
    assert not nu.diagonalizable and "diagonalize_takagi" in nu.note
    with pytest.raises(NotImplementedError, match="diagonalize_takagi"):
        model.diagonalize(nu, names=["n"])


# ---------------------------------------------------------- gauge bosons

def _sm():
    ew = electroweak_scaffold(lam=0.129)
    L = Lagrangian()
    ew.add_higgs(L)
    model = Model("SM", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ParameterSet(*ew.parameters), lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    return model, ew


def test_sm_gauge_blocks_and_default_order():
    model, ew = _sm()
    g, gp, v = ew.gw.s, ew.g1.s, ew.v.s
    W1, W2, W3 = ew.W.components
    B = ew.B.components[0]
    blocks = model.mass_blocks()
    assert [b.kind for b in blocks] == ["scalar"] * 3 + ["gauge"] * 2
    charged, neutral = model.mass_blocks(kind="gauge")
    assert charged.fields == [W1, W2] and charged.charge == 1
    assert charged.pairs == [(W1, W2, 1)]
    assert charged.matrix == sp.Matrix([[g**2 * v**2 / 4]])
    assert neutral.fields == [W3, B] and neutral.charge == 0
    # the photon direction, from Q = T3 + Y: (g' W3 + g B)/sqrt(g² + g'²)
    a = neutral.massless[0]
    assert (a - sp.Matrix([gp, g]) / sp.sqrt(g**2 + gp**2)).applyfunc(
        sp.simplify) == sp.zeros(2, 1)
    assert "massless: 1" in repr(neutral) and "massless: 0" in repr(charged)


def test_sm_generic_rotations_equal_the_validated_helpers():
    """The generic route registers the same matrices as weinberg_rotation /
    charged_current_rotation — the ones the UFO sign conventions were
    validated with — so every downstream vertex is unchanged."""
    model, ew = _sm()
    g, gp, v = ew.gw.s, ew.g1.s, ew.v.s
    charged, neutral = model.mass_blocks(kind="gauge")
    dW = model.diagonalize(charged, names=["Wp"], conjugate_names=["Wm"])
    dZ = model.diagonalize(neutral, names=["Z", "A"])

    ref, ew_ref = _sm()
    Z, A = weinberg_rotation(ref, ew_ref.SU2L, ew_ref.U1Y)
    Wp, Wm = charged_current_rotation(ref, ew_ref.SU2L)
    rot_weinberg, rot_w = ref.rotations

    assert dW.rotation.matrix == rot_w.matrix
    assert dW.rotation.kind == "unitary"
    assert (dZ.rotation.matrix - rot_weinberg.matrix).applyfunc(
        sp.simplify) == sp.zeros(2, 2)
    assert dW.masses_squared == [g**2 * v**2 / 4]
    assert sp.simplify(dZ.masses_squared[0] - (g**2 + gp**2) * v**2 / 4) == 0
    assert dZ.masses_squared[1] == 0                     # the photon, exactly

    phys = model.physical
    assert [str(b) for b in phys.bosons] == ["Wp", "Wm", "Z", "A"]
    assert {str(k): q for k, q in phys.charges.items()} == {
        "Wp": 1, "Wm": -1, "Z": 0, "A": 0}
    assert phys.conjugates[dW.new_fields[0]] == dW.new_right[0]
    assert phys.cmap == {}                               # real components

    # the symbols are equal by name+assumptions, so compare directly:
    # same physical kinetic Lagrangian ⟹ same hWW, hZZ, VSS, VVSS vertices
    diff = sp.expand(model.physical_lagrangian("kinetic")
                     - ref.physical_lagrangian("kinetic"))
    assert sp.simplify(diff) == 0
    # and the same Yang–Mills cubics and quartics (AWW, ZWW, WWWW, …)
    mine = {tuple(map(str, vx.particles)): vx for vx in model.gauge_vertices()}
    theirs = {tuple(map(str, vx.particles)): vx for vx in ref.gauge_vertices()}
    assert mine.keys() == theirs.keys() and len(mine) >= 6
    for key, vx in mine.items():
        other = theirs[key]
        assert sp.simplify(vx.coupling - other.coupling) == 0, key
        for c1, c2 in zip(vx.meta.get("structures", {}).values(),
                          other.meta.get("structures", {}).values()):
            assert sp.simplify(c1 - c2) == 0, key


def test_weinberg_angle_form():
    """angle= writes the photon-fixed 2×2 as rotation_2x2(−θ_W), literally
    the matrix weinberg_rotation uses, with tan θ_W = g'/g attached."""
    model, ew = _sm()
    neutral = model.mass_blocks(kind="gauge")[1]
    d = model.diagonalize(neutral, names=["Z", "A"], angle="thetaW")
    thw = sp.Symbol("thetaW", real=True)
    assert d.rotation.matrix == rotation_2x2(-thw)
    assert d.angle_relation == sp.Eq(sp.tan(thw), ew.g1.s / ew.gw.s)
    assert model.parameters["thetaW"].expr == sp.atan(ew.g1.s / ew.gw.s)
    with pytest.raises(ValueError, match="need 2 names"):
        model.diagonalize(neutral, names=["Z"], register=False)


def test_gauge_guards():
    # gauge fields but no kinetic sector: no mass matrix to read
    ew = electroweak_scaffold()
    H = ew.H
    V = -ew.mu2 * inner(H, H) + ew.lam * inner(H, H)**2
    model = Model("nokin", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ew.parameters,
                  lagrangian=Lagrangian().add(-V, sector="potential"))
    model.solve_tadpoles([ew.mu2])
    with pytest.raises(ValueError, match="no 'kinetic' sector"):
        model.mass_blocks(kind="gauge")

    # the negative-charge partner is never auto-named
    model, _ = _sm()
    charged = model.mass_blocks(kind="gauge")[0]
    with pytest.raises(ValueError, match="conjugate_names"):
        model.diagonalize(charged, names=["Wp"])

    # an unbroken group (colour) has no block and is never rotated
    ew = electroweak_scaffold()
    SU3c = SU3("SU3c", coupling=sp.Symbol("gs", positive=True))
    G = SU3c.bosons("G")
    L = Lagrangian()
    ew.add_higgs(L)
    model = Model("SMc", gauge_groups=ew.gauge_groups + [SU3c],
                  fields=ew.fields + [G], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    blocks = model.mass_blocks(kind="gauge")
    assert len(blocks) == 2
    assert not any(set(b.fields) & set(G.components) for b in blocks)


def _u1x():
    """SM × U(1)_X with an X-charged Higgs (charge a/2) and a singlet S:
    the gauge sector of examples/sm_u1x.py."""
    P = ParameterSet()
    SU2L, U1Y = _ew(P)
    gX = P.external("gX", 0.5, positive=True)
    U1X = U1("U1X", coupling=gX)
    aX = P.external("aX", 0.4)
    qS = P.external("qS", 1.0)
    v = P.external("v", 246.0, positive=True, unit_dim=1)
    vX = P.external("vX", 2000.0, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: half, U1X: aX.s / 2},
               component_names=["Gp", "H0"])
    S = Scalar("S", reps={U1X: qS.s}, component_names=["S0"])
    H.expand_vev({H.H0: v})
    S.expand_vev({S.S0: vX})
    W, B, X = SU2L.bosons("W"), U1Y.bosons("B"), U1X.bosons("X")
    L = Lagrangian()
    L.add_kinetic(H, S)
    model = Model("U1X", gauge_groups=[SU2L, U1Y, U1X],
                  fields=[H, S, W, B, X], parameters=P, lagrangian=L)
    return model, P, (W, B, X)


def test_u1x_neutral_block_photon_and_z_zprime_mixing():
    model, P, (W, B, X) = _u1x()
    g, gp, gx, a, qs, v, vX = (P[n].s for n in
                               ("gw", "g1", "gX", "aX", "qS", "v", "vX"))
    W3, B0, X0 = W.components[2], B.components[0], X.components[0]
    charged, neutral = model.mass_blocks(kind="gauge")
    assert neutral.fields == [W3, B0, X0] and len(neutral.massless) == 1
    # the photon has no X component: Q = T3 + Y, the singlet VEV breaks X
    photon = neutral.massless[0]
    assert (photon - sp.Matrix([gp, g, 0]) / sp.sqrt(g**2 + gp**2)).applyfunc(
        sp.simplify) == sp.zeros(3, 1)

    d = model.diagonalize(neutral, names=["Z", "Zp", "A"], angle="thetap")
    thp = sp.Symbol("thetap", real=True)
    gZ = sp.sqrt(g**2 + gp**2)
    # tan 2θ′ pinned in test_u1x.py::test_zx_block_and_mixing_angle
    expected_tan2 = 2 * a * gZ * gx * v**2 / (
        a**2 * gx**2 * v**2 + 4 * gx**2 * qs**2 * vX**2 - gZ**2 * v**2)
    assert sp.simplify(d.angle_relation.rhs - expected_tan2) == 0
    ok, diff = numeric_equal(d.angle_relation.rhs, expected_tan2,
                             [g, gp, gx, a, qs, v, vX], seed=5)
    assert ok, diff

    # ONE composed rotation = (Z–Z′ rotation) · (Weinberg rotation), the
    # two chained steps examples/sm_u1x.py used to take by hand
    thw = sp.atan(gp / g)
    cw, sw = sp.cos(thw), sp.sin(thw)
    R1 = sp.Matrix([[cw, -sw, 0], [0, 0, 1], [sw, cw, 0]])   # (Z0, X, A)
    R2 = sp.diag(rotation_2x2(thp), 1)
    assert (d.rotation.matrix - R2 * R1).applyfunc(sp.simplify) \
        == sp.zeros(3, 3)
    assert d.masses_squared[2] == 0

    # numerically: diagonal, with the sum of m² equal to the trace
    values = _values(P)
    th = float(d.angle_solution.subs(values))
    Rn = d.rotation.matrix.subs(thp, th).subs(values)
    Mn = neutral.matrix.subs(values)
    D = Rn * Mn * Rn.T
    assert max(abs(D[i, j]) for i in range(3) for j in range(3)
               if i != j) < 1e-6
    assert abs(D[2, 2]) < 1e-6 and abs(D.trace() - Mn.trace()) < 1e-6


def test_u1x_values_order_the_z_and_zprime():
    model, P, _ = _u1x()
    neutral = model.mass_blocks(kind="gauge")[1]
    values = _values(P)
    d = model.diagonalize(neutral, names=["Z", "Zp", "A"], angle="thetap",
                          values=values, order="light_first", register=False)
    thp = sp.Symbol("thetap", real=True)
    th = float(d.angle_solution.subs(values))
    m2 = [float(m.subs(thp, th).subs(values)) for m in d.masses_squared]
    assert 0 < m2[0] < m2[1] and m2[2] == 0
    assert 85 < m2[0] ** 0.5 < 95 and m2[1] ** 0.5 > 900   # Z and a TeV Z′
    assert d.branch_valid_for is not None

    heavy = model.diagonalize(neutral, names=["Zp", "Z", "A"], values=values,
                              register=False)
    th2 = float(heavy.angle_solution.subs(values))
    R = heavy.rotation.matrix.subs(values)
    D = R * neutral.matrix.subs(values) * R.T
    assert D[0, 0] > D[1, 1] > 0 and abs(D[2, 2]) < 1e-6
    assert abs(th2 - th) > 1                      # the other tan(2θ) branch


def _left_right():
    """SU(2)_L × SU(2)_R × U(1)_{B−L} with a bidoublet (κ1, κ2) and a
    right-handed doublet (v_R): no SM-specific code path applies."""
    P = ParameterSet()
    gL = P.external("gL", 0.65, positive=True)
    gR = P.external("gR", 0.60, positive=True)
    gB = P.external("gB", 0.40, positive=True)
    SU2L, SU2R = SU2("SU2L", coupling=gL), SU2("SU2R", coupling=gR)
    U1B = U1("U1BL", coupling=gB)
    k1 = P.external("k1", 240.0, positive=True, unit_dim=1)
    k2 = P.external("k2", 30.0, positive=True, unit_dim=1)
    vR = P.external("vR", 5000.0, positive=True, unit_dim=1)
    Phi = Scalar("Phi", reps={SU2L: 2, SU2R: 2},
                 component_names=["p11", "p12", "p21", "p22"])
    Chi = Scalar("Chi", reps={SU2R: 2, U1B: half},
                 component_names=["chip", "chi0"])
    Phi.expand_vev({Phi.p12: k1, Phi.p21: k2})
    Chi.expand_vev({Chi.chi0: vR})
    WL, WR, B = SU2L.bosons("WL"), SU2R.bosons("WR"), U1B.bosons("B")
    L = Lagrangian()
    L.add_kinetic(Phi, Chi)
    model = Model("LR", gauge_groups=[SU2L, SU2R, U1B],
                  fields=[Phi, Chi, WL, WR, B], parameters=P, lagrangian=L)
    return model, P, (WL, WR, B)


def test_left_right_charged_mixing_and_photon():
    model, P, (WL, WR, B) = _left_right()
    gL, gR, gB, k1, k2, vR = (P[n].s for n in
                              ("gL", "gR", "gB", "k1", "k2", "vR"))
    charged, neutral = model.mass_blocks(kind="gauge")

    # W_L–W_R mix through κ1 κ2 (the mass matrix in the W⁺ basis)
    assert charged.fields == list(WL.components[:2]) + list(WR.components[:2])
    assert len(charged.pairs) == 2 and charged.charge == 1
    expected = sp.Matrix([
        [gL**2 * (k1**2 + k2**2) / 4, gL * gR * k1 * k2 / 2],
        [gL * gR * k1 * k2 / 2, gR**2 * (k1**2 + k2**2 + vR**2) / 4]])
    assert (charged.matrix - expected).applyfunc(sp.simplify) \
        == sp.zeros(2, 2)

    # the photon of Q = T3L + T3R + (B−L): ∝ W3L/gL + W3R/gR + B/gB
    assert neutral.fields == [WL.components[2], WR.components[2],
                              B.components[0]]
    raw = sp.Matrix([1 / gL, 1 / gR, 1 / gB])
    photon = neutral.massless[0]
    assert (photon - raw / sp.sqrt(raw.dot(raw))).applyfunc(sp.simplify) \
        == sp.zeros(3, 1)
    assert (neutral.matrix * raw).applyfunc(sp.simplify) == sp.zeros(3, 1)

    d = model.diagonalize(charged, names=["W1p", "W2p"],
                          conjugate_names=["W1m", "W2m"], angle="xi")
    xi = sp.Symbol("xi", real=True)
    tan2xi = 4 * gL * gR * k1 * k2 / (
        gL**2 * (k1**2 + k2**2) - gR**2 * (k1**2 + k2**2 + vR**2))
    assert sp.simplify(d.angle_relation.rhs - tan2xi) == 0
    # rows: W1⁺ = cξ W_L⁺ + sξ W_R⁺ with W_X⁺ = (W_X¹ − i W_X²)/√2, then
    # W2⁺, then the conjugates
    c, s_, r2 = sp.cos(xi), sp.sin(xi), sp.sqrt(2)
    plus = sp.Matrix([[c, -sp.I * c, s_, -sp.I * s_],
                      [-s_, sp.I * s_, c, -sp.I * c]]) / r2
    assert (d.rotation.matrix - sp.Matrix.vstack(plus, plus.conjugate())
            ).applyfunc(sp.simplify) == sp.zeros(4, 4)

    # independent numeric check on the full 4×4 real-basis mass matrix
    values = _values(P)
    M4 = model.gauge_mass_matrix(charged.fields).subs(values)
    th = float(d.angle_solution.subs(values))
    R = d.rotation.matrix.subs(xi, th)
    D = (R * M4 * R.H).applyfunc(lambda e: complex(sp.N(e)))
    assert max(abs(D[i, j]) for i in range(4) for j in range(4)
               if i != j) < 1e-6
    assert abs(D[0, 0] - D[2, 2]) < 1e-6           # W1⁺ and W1⁻: same mass
    assert abs(D.trace() - M4.trace()) < 1e-6
    assert {str(k): q for k, q in model.physical.charges.items()} == {
        "W1p": 1, "W2p": 1, "W1m": -1, "W2m": -1}


def test_left_right_neutral_block():
    model, P, _ = _left_right()
    neutral = model.mass_blocks(kind="gauge")[1]
    with pytest.raises(ValueError, match="need 3 names"):
        model.diagonalize(neutral, names=["Z", "A"])
    values = _values(P)
    d = model.diagonalize(neutral, names=["Z1", "Z2", "A"], angle="phi",
                          values=values, order="light_first")
    phi = sp.Symbol("phi", real=True)
    th = float(d.angle_solution.subs(values))
    Mn = neutral.matrix.subs(values)
    R = d.rotation.matrix.subs(phi, th).subs(values)
    D = R * Mn * R.T
    assert max(abs(D[i, j]) for i in range(3) for j in range(3)
               if i != j) < 1e-5
    assert 0 < D[0, 0] < D[1, 1] and abs(D[2, 2]) < 1e-6
    assert (R * R.T - sp.eye(3)).norm() < 1e-9
    # det of the massive 2×2 = product of the two masses² (sum of principal
    # 2×2 minors of M, the massless mode contributing nothing)
    minors = sum(Mn.extract(idx, idx).det() for idx in
                 ([0, 1], [0, 2], [1, 2]))
    assert abs(D[0, 0] * D[1, 1] - minors) / minors < 1e-9
    assert {str(k) for k in model.physical.bosons} == {"Z1", "Z2", "A"}


# ----------------------------------------- unequal flavour counts (FG-12)

def _seesaw_3_2(majorana=True):
    """SM leptons (3 generations) + 2 ν_R with a 3×2 Dirac Yukawa."""
    from feynlag import majorana_mass, tilde

    ew = electroweak_scaffold()
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -half}, chirality="L",
                     nflavors=3, component_names=["nuL", "eL"])
    eRf = WeylFermion("eRf", reps={U1Y: -1}, chirality="R", nflavors=3,
                      component_names=["eR"])
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=2,
                      component_names=["nR"])
    Ye, Yv, MR = sp.IndexedBase("Ye"), sp.IndexedBase("Yv"), sp.IndexedBase("MR")
    L = Lagrangian()
    ew.add_higgs(L)
    L.add(yukawa(Ye, Ll, H, eRf) + yukawa(Yv, Ll, tilde(H), nuR),
          sector="yukawa")
    if majorana:
        L.add(majorana_mass(MR, nuR), sector="other")
    model = Model("seesaw32", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Ll, eRf, nuR], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    return model, ew, Ll, nuR, Yv


def test_unequal_flavour_majorana_block_is_reported():
    """3+2 seesaw: the (ν_L, ν_R) block is a 3×2 Dirac matrix reported as
    not diagonalizable with the Takagi note, as for one generation; the
    charged leptons are still reported (it used to raise and take the whole
    fermion report with it)."""
    model, ew, Ll, nuR, Yv = _seesaw_3_2()
    nu, lep = model.mass_blocks(kind="fermion")
    v = ew.v.s
    assert lep.fields == [Ll.eL] and lep.matrix.shape == (3, 3)
    assert nu.fields == [Ll.nuL] and nu.right == [nuR.nR]
    assert nu.matrix == sp.Matrix(3, 2, lambda a, b: Yv[a, b] * v / sp.sqrt(2))
    assert not nu.diagonalizable and "Majorana" in nu.note
    assert "diagonalize_takagi" in nu.note
    assert len(model.mass_blocks()) > 2        # a bare call no longer raises


def test_unequal_flavour_dirac_block_is_rectangular():
    model, ew, Ll, nuR, Yv = _seesaw_3_2(majorana=False)
    nu = next(b for b in model.mass_blocks(kind="fermion")
              if b.fields == [Ll.nuL])
    assert nu.matrix.shape == (3, 2)
    assert not nu.diagonalizable and "different numbers of flavours" in nu.note

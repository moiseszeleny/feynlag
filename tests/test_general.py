"""The most general Lagrangian of a field content, the coverage of a
hand-written one, and self-explaining invariance failures."""

import pytest
import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, FL_I, FL_J, GaugeBoson, Lagrangian, Model,
    SU2, Scalar, U1, WeylFermion, ZN, diracPR, electroweak_scaffold,
    expand_bilinear, hc, inner, most_general_lagrangian, tilde, weinberg,
    yukawa,
)
from feynlag.invariance import u1_charge_breakdown

half = sp.Rational(1, 2)


def _groups():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1), [gw, g1]


def _two_doublets(SU2L, U1Y):
    H1 = Scalar("H1", reps={SU2L: 2, U1Y: half}, component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: half}, component_names=["H2p", "H20"])
    return H1, H2


def _leptons(SU2L, U1Y, n):
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -half}, chirality="L",
                     nflavors=n, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=n,
                     component_names=["eRc"])
    return Ll, eR


# ------------------------------------------------------ most general model

def test_most_general_sm_higgs_and_lepton():
    """One doublet + one lepton generation: μ², λ and one Yukawa."""
    SU2L, U1Y, params = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    Ll, eR = _leptons(SU2L, U1Y, 1)
    model = Model.most_general("SM1", [H, Ll, eR], [SU2L, U1Y],
                               parameters=params)
    general = model.general
    assert [str(c.name) for c, _t in general.terms] == ["m2_1", "lam_1", "y_1"]
    m2, lam, y = (c for c, _t in general.terms)
    assert (m2.unit_dim, lam.unit_dim, y.unit_dim) == (2, 0, 0)
    assert general.flavour_matrices == []

    HdH = inner(H, H)
    assert sp.expand(model.potential - (m2.s * HdH + lam.s * HdH**2)) == 0
    # the Yukawa term is exactly the field-level constructor's
    yuk = sum(t.expr for t in model.lagrangian.terms
              if t.name == general.terms[2][1].label)
    assert sp.expand(expand_bilinear(yuk - yukawa(y, Ll, H, eR))) == 0

    # the gauge bosons were added for the kinetic terms, and all is invariant
    assert sum(isinstance(f, GaugeBoson) for f in model.fields) == 2
    assert model.check_invariance().ok
    assert model.check_parameters().ok
    assert "3 operators" in str(general) and "y_1" in str(general)


def test_most_general_2hdm_has_the_ten_real_potential_operators():
    """Two doublets with the same hypercharge: 3 mass terms + 7 quartics
    (the general 2HDM with real couplings; λ6, λ7 included)."""
    SU2L, U1Y, _ = _groups()
    H1, H2 = _two_doublets(SU2L, U1Y)
    general = most_general_lagrangian([H1, H2], [SU2L, U1Y], kinetic=False)
    dims = [t.dim for _c, t in general.terms]
    assert dims.count(2) == 3 and dims.count(4) == 7
    assert [c.name for c, _t in general.terms][:4] == ["m2_1", "m2_2", "m2_3",
                                                       "lam_1"]

    # an exact Z2 (H2 odd) removes the three Z2-odd operators
    Z2 = ZN("Z2", 2)
    Z2.assign(0, H1)
    Z2.assign(1, H2)
    restricted = most_general_lagrangian([H1, H2], [SU2L, U1Y], [Z2],
                                         kinetic=False)
    assert len(restricted.terms) == 7      # m11², m22², λ1 … λ5


def test_most_general_flavour_matrices_and_names():
    SU2L, U1Y, _ = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    Ll, eR = _leptons(SU2L, U1Y, 3)
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=3,
                      component_names=["nR"])
    general = most_general_lagrangian(
        [H, Ll, eR, nuR], [SU2L, U1Y], kinetic=False,
        names={"Ll̄·H eR": "Ye", "Ll̄·H̃ nuR": "Yv", "(H†H)": "mu2"})
    by_name = {str(getattr(c, "name", c)): (c, t) for c, t in general.terms}
    assert {"Ye", "Yv", "mu2", "lam_1"} <= set(by_name)
    Ye, Yv = by_name["Ye"][0], by_name["Yv"][0]
    assert isinstance(Ye, sp.IndexedBase) and Ye in general.flavour_matrices
    # identical to the constructors with a flavour matrix
    terms = {t.name: t.expr for t in general.lagrangian.terms}
    assert sp.expand(expand_bilinear(
        terms["Ll̄·H eR"] - yukawa(Ye, Ll, H, eR))) == 0
    assert sp.expand(expand_bilinear(
        terms["Ll̄·H̃ nuR"] - yukawa(Yv, Ll, tilde(H), nuR))) == 0


# ------------------------------------------------------------- coverage

def test_coverage_reports_the_operators_a_2hdm_leaves_out():
    """A hand-written softly-broken-Z2 2HDM without the Z2 declared: λ6 and
    λ7 are allowed by the gauge symmetry and absent, and so is the H2
    lepton Yukawa."""
    SU2L, U1Y, _ = _groups()
    H1, H2 = _two_doublets(SU2L, U1Y)
    Ll, eR = _leptons(SU2L, U1Y, 3)
    m11, m22, m12 = sp.symbols("m11sq m22sq m12sq", real=True)
    l1, l2, l3, l4, l5 = sp.symbols("lam1:6", real=True)
    a, b, x = inner(H1, H1), inner(H2, H2), inner(H1, H2)
    V = (m11 * a + m22 * b - m12 * (x + hc(x)) + l1 / 2 * a**2
         + l2 / 2 * b**2 + l3 * a * b + l4 * x * hc(x)
         + l5 / 2 * (x**2 + hc(x)**2))
    Ye = sp.IndexedBase("Ye")
    L = Lagrangian().add(-V, sector="potential")
    L.add(yukawa(Ye, Ll, H1, eR), sector="yukawa")
    L.add_kinetic(Ll, eR)                     # currents are ignored
    model = Model("thdm", gauge_groups=[SU2L, U1Y], fields=[H1, H2, Ll, eR],
                  lagrangian=L)

    cov = model.term_coverage()
    present = {t.label: c for t, c in cov.present}
    assert present == {
        "(H1†H1)": m11, "(H1†H2)": -m12, "(H2†H2)": m22,
        "(H1†H1) (H1†H1)": l1 / 2, "(H1†H1) (H2†H2)": l3,
        "(H1†H2) (H1†H2)": l5 / 2, "(H1†H2) (H2†H1)": l4,
        "(H2†H2) (H2†H2)": l2 / 2, "Ll̄·H1 eR": Ye[FL_I, FL_J]}
    assert [t.label for t in cov.missing] == [
        "(H1†H1) (H1†H2)", "(H1†H2) (H2†H2)", "Ll̄·H2 eR"]
    assert cov.unmatched == {} and not cov.complete
    assert [t.label for t in model.missing_terms()] == \
        [t.label for t in cov.missing]
    assert "MISSING  (H1†H1) (H1†H2) + h.c." in str(cov)

    # declaring the Z2 (with e_R even) makes the same Lagrangian complete
    # except for the soft term, which the Z2 forbids: it shows as unmatched
    Z2 = ZN("Z2", 2)
    Z2.assign(0, H1)
    Z2.assign(1, H2)
    z2_model = Model("thdm_z2", gauge_groups=[SU2L, U1Y], discrete_groups=[Z2],
                     fields=[H1, H2, Ll, eR], lagrangian=L)
    cov2 = z2_model.term_coverage()
    assert cov2.complete and len(cov2.present) == 8
    assert sp.expand(cov2.unmatched["potential"] + m12 * (x + hc(x))) == 0


def test_coverage_of_the_most_general_model_is_complete():
    SU2L, U1Y, params = _groups()
    H1, H2 = _two_doublets(SU2L, U1Y)
    Ll, eR = _leptons(SU2L, U1Y, 1)
    model = Model.most_general("G", [H1, H2, Ll, eR], [SU2L, U1Y],
                               parameters=params)
    cov = model.term_coverage()
    assert cov.complete and cov.unmatched == {}
    # each operator comes back with its own coupling
    assert {str(c) for _t, c in cov.present} == {
        str(c.name) for c, _t in model.general.terms}


def test_coverage_of_the_weinberg_operator_with_a_flavour_matrix():
    """ν_iᵀCP_Lν_j is symmetric in i↔j, so only the symmetric part of c_ij
    is physical: it must come back as the coefficient, with no remainder
    (the antisymmetric part used to show up as an unmatched term)."""
    SU2L, U1Y, _ = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    Ll, eR = _leptons(SU2L, U1Y, 2)
    c, Lam = sp.IndexedBase("c"), sp.Symbol("Lam", positive=True)
    model = Model("W5", gauge_groups=[SU2L, U1Y], fields=[H, Ll],
                  lagrangian=Lagrangian().add(weinberg(c, Ll, H) / Lam,
                                              sector="other"))
    cov = model.term_coverage(max_dim=5)
    assert cov.unmatched == {}
    (term, coeff), = [(t, k) for t, k in cov.present if t.dim == 5]
    assert sp.simplify(coeff - (c[FL_I, FL_J] + c[FL_J, FL_I]) / (2 * Lam)) == 0
    # the same operator with its dummy indices relabelled reads the same
    swapped = weinberg(c, Ll, H, indices=(FL_J, FL_I)) / Lam
    model2 = Model("W5b", gauge_groups=[SU2L, U1Y], fields=[H, Ll],
                   lagrangian=Lagrangian().add(swapped, sector="other"))
    cov2 = model2.term_coverage(max_dim=5)
    assert cov2.unmatched == {}
    (_t, coeff2), = [(t, k) for t, k in cov2.present if t.dim == 5]
    assert sp.simplify(coeff2 - coeff) == 0


def test_coverage_flags_a_complex_coupling_and_a_noninvariant_term():
    SU2L, U1Y, _ = _groups()
    H1, H2 = _two_doublets(SU2L, U1Y)
    x = inner(H1, H2)
    lam5 = sp.Symbol("lam5")                       # complex
    V = inner(H1, H1) + (lam5 * x**2 + sp.conjugate(lam5) * hc(x)**2) / 2
    model = Model("cpv", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                  lagrangian=Lagrangian().add(-V, sector="potential"))
    cov = model.term_coverage()
    present = {t.label: c for t, c in cov.present}
    # the real part sits on (x² + h.c.); the imaginary part is the CP-odd
    # combination i(x² − x†²), which the basis does not contain
    assert sp.simplify(present["(H1†H2) (H1†H2)"]
                       - (lam5 + sp.conjugate(lam5)) / 4) == 0
    rest = cov.unmatched["potential"]
    assert sp.simplify(rest - (lam5 - sp.conjugate(lam5)) / 4
                       * (x**2 - hc(x)**2)) == 0

    bad = Model("bad", gauge_groups=[SU2L, U1Y], fields=[H1, H2],
                lagrangian=Lagrangian().add(
                    -(inner(H1, H1) + H1.H10 * H2.H20), sector="potential"))
    assert sp.expand(bad.term_coverage().unmatched["potential"]
                     - H1.H10 * H2.H20) == 0


# ------------------------------------------------- explained failures

def test_invariance_failures_explain_themselves():
    ew = electroweak_scaffold()
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    Ll, eR = _leptons(SU2L, U1Y, 1)
    y = ExternalParameter("y", 0.1)
    L = Lagrangian()
    ew.add_higgs(L)
    bad = y * H.H0**2 * Bilinear(Ll.eLbar[FL_I], diracPR, eR.eRc[FL_I])
    L.add(bad, sector="yukawa", name="bad yukawa")
    L.add(yukawa(y, Ll, H, eR, hc=False), sector="yukawa", name="no hc")
    # gauge bosons deliberately NOT listed in fields
    model = Model("x", gauge_groups=ew.gauge_groups, fields=[H, Ll, eR],
                  parameters=ew.parameters + [y], lagrangian=L)
    report = model.check_invariance()
    assert not report.ok and len(report.hints) == len(report.failures)
    text = report.explain()
    assert "add SU2L.bosons() to the model's fields" in text
    assert ("[gauge:U1Y] bad yukawa: U1Y charges do not cancel: "
            "H0**2*Bilinear(eLbar[fl_i], PR, eRc[fl_i]) has charge 1/2") in text
    assert "[gauge:SU2L] bad yukawa: the SU2L indices are not contracted" in text
    assert "mass dimension 5 exceeds 4" in text
    assert "missing its '+ h.c.'" in text
    with pytest.raises(ValueError, match="charges do not cancel"):
        report.raise_on_failure()

    # the failures list keeps its (term, name, details) shape
    term, name, details = report.failures[0]
    assert name.startswith("gauge:")

    good = Model("ok", gauge_groups=ew.gauge_groups, fields=ew.fields,
                 parameters=ew.parameters,
                 lagrangian=ew.add_higgs(Lagrangian()))
    assert good.check_invariance().explain() == "all invariance checks passed"


def test_u1_charge_breakdown():
    SU2L, U1Y, _ = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    Ll, eR = _leptons(SU2L, U1Y, 1)
    fields = [H, Ll, eR]
    assert u1_charge_breakdown(inner(H, H)**2, fields, U1Y) == []
    assert u1_charge_breakdown(yukawa(sp.Symbol("y"), Ll, H, eR), fields,
                               U1Y) == []
    lam = sp.Symbol("lam")
    charged = u1_charge_breakdown(lam * H.H0**3 * sp.conjugate(H.Gp), fields,
                                  U1Y)
    assert charged == [(H.H0**3 * sp.conjugate(H.Gp), 1)]
    # a bar leg counts with the opposite sign: L̄ e_R alone has charge −1/2
    b = Bilinear(Ll.eLbar[FL_I], diracPR, eR.eRc[FL_I])
    assert u1_charge_breakdown(b, fields, U1Y) == [(b, -half)]


# ------------------------------------- Majorana masses and global U(1)s

def _sm_nuR(n=1):
    from feynlag import majorana_mass
    SU2L, U1Y, params = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    Ll, eR = _leptons(SU2L, U1Y, n)
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=n,
                      component_names=["nR"])
    return SU2L, U1Y, params, H, Ll, eR, nuR, majorana_mass


def test_coverage_recognises_the_majorana_mass():
    """FG-10: −½ M_R ν_Rᵀ C P_L ν_R + h.c. is an invariant of the basis,
    with coefficient M_R, not an unmatched remainder."""
    SU2L, U1Y, params, H, Ll, eR, nuR, majorana_mass = _sm_nuR()
    yv, MR = sp.symbols("yv MR", positive=True)
    L = Lagrangian()
    L.add(yukawa(yv, Ll, tilde(H), nuR), sector="yukawa")
    L.add(majorana_mass(MR, nuR), sector="other")
    model = Model("ss", gauge_groups=[SU2L, U1Y], fields=[H, Ll, eR, nuR],
                  parameters=params, lagrangian=L)
    cov = model.term_coverage()
    assert cov.unmatched == {}
    present = {t.label: c for t, c in cov.present}
    assert present["½ nuRᵀCnuR [Majorana]"] == MR
    assert present["Ll̄·H̃ nuR"] == yv


def test_most_general_contains_the_majorana_mass():
    from feynlag import majorana_mass
    SU2L, U1Y, params, H, Ll, eR, nuR, _ = _sm_nuR(3)
    general = most_general_lagrangian(
        [H, Ll, eR, nuR], [SU2L, U1Y], kinetic=False,
        names={"½ nuRᵀCnuR [Majorana]": "MR"})
    by_name = {str(getattr(c, "name", c)): (c, t) for c, t in general.terms}
    MR, term = by_name["MR"]
    assert isinstance(MR, sp.IndexedBase) and term.dim == 3
    built = {t.name: t.expr for t in general.lagrangian.terms}
    assert sp.expand(expand_bilinear(
        built[term.label] - majorana_mass(MR, nuR))) == 0


def test_coverage_imposes_the_declared_global_u1():
    """FG-13: with a GlobalU1 declared, no operator carrying net charge is
    reported missing (the flavon's φ², φH†H, … are forbidden, not absent)."""
    from feynlag import GlobalU1
    from feynlag.invariance import check_global_invariance
    SU2L, U1Y, params = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    phi = Scalar("phi", reps={}, component_names=["phi"])
    G = GlobalU1("U1_FN").assign(1, phi)
    m2, lam = sp.symbols("m2 lamphi", real=True)
    V = m2 * phi.components[0] * sp.conjugate(phi.components[0]) \
        + lam * (phi.components[0] * sp.conjugate(phi.components[0])) ** 2
    kw = dict(gauge_groups=[SU2L, U1Y], fields=[H, phi], parameters=params,
              lagrangian=Lagrangian().add(-V, sector="potential"))
    blind = Model("fn0", **kw).term_coverage()
    assert any(not check_global_invariance(t.expr, G)[0] for t in blind.missing)
    cov = Model("fn", global_groups=[G], **kw).term_coverage()
    assert cov.missing and all(check_global_invariance(t.expr, G)[0]
                               for t in cov.missing)
    assert {t.label for t, _ in cov.present} == {"phi phi†", "phi phi phi† phi†"}
    # Model.most_general imposes it too
    gen = Model.most_general("fng", [H, phi], [SU2L, U1Y], parameters=params,
                             global_groups=[G])
    assert all(check_global_invariance(t.expr, G)[0]
               for _c, t in gen.general.terms)


def test_coverage_keeps_flavour_dependent_global_operator():
    """term_coverage with a GlobalU1 whose quark charges depend on the
    flavour: the symbolic Q̄ H̃ u operator stays in the basis (here present,
    since the model writes it at symbolic indices)."""
    from feynlag import GlobalU1
    SU2L, U1Y, params = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6)},
                     chirality="L", nflavors=3, component_names=["uL", "dL"])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3)}, chirality="R",
                     nflavors=3, component_names=["uR"])
    G = GlobalU1("U1_FN").assign((3, 2, 0), QL).assign((3, 1, 0), uR)
    Yu = sp.IndexedBase("Yu")
    model = Model("fnq", gauge_groups=[SU2L, U1Y], fields=[H, QL, uR],
                  parameters=params, global_groups=[G],
                  lagrangian=Lagrangian().add(yukawa(Yu, QL, tilde(H), uR),
                                              sector="yukawa"))
    cov = model.term_coverage()
    present = {t.label: c for t, c in cov.present}
    assert present["QL̄·H̃ uR"] == Yu[FL_I, FL_J]
    assert cov.unmatched == {}

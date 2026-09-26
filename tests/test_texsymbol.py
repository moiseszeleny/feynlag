"""Per-symbol LaTeX: ``component_tex=`` on fields and ``tex=`` on parameters.

Pinned:
- a tex'd symbol is a :class:`TexSymbol` whose identity includes the tex
  (stable hash, pickle round-trip, never equal to the plain ``Symbol``);
- plain ``sympy.latex`` renders tex'd scalar/fermion/gauge components,
  bar legs (one ``\\overline``), ``conjugate_pair`` partners and parameters;
- without tex the components are exactly the plain objects as before;
- the pipeline is blind to the tex: a tex'd SM Higgs + lepton model is
  invariant, gives ``m_W² = g²v²/4``, ``m_h² = 2λv²``, ``hhh = −3i m_h²/v`` and
  the Yukawa mass matrix ``Y v/√2``.
"""

import pickle

import pytest
import sympy as sp

from feynlag import (
    Bilinear, Dmu, ExternalParameter, GaugeBoson, InternalParameter,
    Lagrangian, Model, SU2, Scalar, TexSymbol, U1, Vacuum, WeylFermion,
    conjugate_pair, dag, diracPL, diracPR, fermion_mass_matrix, tex_symbol,
)

i, j = sp.symbols("fl_i fl_j", integer=True)


class TestTexSymbol:
    def test_identity(self):
        a = TexSymbol("H0", "H^0", real=True)
        assert a == TexSymbol("H0", "H^0", real=True)
        assert hash(a) == hash(TexSymbol("H0", "H^0", real=True))
        assert a != sp.Symbol("H0", real=True)
        assert a != TexSymbol("H0", "H_0", real=True)
        assert a.name == str(a) == "H0" and a.is_real

    def test_uncached_tex(self):
        """Symbol's constructor cache must not leak one call's tex into another."""
        assert sp.latex(TexSymbol("x", "X_1")) == "X_1"
        assert sp.latex(TexSymbol("x", "X_2")) == "X_2"

    def test_pickle_and_srepr_round_trip(self):
        a = TexSymbol("W_1", "W^1", real=True)
        b = pickle.loads(pickle.dumps(a))
        assert b == a and sp.latex(b) == "W^1" and b.is_real
        assert eval(sp.srepr(a), {"TexSymbol": TexSymbol}) == a

    def test_sorts_like_symbol(self):
        """A tex'd name must not reorder vertex legs or printed terms."""
        from sympy import default_sort_key
        h, v = sp.Symbol("h", real=True), sp.Symbol("v", positive=True)
        Gp, Gm = TexSymbol("Gp", "G^+"), TexSymbol("Gm", "G^-")
        lam = TexSymbol("lam", r"\lambda", real=True)
        assert sorted([h, Gp, Gm], key=default_sort_key) == [Gm, Gp, h]
        assert str(lam * v**2) == "lam*v**2"
        assert sp.latex(lam * v**2) == r"\lambda v^{2}"

    def test_tex_symbol_switch(self):
        assert type(tex_symbol("x")) is sp.Symbol
        assert type(tex_symbol("x", "X")) is TexSymbol


@pytest.fixture
def ew():
    gw = ExternalParameter("gw", 0.65, positive=True, tex="g")
    g1 = ExternalParameter("g1", 0.36, positive=True, tex="g'")
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1), gw, g1


class TestRendering:
    def test_scalar_doublet(self, ew):
        SU2L, U1Y, *_ = ew
        H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                   component_names=["Gp", "H0"], component_tex=["G^+", "H^0"])
        assert sp.latex(H[0]) == "G^+"
        assert sp.latex((dag(H) * H.mat)[0]) == \
            r"G^+ \overline{G^+} + H^0 \overline{H^0}"
        assert H._repr_latex_() == r"$\displaystyle H = (G^+, H^0)$"

    def test_fermion_bilinear(self, ew):
        SU2L, U1Y, *_ = ew
        L = WeylFermion("L", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                        nflavors=3, component_names=["nuL", "eL"],
                        component_tex=[r"\nu_L", "e_L"])
        nuLbar, eLbar = L.bar_components
        assert sp.latex(Bilinear(nuLbar[i], diracPL, L[1][j])) == \
            r"\overline{{\nu_L}_{fl_{i}}}\,P_L\,{e_L}_{fl_{j}}"
        assert sp.latex(eLbar[i]) == r"{\overline{e_L}}_{fl_{i}}"

    def test_singlet_uses_field_tex(self):
        nuR = WeylFermion("nuR", reps={}, chirality="R", tex=r"\nu_R")
        assert sp.latex(nuR[0]) == r"\nu_R"

    def test_gauge_boson(self, ew):
        SU2L, *_ = ew
        W = GaugeBoson("W", SU2L, component_tex=["W^1", "W^2", "W^3"])
        assert all(c.is_real for c in W.components)
        assert sp.latex(W[2]) == "W^3"

    def test_conjugate_pair(self):
        Gp = TexSymbol("Gp", "G^+")
        Gm, cmap = conjugate_pair(Gp, "Gm", tex="G^-")
        assert sp.latex(Gm) == "G^-" and cmap == {sp.conjugate(Gp): Gm}

    def test_parameter(self):
        lam = ExternalParameter("lam", 0.13, tex=r"\lambda")
        assert sp.latex(lam.symbol) == r"\lambda" and lam.symbol.is_real
        assert sp.latex(InternalParameter("mu2", tex=r"\mu^2").s) == r"\mu^2"

    def test_wrong_length_raises(self, ew):
        SU2L, *_ = ew
        with pytest.raises(ValueError, match="component tex"):
            Scalar("H", reps={SU2L: 2}, component_tex=["H^0"])

    def test_default_path_unchanged(self, ew):
        SU2L, U1Y, *_ = ew
        H = Scalar("H", reps={SU2L: 2}, component_names=["Gp", "H0"])
        assert all(type(c) is sp.Symbol for c in H.components)
        assert H[1] == sp.Symbol("H0")
        L = WeylFermion("L", reps={SU2L: 2}, component_names=["nuL", "eL"])
        assert L[0] == sp.IndexedBase("nuL")
        assert L.bar_components[0] == sp.IndexedBase("nuLbar")
        assert ExternalParameter("lam", 0.1).symbol == sp.Symbol("lam", real=True)


class TestPipelineBlindToTex:
    @pytest.fixture
    def model(self, ew):
        SU2L, U1Y, gw, g1 = ew
        v = ExternalParameter("v", 246.0, positive=True, unit_dim=1, tex="v")
        lam = ExternalParameter("lam", 0.129, tex=r"\lambda")
        mu2 = InternalParameter("mu2", unit_dim=2, tex=r"\mu^2")
        H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                   component_names=["Gp", "H0"], component_tex=["G^+", "H^0"])
        H.expand_vev({H[1]: v})
        W = SU2L.bosons("W", component_tex=["W^1", "W^2", "W^3"])
        B = U1Y.bosons("B", tex="B")
        Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                         nflavors=3, component_names=["nuL", "eL"],
                         component_tex=[r"\nu_L", "e_L"])
        eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                         tex="e_R")
        Y = sp.IndexedBase("Ye")
        (nuLbar, eLbar), eRbar = Ll.bar_components, eR.bar_components[0]
        Gp, H0 = H.components
        yuk = -(Y[i, j] * Gp * Bilinear(nuLbar[i], diracPR, eR[0][j])
                + Y[i, j] * H0 * Bilinear(eLbar[i], diracPR, eR[0][j]))
        yuk += -(sp.conjugate(Y[i, j]) * sp.conjugate(Gp)
                 * Bilinear(eRbar[j], diracPL, Ll[0][i])
                 + sp.conjugate(Y[i, j]) * sp.conjugate(H0)
                 * Bilinear(eRbar[j], diracPL, Ll[1][i]))
        HdH = (dag(H) * H.mat)[0]
        L = (Lagrangian()
             .add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
             .add(mu2.s * HdH - lam.s * HdH**2, sector="potential")
             .add(yuk, sector="yukawa"))
        model = Model("tex-SM", gauge_groups=[SU2L, U1Y],
                      fields=[H, W, B, Ll, eR],
                      parameters=[gw, g1, v, lam, mu2], lagrangian=L)
        return model, H, W, Ll, eR, Y, v, lam, mu2, gw

    def test_invariance(self, model):
        report = model[0].check_invariance()
        assert report.ok, report.failures

    def test_masses_and_hhh(self, model):
        model, H, W, Ll, eR, Y, v, lam, mu2, gw = model
        model.solve_tadpoles([mu2])
        h = sp.Symbol("H0_r", real=True)
        assert sp.simplify(model.mass_matrix([h])[0, 0]
                           - 2 * lam.s * v.s**2) == 0
        assert sp.simplify(model.mass_matrix([H[0]], charged=True)[0, 0]) == 0
        MV = model.gauge_mass_matrix(W.components)
        assert sp.simplify(MV[0, 0] - gw.s**2 * v.s**2 / 4) == 0
        (vtx,) = [x for x in model.vertices([h], sector="potential")
                  if x.particles == (h, h, h)]
        mh2 = 2 * lam.s * v.s**2
        assert sp.simplify(vtx.coupling + 3 * sp.I * mh2 / v.s) == 0

    def test_yukawa_mass_matrix(self, model):
        model, H, W, Ll, eR, Y, v, *_ = model
        M = fermion_mass_matrix(model.lagrangian.sector("yukawa"),
                                Ll.bar_components[1], eR[0], Vacuum([H]), 3,
                                (i, j), gamma=diracPR)
        for a in range(3):
            for b in range(3):
                assert sp.simplify(M[a, b] - Y[a, b] * v.s / sp.sqrt(2)) == 0


class TestOverridesAndBuilders:
    def test_symbol_names_override_wins(self):
        a = TexSymbol("Gp", "G^+")
        assert sp.latex(a, symbol_names={a: "X"}) == "X"
        assert sp.latex(a, symbol_names={sp.Symbol("y"): "Y"}) == "G^+"

    def test_bosons_cache_rejects_conflicting_tex(self, ew):
        SU2L, *_ = ew
        W = SU2L.bosons("W", component_tex=["W^1", "W^2", "W^3"])
        assert SU2L.bosons() is W
        assert SU2L.bosons(component_tex=["W^1", "W^2", "W^3"]) is W
        with pytest.raises(ValueError, match="already created"):
            SU2L.bosons(component_tex=["A", "B", "C"])

    def test_electroweak_scaffold_opt_in(self):
        from feynlag.models import electroweak_scaffold
        plain = electroweak_scaffold()
        assert all(type(c) is sp.Symbol
                   for F in plain.fields for c in F.components)
        ew = electroweak_scaffold(higgs_tex=["G^+", "H^0"],
                                  w_tex=["W^1", "W^2", "W^3"], b_tex="B")
        assert [sp.latex(c) for c in ew.H.components] == ["G^+", "H^0"]
        assert sp.latex(ew.W[0]) == "W^1" and sp.latex(ew.B[0]) == "B"

    def test_to_physical_basis_gm_tex(self):
        from feynlag.models import electroweak_scaffold, to_physical_basis
        ew = electroweak_scaffold(higgs_tex=["G^+", "H^0"])
        L = ew.add_higgs(Lagrangian())
        model = Model("sm", gauge_groups=ew.gauge_groups, fields=ew.fields,
                      parameters=ew.parameters, lagrangian=L)
        pb = to_physical_basis(model, ew, gm_tex="G^-")
        assert sp.latex(pb.Gm) == "G^-"
        assert pb.cmap == {sp.conjugate(ew.H[0]): pb.Gm}

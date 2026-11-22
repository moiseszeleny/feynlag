"""Field-level term constructors (``feynlag.terms``).

Every constructor must return *exactly* the expression the component-level
code produces — the references below are the hand-written terms of
``examples/sm_vll.py``, ``sm_scalar_gauge.py``, ``sm_seesaw.py``,
``sm_weinberg.py`` and ``thdm.py``.
"""

import pytest
import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, InternalParameter, Lagrangian,
    Dmu, MajoranaBilinear, Model, PartialMu, SU2, SU3, Scalar, U1, WeylFermion,
    dag, diracC,
    diracPL, diracPR, expand_bilinear, fermion_gauge_current, hc,
    majorana_mass_matrix,
)
from feynlag.suggest import suggest_kinetic
from feynlag.terms import (
    FL_I, FL_J, dirac_mass, eps, inner, kinetic, majorana_mass, tilde,
    weinberg, yukawa,
)

half = sp.Rational(1, 2)


def _same(a, b):
    return sp.expand(expand_bilinear(sp.expand(a - b))) == 0


@pytest.fixture
def ew():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: half}, component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    return SU2L, U1Y, H, [gw, g1, v]


def _leptons(SU2L, U1Y, n=1):
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -half}, chirality="L",
                     nflavors=n, component_names=["nuL", "eL"])
    eR = WeylFermion("eRf", reps={U1Y: -1}, chirality="R", nflavors=n,
                     component_names=["eR"])
    return Ll, eR


# ---------------------------------------------------------------- yukawa

def test_yukawa_matches_handwritten_lepton_term(ew):
    """−y L̄ H e_R + h.c. with a flavour-universal coupling (sm_vll.py)."""
    SU2L, U1Y, H, _ = ew
    Ll, eRf = _leptons(SU2L, U1Y)
    ye = ExternalParameter("ye", 0.01, positive=True)
    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    eR, eRbar = eRf.components[0], eRf.bar_components[0]
    i = FL_I
    ref = -(ye.s * Gp * Bilinear(nuLbar[i], diracPR, eR[i])
            + ye.s * H0 * Bilinear(eLbar[i], diracPR, eR[i]))
    ref += -(ye.s * sp.conjugate(Gp) * Bilinear(eRbar[i], diracPL, nuL[i])
             + ye.s * sp.conjugate(H0) * Bilinear(eRbar[i], diracPL, eL[i]))
    assert _same(yukawa(ye, Ll, H, eRf), ref)
    # hc=False is the first half only
    assert _same(yukawa(ye, Ll, H, eRf, hc=False),
                 -(ye.s * Gp * Bilinear(nuLbar[i], diracPR, eR[i])
                   + ye.s * H0 * Bilinear(eLbar[i], diracPR, eR[i])))


def test_yukawa_flavour_matrix_colour_sum_and_tilde(ew):
    """Quark Yukawas of sm_scalar_gauge.py: Y[i,j], colour δ-sum, H̃."""
    SU2L, U1Y, H, _ = ew
    SU3c = SU3("SU3c", coupling=ExternalParameter("gs", 1.2, positive=True))
    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3},
                     chirality="L", nflavors=3,
                     component_names=["uL_1", "uL_2", "uL_3",
                                      "dL_1", "dL_2", "dL_3"])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["uR_1", "uR_2", "uR_3"])
    dR = WeylFermion("dR", reps={U1Y: -sp.Rational(1, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["dR_1", "dR_2", "dR_3"])
    Yu, Yd = sp.IndexedBase("Yu"), sp.IndexedBase("Yd")
    i, j = FL_I, FL_J
    Gp, H0 = H.components
    qL_u, qL_d = QL.components[:3], QL.components[3:]
    qLbar_u, qLbar_d = QL.bar_components[:3], QL.bar_components[3:]
    uRc, uRbar = uR.components, uR.bar_components
    dRc, dRbar = dR.components, dR.bar_components

    def qcolor(bar_list, gamma, field_list, idx1, idx2):
        return sum(Bilinear(bar_list[c][idx1], gamma, field_list[c][idx2])
                   for c in range(3))

    ref_d = -(Yd[i, j] * Gp * qcolor(qLbar_u, diracPR, dRc, i, j)
              + Yd[i, j] * H0 * qcolor(qLbar_d, diracPR, dRc, i, j))
    ref_d += -(sp.conjugate(Yd[i, j]) * sp.conjugate(Gp)
               * qcolor(dRbar, diracPL, qL_u, j, i)
               + sp.conjugate(Yd[i, j]) * sp.conjugate(H0)
               * qcolor(dRbar, diracPL, qL_d, j, i))
    ref_u = -(Yu[i, j] * sp.conjugate(H0) * qcolor(qLbar_u, diracPR, uRc, i, j)
              + Yu[i, j] * (-sp.conjugate(Gp))
              * qcolor(qLbar_d, diracPR, uRc, i, j))
    ref_u += -(sp.conjugate(Yu[i, j]) * H0 * qcolor(uRbar, diracPL, qL_u, j, i)
               + sp.conjugate(Yu[i, j]) * (-Gp)
               * qcolor(uRbar, diracPL, qL_d, j, i))

    assert _same(yukawa(Yd, QL, H, dR), ref_d)
    assert _same(yukawa(Yu, QL, tilde(H), uR), ref_u)


def test_yukawa_custom_indices(ew):
    SU2L, U1Y, H, _ = ew
    Ll, eRf = _leptons(SU2L, U1Y, n=3)
    a, b = sp.symbols("a b", integer=True)
    Ye = sp.IndexedBase("Ye")
    term = yukawa(Ye, Ll, H, eRf, indices=(a, b))
    assert term.has(Ye[a, b]) and not term.has(FL_I)
    with pytest.raises(ValueError, match="two distinct leg indices"):
        yukawa(Ye, Ll, H, eRf, indices=a)


def test_yukawa_singlet_scalar(ew):
    """A gauge-singlet scalar couples to a vector-like pair: −y S Ψ̄_L Ψ_R."""
    SU2L, U1Y, _, _ = ew
    S = Scalar("S", reps={}, real=True)
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -half}, chirality="L",
                       component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -half}, chirality="R",
                       component_names=["NR", "ER"])
    y = sp.Symbol("yS", positive=True)
    s0 = S.components[0]
    i = FL_I
    NLbar, ELbar = PsiL.bar_components
    NR, ER = PsiR.components
    half_term = -y * s0 * (Bilinear(NLbar[i], diracPR, NR[i])
                           + Bilinear(ELbar[i], diracPR, ER[i]))
    assert _same(yukawa(y, PsiL, S, PsiR), half_term + sp.conjugate(half_term))


def test_yukawa_wrong_hypercharge_names_the_sum_and_the_fix(ew):
    SU2L, U1Y, H, _ = ew
    Ll, eRf = _leptons(SU2L, U1Y)
    y = sp.Symbol("y", positive=True)
    # L̄ H̃ e_R: −(−1/2) − 1/2 − 1 = −1
    with pytest.raises(ValueError) as err:
        yukawa(y, Ll, tilde(H), eRf)
    msg = str(err.value)
    assert "U1Y sums to -1" in msg
    assert "balance with H instead" in msg
    # and the mirror image: a neutral singlet needs H̃
    nuR = WeylFermion("nuR", reps={}, chirality="R", component_names=["nuR"])
    with pytest.raises(ValueError, match=r"balance with tilde\(H\) instead"):
        yukawa(y, Ll, H, nuR)


def test_yukawa_rejects_unsupported_reps_and_wrong_order(ew):
    SU2L, U1Y, H, _ = ew
    Ll, eRf = _leptons(SU2L, U1Y)
    y = sp.Symbol("y", positive=True)
    Delta = Scalar("Delta", reps={SU2L: 3, U1Y: 1})
    with pytest.raises(NotImplementedError, match="component level"):
        yukawa(y, Ll, Delta, eRf)
    with pytest.raises(ValueError, match="left-handed"):
        yukawa(y, eRf, H, Ll)


# ------------------------------------------------------- masses, Weinberg

def test_dirac_mass_matches_handwritten_vll_mass(ew):
    SU2L, U1Y, _, _ = ew
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -half}, chirality="L",
                       component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -half}, chirality="R",
                       component_names=["NR", "ER"])
    M = ExternalParameter("MPsi", 1000.0, positive=True, unit_dim=1)
    NL, EL = PsiL.components
    NLbar, ELbar = PsiL.bar_components
    NR, ER = PsiR.components
    NRbar, ERbar = PsiR.bar_components
    i = FL_I
    ref = -M.s * (Bilinear(NLbar[i], diracPR, NR[i])
                  + Bilinear(ELbar[i], diracPR, ER[i]))
    ref += -M.s * (Bilinear(NRbar[i], diracPL, NL[i])
                   + Bilinear(ERbar[i], diracPL, EL[i]))
    assert _same(dirac_mass(M, PsiL, PsiR), ref)

    Ll, eRf = _leptons(SU2L, U1Y)
    with pytest.raises(ValueError, match="different non-abelian"):
        dirac_mass(M, Ll, eRf)


def test_majorana_mass_matches_seesaw_example_and_mass_matrix(ew):
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=2,
                      component_names=["nuR"])
    nR = nuR.components[0]
    CPL = diracC * diracPL
    i, j = FL_I, FL_J

    MR = ExternalParameter("MR", 1.0e3, positive=True, unit_dim=1)
    op = -half * MR.s * MajoranaBilinear(nR[i], CPL, nR[i])
    assert _same(majorana_mass(MR, nuR), op + sp.conjugate(op))

    Mmat = sp.IndexedBase("M")
    term = majorana_mass(Mmat, nuR)
    op2 = -half * Mmat[i, j] * MajoranaBilinear(nR[i], CPL, nR[j])
    assert _same(term, op2 + sp.conjugate(op2))

    # a flavour-universal M gives M·1 through the ordinary mass-matrix reader
    SU2L, U1Y, H, params = ew
    L = Lagrangian().add(majorana_mass(MR, nuR), sector="other")
    model = Model("maj", gauge_groups=[SU2L, U1Y], fields=[H, nuR],
                  parameters=params + [MR], lagrangian=L)
    M = majorana_mass_matrix(majorana_mass(MR, nuR), nR, model.vacuum, 2,
                             (i, j), gamma=CPL)
    assert M == MR.s * sp.eye(2)

    Ll, _ = _leptons(SU2L, U1Y)
    with pytest.raises(ValueError, match="Weinberg"):
        majorana_mass(MR, Ll)


def test_weinberg_matches_handwritten_operator_and_neutrino_mass(ew):
    """c (LεH)(LεH) + h.c. reproduces sm_weinberg.py and m_ν = −c v²/Λ."""
    SU2L, U1Y, H, params = ew
    v = params[2]
    Ll, _ = _leptons(SU2L, U1Y, n=2)
    Lam = ExternalParameter("Lam", 1.0e14, positive=True, unit_dim=1)
    c = sp.IndexedBase("c")
    i, j = FL_I, FL_J
    Gp, H0 = H.components
    nuL, eL = Ll.components
    CPL = diracC * diracPL
    epsH = [H0, -Gp]
    comp = [nuL, eL]
    O = sp.S.Zero
    for a in range(2):
        for b in range(2):
            O += (c[i, j] / Lam.s) * epsH[a] * epsH[b] * \
                MajoranaBilinear(comp[a][i], CPL, comp[b][j])
    O = sp.expand(O)
    assert _same(weinberg(c, Ll, H, hc=False) / Lam.s, O)
    term = weinberg(c, Ll, H) / Lam.s
    assert _same(term, O + sp.conjugate(O))

    cs = sp.Symbol("cW", positive=True)
    L = Lagrangian().add(weinberg(cs / Lam.s, Ll, H), sector="other")
    model = Model("W", gauge_groups=[SU2L, U1Y], fields=[H, Ll],
                  parameters=params + [Lam], lagrangian=L)
    M = majorana_mass_matrix(L.total, nuL, model.vacuum, 2, (i, j), gamma=CPL)
    assert sp.simplify(M - (-cs * v.s**2 / Lam.s) * sp.eye(2)) == sp.zeros(2, 2)

    with pytest.raises(ValueError, match="not U\\(1\\) invariant"):
        weinberg(cs, Ll, Scalar("Hm", reps={SU2L: 2, U1Y: -half}))


# ------------------------------------------------------- scalar invariants

def test_inner_and_eps_rebuild_the_thdm_potential(ew):
    SU2L, U1Y, _, _ = ew
    H1 = Scalar("H1", reps={SU2L: 2, U1Y: half}, component_names=["H1p", "H10"])
    H2 = Scalar("H2", reps={SU2L: 2, U1Y: half}, component_names=["H2p", "H20"])
    m11, m22, m12 = sp.symbols("m11sq m22sq m12sq", real=True)
    l1, l2, l3, l4, l5 = sp.symbols("lam1:6", real=True)

    H1dH1 = (dag(H1) * H1.mat)[0]
    H2dH2 = (dag(H2) * H2.mat)[0]
    H1dH2 = (dag(H1) * H2.mat)[0]
    ref = (m11 * H1dH1 + m22 * H2dH2
           - m12 * (H1dH2 + sp.conjugate(H1dH2))
           + l1 / 2 * H1dH1**2 + l2 / 2 * H2dH2**2
           + l3 * H1dH1 * H2dH2 + l4 * H1dH2 * sp.conjugate(H1dH2)
           + l5 / 2 * (H1dH2**2 + sp.conjugate(H1dH2)**2))

    a, b, x = inner(H1, H1), inner(H2, H2), inner(H1, H2)
    V = (m11 * a + m22 * b - m12 * (x + hc(x)) + l1 / 2 * a**2 + l2 / 2 * b**2
         + l3 * a * b + l4 * x * hc(x) + l5 / 2 * (x**2 + hc(x)**2))
    assert sp.expand(V - ref) == 0

    h1p, h10 = H1.components
    h2p, h20 = H2.components
    assert eps(H1, H2) == sp.expand(h1p * h20 - h10 * h2p)
    assert eps(H1, H1) == 0
    # |H1†H2|² + |H1ᵀεH2|² = (H1†H1)(H2†H2)  (the SU(2) Fierz identity)
    e = eps(H1, H2)
    assert sp.expand(x * hc(x) + e * hc(e) - a * b) == 0

    S = Scalar("S", reps={})
    with pytest.raises(ValueError, match="different non-abelian"):
        inner(H1, S)
    with pytest.raises(ValueError, match="doublets"):
        eps(H1, S)


# --------------------------------------------- the constructed model checks

def test_constructed_vll_model_is_invariant(ew):
    """yukawa + dirac_mass + kinetic build an invariant, hermitian model with
    the charged-lepton mass matrix of sm_vll.py."""
    from feynlag import fermion_mass_matrix

    SU2L, U1Y, H, params = ew
    v = params[2]
    Ll, eRf = _leptons(SU2L, U1Y)
    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -half}, chirality="L",
                       component_names=["NL", "EL"])
    PsiR = WeylFermion("PsiR", reps={SU2L: 2, U1Y: -half}, chirality="R",
                       component_names=["NR", "ER"])
    ye = ExternalParameter("ye", 0.01, positive=True)
    lamE = ExternalParameter("lamE", 0.4, positive=True)
    MPsi = ExternalParameter("MPsi", 1000.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)

    LYuk = (yukawa(ye, Ll, H, eRf) + yukawa(lamE, PsiL, H, eRf)
            + dirac_mass(MPsi, PsiL, PsiR))
    L = Lagrangian()
    L.add(-(-mu2.s * inner(H, H) + lam.s * inner(H, H)**2), sector="potential")
    L.add(LYuk, sector="yukawa")
    L.add_kinetic(H, Ll, eRf, PsiL, PsiR)
    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    model = Model("VLL", gauge_groups=[SU2L, U1Y],
                  fields=[H, W, B, Ll, eRf, PsiL, PsiR],
                  parameters=params + [ye, lamE, MPsi, lam, mu2], lagrangian=L)
    report = model.check_invariance()
    assert report.ok, report

    # kinetic() is the same current the examples add by hand
    by_sector = dict((s, e) for e, s in kinetic(Ll))
    assert _same(by_sector["yukawa"], fermion_gauge_current(Ll, FL_I))

    eLbar, ELbar = Ll.bar_components[1], PsiL.bar_components[1]
    eR, ER = eRf.components[0], PsiR.components[1]
    bars, rights = (eLbar, ELbar), (eR, ER)
    M2 = sp.Matrix(2, 2, lambda a, b: fermion_mass_matrix(
        LYuk, bars[a], rights[b], model.vacuum, 1, (FL_I, FL_J),
        gamma=diracPR)[0, 0])
    s2 = sp.sqrt(2)
    assert M2 == sp.Matrix([[ye.s * v.s / s2, 0], [lamE.s * v.s / s2, MPsi.s]])


# ---------------------------------------------------------------- kinetic (FG-8)

class TestRealScalarKinetic:
    """A real scalar's kinetic term is the canonical ½(∂S)² (feynlag-models FG-8).

    ``(D_μφ)†(D^μφ)`` is right for a complex field, but a real field is its own
    conjugate, so the same expression would give ``(∂S)²`` — twice the
    canonical term, silently doubling the propagator normalisation.
    """

    @staticmethod
    def _real_singlet():
        S = Scalar("S", reps={}, real=True)
        return S, half * PartialMu(S.components[0]) ** 2

    def test_kinetic_real_singlet_is_half(self):
        S, ref = self._real_singlet()
        [(expr, sector)] = kinetic(S)
        assert sector == "kinetic"
        assert sp.expand(expr - ref) == 0

    def test_add_kinetic_real_singlet_is_half(self):
        S, ref = self._real_singlet()
        L = Lagrangian().add_kinetic(S)
        assert sp.expand(L.sector("kinetic") - ref) == 0

    def test_most_general_real_singlet_is_half(self, ew):
        SU2L, U1Y, H, params = ew
        S, ref = self._real_singlet()
        model = Model.most_general("xSM", [S], [SU2L, U1Y], parameters=params)
        assert sp.expand(model.lagrangian.sector("kinetic") - ref) == 0
        # the Higgs doublet beside it keeps (DH)†(DH)
        model2 = Model.most_general("xSM2", [H, S], [SU2L, U1Y],
                                    parameters=params)
        DH = Dmu(H)
        full = sp.expand((dag(DH) * DH)[0]) + ref
        assert sp.expand(model2.lagrangian.sector("kinetic") - full) == 0

    def test_suggest_kinetic_label(self):
        S, _ = self._real_singlet()
        [t] = suggest_kinetic([S])
        assert t.label == "½(DμS)(DμS)"

    def test_complex_scalars_unchanged(self, ew):
        SU2L, U1Y, H, _ = ew
        DH = Dmu(H)
        [(expr, _)] = kinetic(H)
        assert sp.expand(expr - (dag(DH) * DH)[0]) == 0
        phi = Scalar("phi", reps={})                  # complex singlet
        p = phi.components[0]
        [(expr, _)] = kinetic(phi)
        assert sp.expand(expr - PartialMu(sp.conjugate(p)) * PartialMu(p)) == 0

    def test_canonical_normalisation_matches_handwritten(self):
        """Same momentum-space two-point coefficient as ½ PartialMu(S)²."""
        from feynlag import to_momentum_space
        S, ref = self._real_singlet()
        [(expr, _)] = kinetic(S)
        f = S.components
        mom = sp.expand(to_momentum_space(expr, f))
        assert mom != 0
        assert sp.expand(mom - to_momentum_space(ref, f)) == 0


# ------------------------------------------- flavour indices (FG-9, FG-11)

class TestIndexRules:
    """A scalar coupling is flavour-universal (δ_ij); a flavour matrix needs
    two leg indices, which may be equal integers."""

    def test_scalar_coupling_with_distinct_pair_raises(self, ew):
        from feynlag.terms import dirac_mass as dm
        SU2L, U1Y, H, _ = ew
        Ll, eRf = _leptons(SU2L, U1Y, 3)
        nuR = WeylFermion("nuRx", reps={}, chirality="R", nflavors=3,
                          component_names=["nRx"])
        y = sp.Symbol("y")
        a, b = sp.symbols("a b", integer=True)
        for pair in [(a, b), (0, 1)]:
            with pytest.raises(ValueError, match="flavour-universal"):
                yukawa(y, Ll, H, eRf, indices=pair)
            with pytest.raises(ValueError, match="flavour-universal"):
                majorana_mass(y, nuR, indices=pair)
            with pytest.raises(ValueError, match="flavour-universal"):
                weinberg(y, Ll, H, indices=pair)
        PsiL = WeylFermion("PsL", reps={}, chirality="L", component_names=["PL"])
        PsiR = WeylFermion("PsR", reps={}, chirality="R", component_names=["PR"])
        with pytest.raises(ValueError, match="flavour-universal"):
            dm(y, PsiL, PsiR, indices=(a, b))

    def test_scalar_coupling_single_or_repeated_index_unchanged(self, ew):
        SU2L, U1Y, H, _ = ew
        Ll, eRf = _leptons(SU2L, U1Y, 3)
        y = sp.Symbol("y")
        a = sp.Symbol("a", integer=True)
        assert _same(yukawa(y, Ll, H, eRf, indices=(a, a)),
                     yukawa(y, Ll, H, eRf, indices=a))
        assert _same(yukawa(y, Ll, H, eRf, indices=(1, 1)),
                     yukawa(y, Ll, H, eRf, indices=1))
        # the default stays the flavour-universal diagonal on FL_I
        assert _same(yukawa(y, Ll, H, eRf), yukawa(y, Ll, H, eRf, indices=FL_I))

    def test_flavour_matrix_integer_diagonal(self, ew):
        SU2L, U1Y, H, _ = ew
        Ll, eRf = _leptons(SU2L, U1Y, 3)
        Y = sp.IndexedBase("Y")
        Gp, H0 = H.components
        nuLbar, eLbar = Ll.bar_components
        eR = eRf.components[0]
        half_term = -Y[0, 0] * (Bilinear(nuLbar[0], diracPR, eR[0]) * Gp
                                + Bilinear(eLbar[0], diracPR, eR[0]) * H0)
        term = yukawa(Y, Ll, H, eRf, indices=(0, 0))
        assert _same(term, half_term + sp.conjugate(half_term))
        assert not term.has(Y[0, 1])
        with pytest.raises(ValueError, match="two distinct leg indices"):
            yukawa(Y, Ll, H, eRf, indices=(FL_I, FL_I))


# ---------------------------------------- real-rep Majorana masses (FG-10)

class TestRealRepMajorana:

    def test_singlet_form_and_term_unchanged(self):
        from feynlag.terms import majorana_form
        nuR = WeylFermion("nRs", reps={}, chirality="R", nflavors=1,
                          component_names=["nRs"])
        assert majorana_form(nuR) == sp.Matrix([[1]])
        M = sp.Symbol("M", positive=True)
        n = nuR.components[0]
        op = -half * M * MajoranaBilinear(n[FL_I], diracC * diracPL, n[FL_I])
        assert _same(majorana_mass(M, nuR, indices=FL_I), op + sp.conjugate(op))

    def test_su2_triplet_y0(self, ew):
        """Type-III Σ (3 of SU(2), Y = 0): −½M Σ^aᵀ C P_L Σ^a + h.c., gauge
        invariant (the adjoint basis has Tᵀ = −T, so B = δ)."""
        from feynlag.terms import majorana_form
        SU2L, U1Y, H, params = ew
        Sig = WeylFermion("Sig", reps={SU2L: 3, U1Y: 0}, chirality="R",
                          component_names=["S1", "S2", "S3"])
        assert majorana_form(Sig) == sp.eye(3)
        M = sp.Symbol("M", positive=True)
        i = FL_I
        op = -half * M * sum(MajoranaBilinear(c[i], diracC * diracPL, c[i])
                             for c in Sig.components)
        term = majorana_mass(M, Sig, indices=i)
        assert _same(term, op + sp.conjugate(op))
        model = Model("sig", gauge_groups=[SU2L, U1Y],
                      fields=[Sig, SU2L.bosons("W")], parameters=params,
                      lagrangian=Lagrangian().add(term, sector="other"))
        assert model.check_invariance(dimension=False).ok

    def test_su2_quintet_gt_basis_form(self, ew):
        """A real rep in the Gelfand–Tsetlin basis: B pairs m with −m with
        alternating signs, and the neutral (middle) entry is +1."""
        from feynlag.terms import majorana_form
        SU2L = ew[0]
        F = WeylFermion("F5", reps={SU2L: 5}, chirality="R")
        B = majorana_form(F)
        assert B.is_symmetric() and B.H * B == sp.eye(5) and B[2, 2] == 1
        for T in F.generators(SU2L):
            assert (T.T * B + B * T).is_zero_matrix

    def test_non_real_reps_raise(self, ew):
        from feynlag.terms import majorana_form
        SU2L, U1Y, H, _ = ew
        D = WeylFermion("Dd", reps={SU2L: 2}, chirality="L",
                        component_names=["D1", "D2"])
        Q = WeylFermion("Qq", reps={SU3("SU3c", coupling=sp.Symbol("gs")): 3},
                        chirality="R")
        M = sp.Symbol("M")
        for F in (D, Q):
            assert majorana_form(F) is None
            with pytest.raises(ValueError, match="not real"):
                majorana_mass(M, F)
        E = WeylFermion("Ech", reps={U1Y: -1}, chirality="R")
        with pytest.raises(ValueError, match="charged under U1Y"):
            majorana_mass(M, E)

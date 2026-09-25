"""Type-I seesaw: mass sector, Majorana rotation, and physical couplings.

Pins the seesaw mechanism end to end — the block mass matrix, the seesaw light
mass ``m_ν = −m_D M_R⁻¹ m_Dᵀ`` (matched against the exact Takagi
diagonalisation), and the physical heavy-neutrino couplings extracted through
the :class:`~feynlag.MajoranaRotation`: ``W ℓ̄ N = (g/√2)·V`` with the
light–heavy mixing ``V ≈ m_D/M_R``, vanishing as ``M_R → ∞`` (decoupling).
"""

import sympy as sp
import pytest

from feynlag import (
    Bilinear, DiracGamma, Dmu, ExternalParameter, InternalParameter, Lagrangian,
    MajoranaBilinear, MajoranaRotation, Model, Rotation, SU2, Scalar, U1,
    WeylFermion, dag, diagonalize_takagi, diracC, diracPL, diracPR,
    extract_fermion_vertices, fermion_gauge_current, fermion_mass_matrix,
    majorana_mass_matrix, rotation_2x2, seesaw_light_mass, seesaw_mass_matrix,
)


def _num(x):
    return complex(sp.N(x))


# --------------------------------------------------------------------------
# mass-sector helpers
# --------------------------------------------------------------------------

class TestNumericTakagi:
    """``diagonalize_takagi`` on a numeric N > 2 matrix goes numeric (mpmath):
    the symbolic ``Matrix.diagonalize`` does not finish on a generic 5×5."""

    @staticmethod
    def _generic_2n():
        """3 ν_L + 2 ν_R: generic 3×2 m_D (rationals and √2), M_R = diag(1, 3) TeV."""
        mD = sp.Matrix([[sp.Rational(3, 100), sp.Rational(-7, 100)],
                        [sp.Rational(11, 100) / sp.sqrt(2), sp.Rational(5, 100)],
                        [sp.Rational(-2, 100), sp.Rational(13, 100) / sp.sqrt(2)]])
        MR = sp.diag(sp.Integer(1000), sp.Integer(3000))
        return mD, MR, seesaw_mass_matrix(mD, MR)

    @staticmethod
    def _maxabs(expr_matrix):
        return max(abs(x) for x in expr_matrix.evalf(45))

    def test_generic_5x5_finishes_and_reconstructs(self):
        import signal

        def timeout(*_):
            raise TimeoutError("diagonalize_takagi did not finish in 10 s")
        mD, MR, M = self._generic_2n()
        assert M.shape == (5, 5)
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(10)
        try:
            U, D = diagonalize_takagi(M)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        assert self._maxabs(U * D * U.T - M) < 1e-40 * 3000
        assert self._maxabs(U * U.conjugate().T - sp.eye(5)) < 1e-40
        masses = [D[k, k] for k in range(5)]
        assert all(m >= 0 for m in masses) and masses == sorted(masses)
        # rank-2 m_D: exactly one massless light state, two massive light, two heavy
        assert masses[0] == 0
        assert 0 < masses[1] < masses[2] < 1e-3 < 1e2 < masses[3]
        # dual check: light masses = |eigenvalues| of −m_D M_R⁻¹ m_Dᵀ (to O(m_D²/M_R²))
        lam = sp.Symbol("lam")
        charpoly = seesaw_light_mass(mD, MR).charpoly(lam).as_expr()
        formula = sorted(abs(complex(r)) for r in sp.Poly(charpoly, lam).nroots(n=40))
        for a, b in zip(masses[1:3], formula[1:]):
            assert abs(float(a) - b) / b < 1e-6
        for k, M_k in ((3, 1000), (4, 3000)):
            assert abs(float(masses[k]) / M_k - 1) < 1e-6

    def test_numeric_matches_symbolic_on_block_matrix(self):
        """A diagonal m_D splits the 6×6 into 2×2 blocks: the exact route
        still finishes there, and both routes agree."""
        M = seesaw_mass_matrix(sp.diag(sp.Rational(2, 100), sp.Rational(3, 100),
                                       sp.Rational(5, 100)),
                               sp.diag(sp.Integer(1000), 2000, 5000))
        Us, Ds = diagonalize_takagi(M, method="symbolic")
        Un, Dn = diagonalize_takagi(M)                # auto → numeric
        exact = sorted((sp.N(Ds[k, k], 45) for k in range(6)), key=abs)
        for a, b in zip([Dn[k, k] for k in range(6)], exact):
            assert abs(a - b) < sp.Float("1e-35") * max(abs(b), 1)
        assert self._maxabs(Un * Dn * Un.T - M) < 1e-40 * 5000

    def test_negative_eigenvalue_gets_phase_i(self):
        M = sp.Matrix([[0, 1], [1, 0]])               # eigenvalues ±1
        U, D = diagonalize_takagi(M, method="numeric")
        assert all(abs(D[k, k] - 1) < 1e-45 for k in range(2))
        assert any(x.has(sp.I) for x in U)
        assert self._maxabs(U * D * U.T - M) < 1e-45

    def test_auto_keeps_small_and_symbolic_matrices_exact(self):
        mD, MR = sp.symbols("mD MR", positive=True)
        U, D = diagonalize_takagi(sp.Matrix([[0, mD], [mD, MR]]))
        assert not any(x.atoms(sp.Float) for x in U)  # symbolic route
        U, D = diagonalize_takagi(sp.Matrix([[0, 1], [1, 1000]]))
        assert not any(x.atoms(sp.Float) for x in U)  # 2×2 numeric stays exact

    def test_rotation_third_element(self):
        _, _, M = self._generic_2n()
        f = sp.symbols("n0:5")
        g = sp.symbols("x0:5")
        U, D, rot = diagonalize_takagi(M, f, g)
        assert rot.kind == "unitary" and rot.matrix == U.conjugate().T

    def test_numeric_input_validation(self):
        m = sp.Symbol("m")
        with pytest.raises(ValueError, match="numeric"):
            diagonalize_takagi(sp.Matrix([[0, m], [m, 1]]), method="numeric")
        with pytest.raises(ValueError, match="real"):
            diagonalize_takagi(sp.Matrix([[sp.I, 1], [1, 1]]), method="numeric")
        with pytest.raises(ValueError, match="method"):
            diagonalize_takagi(sp.Matrix([[0, 1], [1, 0]]), method="svd")
        with pytest.raises(ValueError, match="symmetric"):
            diagonalize_takagi(sp.Matrix([[0, 1, 0], [2, 0, 0], [0, 0, 1]]))


def test_seesaw_mass_matrix_block_structure():
    mD, MR = sp.symbols("mD MR", positive=True)
    M = seesaw_mass_matrix([[mD]], [[MR]])
    assert M == sp.Matrix([[0, mD], [mD, MR]])
    assert sp.simplify(M - M.T) == sp.zeros(2)


def test_seesaw_light_mass_formula():
    mD, MR = sp.symbols("mD MR", positive=True)
    assert sp.simplify(seesaw_light_mass([[mD]], [[MR]])[0, 0] + mD**2 / MR) == 0


def test_seesaw_light_mass_matches_takagi():
    """The −m_D M_R⁻¹ m_Dᵀ formula equals the light block of the exact Takagi
    diagonalisation (dual verification)."""
    M = seesaw_mass_matrix([[1]], [[1000]])       # numeric 1-gen
    U, D = diagonalize_takagi(M)
    masses = sorted(abs(_num(D[k, k])) for k in range(2))
    formula = abs(_num(seesaw_light_mass([[1]], [[1000]])[0, 0]))
    assert abs(masses[0] - formula) < 1e-9        # light mass
    assert abs(masses[1] - 1000.0) < 1e-2         # heavy ≈ M_R


def test_seesaw_three_generation_spectrum():
    """A 3+3 seesaw has 3 light (≪ M_R) + 3 heavy (~ M_R) states, and the
    light masses match the seesaw formula."""
    mD = sp.Matrix([[sp.Rational(2, 100), 0, 0],
                    [0, sp.Rational(3, 100), 0],
                    [0, 0, sp.Rational(5, 100)]])
    MR = sp.diag(sp.Integer(10)**3, 2 * 10**3, 5 * 10**3)
    M = seesaw_mass_matrix(mD, MR)
    U, D = diagonalize_takagi(M)
    masses = sorted(abs(_num(D[k, k])) for k in range(6))
    assert all(m < 1e-3 for m in masses[:3])      # 3 light
    assert all(m > 1e2 for m in masses[3:])       # 3 heavy
    light_formula = sorted(abs(_num(e))
                           for e in seesaw_light_mass(mD, MR).eigenvals())
    for a, b in zip(masses[:3], light_formula):
        assert abs(a - b) / b < 1e-6


# --------------------------------------------------------------------------
# the full SM + seesaw model (1 generation) + coupling extraction
# --------------------------------------------------------------------------

@pytest.fixture
def seesaw_model():
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)
    yv = ExternalParameter("yv", 0.01, positive=True)
    MR = ExternalParameter("MR", 1.0e3, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    nuR = WeylFermion("nuR", reps={}, chirality="R", nflavors=1,
                      component_names=["nuR"])
    i, j = sp.symbols("i j", integer=True)
    Gp, H0 = H.components
    nuL, eL = Ll.components
    nuLbar, eLbar = Ll.bar_components
    nR, nRbar = nuR.components[0], nuR.bar_components[0]
    CPL = diracC * diracPL
    LYukD = -(yv.s * sp.conjugate(H0) * Bilinear(nuLbar[i], diracPR, nR[j])
              + yv.s * (-sp.conjugate(Gp)) * Bilinear(eLbar[i], diracPR, nR[j]))
    LYukD += -(sp.conjugate(yv.s) * H0 * Bilinear(nRbar[j], diracPL, nuL[i])
               + sp.conjugate(yv.s) * (-Gp) * Bilinear(nRbar[j], diracPL, eL[i]))
    op = -sp.Rational(1, 2) * MR.s * MajoranaBilinear(nR[i], CPL, nR[j])
    LMaj = op + sp.conjugate(op)
    current = fermion_gauge_current(Ll, i) + fermion_gauge_current(eR, i)
    L = Lagrangian()
    L.add((dag(Dmu(H)) * Dmu(H))[0], sector="kinetic")
    L.add(-(-mu2.s * (dag(H) * H.mat)[0] + lam.s * (dag(H) * H.mat)[0]**2),
          sector="potential")
    L.add(LYukD + current, sector="yukawa")
    L.add(LMaj, sector="other")
    model = Model("SM-seesaw", gauge_groups=[SU2L, U1Y],
                  fields=[H, Ll, eR, nuR, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2, yv, MR], lagrangian=L)
    model.solve_tadpoles([mu2])
    W, B = SU2L.bosons(), U1Y.bosons()
    W1, W2, W3 = W.components
    B0 = B.components[0]
    Z, A = sp.symbols("Z A", real=True)
    model.rotate(Rotation([W3, B0], [Z, A], rotation_2x2(-sp.atan(g1.s / gw.s))))
    Wp, Wm = sp.symbols("Wp Wm")
    model.rotate(Rotation([W1, W2], [Wp, Wm],
                          sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2),
                          kind="unitary"))
    return dict(model=model, LYukD=LYukD, LMaj=LMaj, CPL=CPL,
                nuL=nuL, nuLbar=nuLbar, nR=nR, nRbar=nRbar, eLbar=eLbar,
                i=i, j=j, gw=gw, g1=g1, v=v, yv=yv, MR=MR, Wp=Wp, Wm=Wm, Z=Z)


def test_seesaw_model_invariant(seesaw_model):
    """SM + Dirac Yukawa + Majorana mass is gauge invariant."""
    assert seesaw_model["model"].check_invariance().ok


def _diagonalize(m, bench, method="auto"):
    Mnu = seesaw_mass_matrix(
        fermion_mass_matrix(m["LYukD"], m["nuLbar"], m["nR"], m["model"].vacuum,
                            1, (m["i"], m["j"]), gamma=diracPR),
        majorana_mass_matrix(m["LMaj"], m["nR"], m["model"].vacuum, 1,
                             (m["i"], m["j"]), gamma=m["CPL"]))
    Mn = sp.Matrix(2, 2, lambda a, b: sp.nsimplify(Mnu[a, b].subs(bench)))
    U, D = diagonalize_takagi(Mn, method=method)
    masses = [abs(_num(D[k, k])) for k in range(2)]
    light = 0 if masses[0] < masses[1] else 1
    return U, masses, light, 1 - light


def test_seesaw_spectrum_and_mixing(seesaw_model):
    m = seesaw_model
    bench = {m["yv"].s: 0.01, m["v"].s: 246.0, m["MR"].s: 1000.0,
             m["gw"].s: 0.6535, m["g1"].s: 0.3580}
    U, masses, light, heavy = _diagonalize(m, bench)
    # m_D = yv v/√2 = 1.739 GeV, m_light ≈ m_D²/M_R = 3.03 MeV, m_heavy ≈ M_R
    assert abs(masses[light] - 1.739**2 / 1000) / masses[light] < 1e-2
    assert abs(masses[heavy] - 1000.0) < 1.0
    mixing = abs(_num(U.conjugate()[0, heavy]))
    assert abs(mixing - 1.739 / 1000) / mixing < 1e-2      # V ≈ m_D/M_R


def _extract(m, U, bench):
    nuL, nR, nuLbar, nRbar = m["nuL"], m["nR"], m["nuLbar"], m["nRbar"]
    chiL, chiR = sp.IndexedBase("chiL"), sp.IndexedBase("chiR")
    chiLbar, chiRbar = sp.IndexedBase("chiLbar"), sp.IndexedBase("chiRbar")
    rot = MajoranaRotation(U, nuL, nR, nuLbar, nRbar, chiL, chiR, chiLbar,
                           chiRbar, n_L=1)
    Lchi = rot.apply(m["model"].physical_lagrangian(sector="yukawa"),
                     (m["i"], m["j"]), 1)
    tab = extract_fermion_vertices(Lchi, [m["Wp"], m["Wm"], m["Z"]])
    return tab, chiL, chiLbar


@pytest.mark.parametrize("method", ["symbolic", "numeric"])
def test_seesaw_W_production_coupling(seesaw_model, method):
    """W⁻ ē ν_light = g/√2 (full SM); W⁻ ē N_heavy = (g/√2)·V (∝ m_D/M_R).
    The numeric Takagi ``U`` feeds :class:`MajoranaRotation` unchanged."""
    m = seesaw_model
    bench = {m["yv"].s: 0.01, m["v"].s: 246.0, m["MR"].s: 1000.0,
             m["gw"].s: 0.6535, m["g1"].s: 0.3580}
    U, masses, light, heavy = _diagonalize(m, bench, method)
    tab, chiL, chiLbar = _extract(m, U, bench)
    mu = sp.Symbol("mu", integer=True)
    gL = DiracGamma(mu) * diracPL
    gsq = float((m["gw"].s / sp.sqrt(2)).subs(bench))

    def cW(k):
        c = tab.get((m["eLbar"][0], gL, chiL[k]), {}).get(1, {}).get((m["Wm"],), 0)
        return abs(_num(sp.simplify(c).subs(bench))) if c != 0 else 0.0

    assert abs(cW(light) / gsq - 1.0) < 1e-3                 # light: full SM
    mixing = abs(_num(U.conjugate()[0, heavy]))
    assert abs(cW(heavy) / gsq - mixing) / mixing < 1e-3     # heavy: (g/√2)·V


def test_seesaw_decoupling(seesaw_model):
    """The heavy-N production coupling → 0 as M_R → ∞ (seesaw decoupling)."""
    m = seesaw_model
    mu = sp.Symbol("mu", integer=True)
    gL = DiracGamma(mu) * diracPL
    prev = None
    for MRval in (1.0e3, 1.0e4, 1.0e5):
        bench = {m["yv"].s: 0.01, m["v"].s: 246.0, m["MR"].s: MRval,
                 m["gw"].s: 0.6535, m["g1"].s: 0.3580}
        U, masses, light, heavy = _diagonalize(m, bench)
        tab, chiL, chiLbar = _extract(m, U, bench)
        c = tab.get((m["eLbar"][0], gL, chiL[heavy]), {}).get(1, {}).get((m["Wm"],), 0)
        cW = abs(_num(sp.simplify(c).subs(bench))) if c != 0 else 0.0
        if prev is not None:
            assert cW < prev / 5          # falls ~ 1/M_R each decade
        prev = cW

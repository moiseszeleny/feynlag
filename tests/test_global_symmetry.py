"""Global U(1) with per-flavour charges (FG-7): Froggatt–Nielsen invariance.

A horizontal U(1)_FN gives each generation of a multiplet its own charge; the
Yukawa ``Q̄_i H̃ u_j`` then needs ``(φ/Λ)^{n_ij}`` with
``n_ij = q(Q_i) − q(u_j)`` (``φ*`` for ``n < 0``).  ``ZN.assign`` cannot express
this (one charge per multiplet), and a coupling-less gauge ``U1`` would be
gauged by ``Dmu``.
"""

import pytest
import sympy as sp

from feynlag import (
    Bilinear, Dmu, GlobalU1, Lagrangian, Model, PartialMu, SU2, Scalar, U1,
    WeylFermion, ZN, check_discrete_invariance, check_global_invariance, dag,
    diracPL, diracPR, fermion_gauge_current,
)

QQ = (3, 2, 0)          # q(Q_L) per generation
QU = (3, 1, 0)          # q(u_R) per generation


@pytest.fixture(scope="module")
def fn():
    gw, g1 = sp.symbols("gw_fn g1_fn", positive=True)
    SU2L, U1Y = SU2("SU2L_fn", coupling=gw), U1("U1Y_fn", coupling=g1)
    H = Scalar("Hfn", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gpfn", "H0fn"])
    QL = WeylFermion("QLfn", reps={SU2L: 2, U1Y: sp.Rational(1, 6)},
                     chirality="L", nflavors=3,
                     component_names=["uLfn", "dLfn"])
    uR = WeylFermion("uRfn", reps={U1Y: sp.Rational(2, 3)}, chirality="R",
                     nflavors=3, component_names=["uRfn"])
    phiF = Scalar("phifn", reps={}, component_names=["phifn"])
    Lam = sp.Symbol("Lam_fn", positive=True)
    c = sp.IndexedBase("cfn")

    FN = GlobalU1("U1_FN")
    FN.assign(QQ, QL).assign(QU, uR).assign(1, phiF)
    return dict(SU2L=SU2L, U1Y=U1Y, H=H, QL=QL, uR=uR, phiF=phiF, Lam=Lam,
                c=c, FN=FN)


def flavon_power(phi, n):
    return phi ** n if n >= 0 else sp.conjugate(phi) ** (-n)


def up_yukawa(m, powers):
    """``−c_ij F_ij Q̄_i H̃ u_j + h.c.`` written flavour by flavour."""
    Gp, H0 = m["H"].components
    uLbar, dLbar = m["QL"].bar_components
    uL, dL = m["QL"].components
    uR, uRbar = m["uR"].components[0], m["uR"].bar_components[0]
    phi = m["phiF"].components[0]
    L = 0
    for a in range(3):
        for b in range(3):
            F = flavon_power(phi / m["Lam"], powers[a][b])
            cab = m["c"][a, b]
            # Q̄ H̃ = ūL H0* − d̄L G+*
            L += -cab * F * (sp.conjugate(H0) * Bilinear(uLbar[a], diracPR, uR[b])
                             - sp.conjugate(Gp) * Bilinear(dLbar[a], diracPR, uR[b]))
            L += -sp.conjugate(cab) * sp.conjugate(F) * (
                H0 * Bilinear(uRbar[b], diracPL, uL[a])
                - Gp * Bilinear(uRbar[b], diracPL, dL[a]))
    return L


FN_POWERS = [[QQ[a] - QU[b] for b in range(3)] for a in range(3)]


class TestFroggattNielsenYukawa:
    def test_correct_powers_invariant(self, fn):
        ok, viol = check_global_invariance(up_yukawa(fn, FN_POWERS), fn["FN"])
        assert ok, viol

    def test_negative_powers_use_conjugate_flavon(self, fn):
        # n_ij < 0 occurs (q(Q_3)=0 < q(u_1)=3): the benchmark exercises φ*
        assert min(min(r) for r in FN_POWERS) < 0
        assert check_global_invariance(up_yukawa(fn, FN_POWERS), fn["FN"])[0]

    def test_one_wrong_power_is_flagged_exactly(self, fn):
        """Negative control: n_12 off by one leaves exactly the (1,2) entry
        and its h.c. charged (charge ∓1), nothing else."""
        bad = [row[:] for row in FN_POWERS]
        bad[0][1] += 1
        ok, viol = check_global_invariance(up_yukawa(fn, bad), fn["FN"])
        assert not ok
        (_, coeff), = viol
        # the residual i·Σ Q_m m is exactly ±i times the bad (1,2) pieces
        c12 = fn["c"][0, 1]
        terms = sp.Add.make_args(sp.expand(up_yukawa(fn, bad)))
        direct = sum(t for t in terms
                     if t.has(c12) and not t.has(sp.conjugate(c12)))
        conj = sum(t for t in terms if t.has(sp.conjugate(c12)))
        # the residual is i·Σ Q_m m: the bad entry has charge +1 (one φ too
        # many), its h.c. −1, every other monomial 0
        assert sp.expand(coeff - sp.I * (direct - conj)) == 0

    def test_zn_cannot_express_it(self, fn):
        """Why FG-7 needed a new group: a Z_N with the first-generation
        charges flags the off-diagonal entries of the same Yukawa."""
        Z = ZN("Z_FN_fn", 64)
        Z.assign(1, fn["phiF"]).assign(QQ[0], fn["QL"]).assign(QU[0], fn["uR"])
        ok, _ = check_discrete_invariance(up_yukawa(fn, FN_POWERS), Z)
        assert not ok


class TestFlavonSector:
    def test_kinetic_and_potential(self, fn):
        phi = fn["phiF"].components[0]
        H = fn["H"]
        mu2, lam, lamHP = sp.symbols("mu2_fn lam_fn lamHP_fn", real=True)
        a2 = phi * sp.conjugate(phi)
        HdH = (dag(H) * H.mat)[0]
        for term in (PartialMu(sp.conjugate(phi)) * PartialMu(phi),
                     -mu2 * a2 + lam * a2 ** 2 + lamHP * HdH * a2):
            assert check_global_invariance(term, fn["FN"])[0]

    def test_charged_operators_flagged(self, fn):
        phi = fn["phiF"].components[0]
        m = sp.Symbol("m_fn", positive=True)
        ok, viol = check_global_invariance(m ** 2 * phi ** 2 + m ** 2 * sp.conjugate(phi) ** 2,
                                           fn["FN"])
        assert not ok
        # O(α) coefficient is i·Σ Q_m m: +2 for φ², −2 for φ*²
        assert sp.expand(viol[0][1] - 2 * sp.I * m ** 2 * (
            phi ** 2 - sp.conjugate(phi) ** 2)) == 0
        derivative = PartialMu(phi) * PartialMu(phi)
        assert not check_global_invariance(derivative, fn["FN"])[0]

    def test_uncharged_derivative_kept(self, fn):
        """A derivative of a field the U(1) does not act on is not zeroed
        (the FG-2 lesson): the Higgs kinetic term is invariant."""
        H = fn["H"]
        term = (dag(Dmu(H)) * Dmu(H))[0]
        assert check_global_invariance(term, fn["FN"])[0]


class TestFlavourIndices:
    def test_symbolic_index_needs_uniform_charges(self, fn):
        i, j = sp.symbols("i_fn j_fn", integer=True)
        uLbar = fn["QL"].bar_components[0]
        uR = fn["uR"].components[0]
        with pytest.raises(ValueError, match="flavour-dependent"):
            check_global_invariance(Bilinear(uLbar[i], diracPR, uR[j]),
                                    fn["FN"])

    def test_symbolic_index_with_uniform_charges(self):
        i = sp.Symbol("i_u", integer=True)
        psi = WeylFermion("psiU", reps={}, chirality="L", nflavors=3,
                          component_names=["psiU"])
        chi = WeylFermion("chiU", reps={}, chirality="R", nflavors=3,
                          component_names=["chiU"])
        G = GlobalU1("U1_uniform").assign(2, psi).assign((2, 2, 2), chi)
        m = sp.IndexedBase("mU")
        term = m[i] * Bilinear(psi.bar_components[0][i], diracPR,
                               chi.components[0][i])
        assert check_global_invariance(term, G)[0]
        G2 = GlobalU1("U1_shifted").assign(2, psi).assign(1, chi)
        assert not check_global_invariance(term, G2)[0]

    def test_symbolic_charges(self):
        a, b = sp.symbols("qa qb", real=True)
        chi = Scalar("chiS", reps={}, component_names=["chiS"])
        phi = Scalar("phiS", reps={}, component_names=["phiS"])
        G = GlobalU1("U1_sym").assign(a, chi).assign(b, phi)
        x, y = chi.components[0], phi.components[0]
        ok, viol = check_global_invariance(sp.conjugate(x) * y
                                           + x * sp.conjugate(y), G)
        assert not ok
        assert sp.expand(viol[0][1] - sp.I * (b - a) * (
            sp.conjugate(x) * y - x * sp.conjugate(y))) == 0
        assert sp.expand(viol[0][1].subs(b, a)) == 0


class TestAssignmentErrors:
    def test_wrong_number_of_charges(self, fn):
        with pytest.raises(ValueError, match="flavour"):
            GlobalU1("bad").assign((1, 2), fn["QL"])

    def test_reassign(self, fn):
        G = GlobalU1("twice").assign(1, fn["phiF"])
        with pytest.raises(ValueError, match="already"):
            G.assign(2, fn["phiF"])

    def test_real_scalar_cannot_be_charged(self):
        s = Scalar("sReal", reps={}, component_names=["sReal"], real=True)
        with pytest.raises(ValueError, match="real component"):
            GlobalU1("r").assign(1, s)

    def test_charge_must_be_real(self):
        s = Scalar("sCplx", reps={}, component_names=["sCplx"])
        with pytest.raises(ValueError, match="must be real"):
            GlobalU1("c").assign(sp.Symbol("q_unknown"), s)

    def test_scalar_rejects_tuple(self, fn):
        with pytest.raises(ValueError, match="fermions only"):
            GlobalU1("t").assign((1, 2, 3), fn["phiF"])


class TestModelWiring:
    def _model(self, fn, powers):
        L = Lagrangian()
        L.add(up_yukawa(fn, powers), sector="yukawa", name="yuk_up")
        phi = fn["phiF"].components[0]
        L.add(PartialMu(sp.conjugate(phi)) * PartialMu(phi), sector="kinetic",
              name="flavon_kin")
        return Model("fn_toy", gauge_groups=[fn["SU2L"], fn["U1Y"]],
                     global_groups=[fn["FN"]],
                     fields=[fn["H"], fn["QL"], fn["uR"], fn["phiF"]],
                     lagrangian=L)

    def test_model_reports_global_failure(self, fn):
        bad = [row[:] for row in FN_POWERS]
        bad[2][2] = 1
        rep = self._model(fn, bad).check_invariance(dimension=False)
        labels = {label for _, label, _ in rep.failures}
        assert labels == {"global:U1_FN"}

    def test_model_passes_with_correct_powers(self, fn):
        rep = self._model(fn, FN_POWERS).check_invariance(dimension=False)
        assert not rep.failures, rep.failures

    def test_rejects_non_global(self, fn):
        with pytest.raises(TypeError, match="GlobalU1"):
            Model("x", global_groups=[fn["U1Y"]])


class TestReviewFollowUps:
    """PR #32 review: flavour-diagonal currents, index range, validate(),
    and the anomaly check."""

    def _kinetic_model(self, fn, charges_QL=QQ, extra_terms=()):
        i = sp.Symbol("i_kin", integer=True)
        G = GlobalU1("U1_kin").assign(charges_QL, fn["QL"]).assign(QU, fn["uR"])
        L = Lagrangian()
        L.add(fermion_gauge_current(fn["QL"], i)
              + fermion_gauge_current(fn["uR"], i), sector="kinetic",
              name="fermion_currents")
        for k, t in enumerate(extra_terms):
            L.add(t, sector="other", name=f"extra{k}")
        return Model("kin_toy", gauge_groups=[fn["SU2L"], fn["U1Y"]],
                     global_groups=[G],
                     fields=[fn["H"], fn["QL"], fn["uR"], fn["SU2L"].bosons(),
                             fn["U1Y"].bosons()],
                     lagrangian=L), G

    def test_flavour_diagonal_current_with_symbolic_index(self, fn):
        """ψ̄_i γ^μ T ψ_i (incl. the off-diagonal ū_L…d_L doublet pieces) is
        neutral for every i, so it must not raise despite charges (3,2,0)."""
        model, _ = self._kinetic_model(fn)
        rep = model.check_invariance(dimension=False)
        assert not rep.failures, rep.failures

    def test_different_symbolic_indices_still_raise(self, fn):
        i, j = sp.symbols("i_d j_d", integer=True)
        uLbar, uL = fn["QL"].bar_components[0], fn["QL"].components[0]
        with pytest.raises(ValueError, match="flavour-dependent"):
            check_global_invariance(Bilinear(uLbar[i], diracPL, uL[j]),
                                    fn["FN"])

    def test_same_index_different_charges_raise(self, fn):
        """Q̄_i u_i shares an index, but −q(Q_i)+q(u_i) = (0, 1, 0) is not
        flavour independent."""
        i = sp.Symbol("i_s", integer=True)
        uLbar = fn["QL"].bar_components[0]
        uR = fn["uR"].components[0]
        with pytest.raises(ValueError, match="depends on the flavour"):
            check_global_invariance(Bilinear(uLbar[i], diracPR, uR[i]),
                                    fn["FN"])

    @pytest.mark.parametrize("k", [3, -1])
    def test_out_of_range_flavour_index(self, fn, k):
        uLbar = fn["QL"].bar_components[0]
        uR = fn["uR"].components[0]
        with pytest.raises(ValueError, match="out of range"):
            check_global_invariance(Bilinear(uLbar[0], diracPR, uR[k]),
                                    fn["FN"])

    def test_shared_index_with_mismatched_flavour_counts(self):
        """Legs sharing a symbolic index but with 3 and 2 flavours: a clear
        ValueError, not an IndexError from the shorter charge tuple."""
        i = sp.Symbol("i_mm", integer=True)
        A = WeylFermion("Amm", reps={}, chirality="L", nflavors=3,
                        component_names=["Amm"])
        B = WeylFermion("Bmm", reps={}, chirality="R", nflavors=2,
                        component_names=["Bmm"])
        G = GlobalU1("U1_mm").assign((1, 1, 1), A).assign((1, 1), B)
        term = Bilinear(A.bar_components[0][i], diracPR, B.components[0][i])
        with pytest.raises(ValueError, match="different numbers of flavours"):
            check_global_invariance(term, G)

    def test_validate_reports_global_failure(self, fn):
        phi = fn["phiF"].components[0]
        G = GlobalU1("U1_val").assign(1, fn["phiF"])
        L = Lagrangian().add(phi ** 2 + sp.conjugate(phi) ** 2,
                             sector="potential", name="charged")
        model = Model("val_toy", global_groups=[G], fields=[fn["phiF"]],
                      lagrangian=L)
        report = model.validate(dimension=False)
        assert not report.ok
        labels = {label for _, label, _ in
                  report.checks["invariance"].failures}
        assert labels == {"global:U1_val"}

    def test_anomaly_check_ignores_global_groups(self, fn):
        """A global U(1) carries no gauge-anomaly constraint: adding one with
        non-vanishing Σq³ leaves the anomaly report unchanged."""
        with_global, G = self._kinetic_model(fn)
        without = Model("kin_toy_plain",
                        gauge_groups=with_global.gauge_groups,
                        fields=with_global.fields,
                        lagrangian=with_global.lagrangian)
        a, b = with_global.check_anomalies(), without.check_anomalies()
        assert a.coefficients == b.coefficients
        assert not any(G.name in name for name in a.coefficients)

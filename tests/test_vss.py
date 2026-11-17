"""Tier 4, stage 4c of the 2→2 scattering roadmap: the ``VSS`` derivative
coupling — a vector meeting two scalars, with the ``p(φ)`` momentum tags of
the extractor resolved to diagram momenta.

The model is scalar QED with a Yukawa: a U(1) with an electron, a neutral
left-handed fermion ``χ`` and a charged scalar ``φ`` (``|D_μφ|²``, a mass
term, ``−y φ† χ̄ P_R e + h.c.``), so the ``γφφ`` vertex really comes from the
extractor and processes sensitive to its sign exist.

What is pinned, and against what:

- the tag rule (:data:`feynlag.pheno.boson_vertices.VSS_KAPPA`) against a
  direct plane-wave evaluation of the kinetic term, bypassing the extractor;
- ``σ(e⁺e⁻→φ⁺φ⁻) = πα²β³/(3s)``, the point-like scalar Born cross section;
- the **sign**, exactly, through the Ward identity of ``e⁻γ→χφ⁻`` (electron
  exchange + ``φ`` exchange through ``γφφ``) — and a flipped sign breaks it;
- both processes, and a massive-vector/unequal-mass case, against an
  explicit spinor evaluator with the Feynman rules written out by hand.
"""

import itertools
import math

import numpy as np
import pytest
import sympy as sp

from feynlag import (
    Bilinear, ExternalParameter, Lagrangian, Model, Scalar, U1, WeylFermion,
    conjugate_pair, diracPR,
)
from feynlag.dirac import _dirac_rep
from feynlag.operators import PartialMu, momentum
from feynlag.pheno import (
    DecayVertex, DiracParticle, ExternalBoson, ExternalFermion,
    ScatteringCalculator, TwoToTwoKinematics, VSS_KAPPA,
    enumerate_line_diagrams, line_amplitude, vss_tensor,
)
from feynlag.pheno import boson_vertices

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_G5 = np.array(_REP["g5"].evalf(), dtype=complex)
_PL, _PR = (np.eye(4) - _G5) / 2, (np.eye(4) + _G5) / 2
_MET = np.diag([1.0, -1.0, -1.0, -1.0])

_E, _Y, _MPHI = 0.3, 0.5, 30.0


@pytest.fixture(scope="module")
def sqed():
    e_em = ExternalParameter("q_em", _E, positive=True)
    y = ExternalParameter("y_chi", _Y, positive=True)
    mphi = ExternalParameter("m_phi", _MPHI, positive=True)
    U1em = U1("U1em_vss", coupling=e_em)
    phi = Scalar("vphi", reps={U1em: -1}, component_names=["phim"])
    eL = WeylFermion("vsL", reps={U1em: -1}, chirality="L", nflavors=1,
                     component_names=["veL"])
    eR = WeylFermion("vsR", reps={U1em: -1}, chirality="R", nflavors=1,
                     component_names=["veR"])
    chiL = WeylFermion("vcL", reps={U1em: 0}, chirality="L", nflavors=1,
                       component_names=["vchiL"])
    A = U1em.bosons("A")
    i = sp.Symbol("i", integer=True)
    phim = phi.components[0]
    L = Lagrangian()
    L.add_kinetic(phi, eL, eR, chiL, index=i)
    L.add(-mphi.s**2 * sp.conjugate(phim) * phim, sector="potential")
    op = (-y.s * sp.conjugate(phim)
          * Bilinear(chiL.bar_components[0][i], diracPR, eR.components[0][i]))
    L.add(op + sp.conjugate(op), sector="yukawa")
    model = Model("ScalarQED", gauge_groups=[U1em],
                  fields=[phi, eL, eR, chiL, A],
                  parameters=[e_em, y, mphi], lagrangian=L)
    phip, cmap = conjugate_pair(phim, "phip")
    photon = A.components[0]
    chi = DiracParticle("vchi", chiL.components[0], None, 0)

    def setup(m_e=0, yukawa=_Y):
        e = DiracParticle("ve", eL.components[0], eR.components[0], m_e)
        calc = ScatteringCalculator(
            model, [e, chi], masses={photon: 0, phim: mphi.s},
            boson_fields=[photon, phim, phip], fermion_sectors=("yukawa",),
            conjugate_map=cmap, conjugates={phim: phip, phip: phim})
        return calc, e, {e_em.s: _E, y.s: yukawa, mphi.s: _MPHI}

    return dict(model=model, A=photon, phim=phim, phip=phip, cmap=cmap,
                chi=chi, setup=setup, e_em=e_em, y=y, U1em=U1em)


# --------------------------------------------------------------------------
# the model itself: what the extractor returns
# --------------------------------------------------------------------------

def test_model_without_vevs_is_invariant_and_extracts(sqed):
    """An unbroken model has no vacuum shift; its physical Lagrangian is the
    one written.  Both halves of the charged-scalar Yukawa must be found —
    the ``φ†`` half needs the conjugate map applied on the fermion track."""
    sqed["model"].check_invariance().raise_on_failure()
    calc, e, _ = sqed["setup"]()
    found = {(tuple(map(str, v.particles)), v.vertex_type)
             for v in calc.vertices()}
    assert found == {
        (("A", "phim", "phip"), "VSS"),
        (("vebar", "ve", "A"), "FFV"),
        (("vebar", "vchi", "phim"), "FFS"),
        (("vchibar", "ve", "phip"), "FFS"),
    }


def test_stray_field_filter_keeps_the_momentum_tags(sqed):
    """``_at_zero_fields`` must not read the scalar symbols *inside*
    ``p(φ)`` as field dependence (it would turn the tags into ``p(0)``)."""
    calc, _, _ = sqed["setup"]()
    (vss,) = [v for v in calc.vertices() if v.vertex_type == "VSS"]
    e, phim, phip = sqed["e_em"].s, sqed["phim"], sqed["phip"]
    assert sp.expand(vss.coupling
                     - sp.I * e * (momentum(phim) - momentum(phip))) == 0


# --------------------------------------------------------------------------
# the tag rule, from the kinetic term directly
# --------------------------------------------------------------------------

def test_tag_rule_from_plane_waves(sqed):
    """Bypass ``to_momentum_space``: put physical incoming plane waves
    ``e^{−ik·x}`` on each field leg of ``(D_μφ)†(D^μφ)``, i.e. replace every
    ``∂_μ f`` by ``−i k_f f``, and read the ``A φ φ†`` vertex ``i × coeff``.
    It must equal the extractor's tagged coupling with
    ``p(φ) → VSS_KAPPA · k_φ``."""
    model, A, phim, phip = sqed["model"], sqed["A"], sqed["phim"], sqed["phip"]
    k_phi, k_bar = sp.symbols("k_phi k_bar", real=True)
    waves = {phim: k_phi, sp.conjugate(phim): k_bar}
    kinetic = model.lagrangian.sector("kinetic")
    kinetic = kinetic.replace(
        lambda node: isinstance(node, PartialMu),
        lambda node: -sp.I * waves[node.args[0]] * node.args[0])
    kinetic = sp.expand(kinetic.xreplace(sqed["cmap"]))
    direct = sp.I * kinetic.coeff(A, 1).coeff(phim, 1).coeff(phip, 1)
    assert direct != 0

    calc, _, _ = sqed["setup"]()
    (vss,) = [v for v in calc.vertices() if v.vertex_type == "VSS"]
    head_phi, head_bar = sp.symbols("head_phi head_bar")
    pieces = vss_tensor(vss.coupling, "mu", {phim: ((1, head_phi),),
                                            phip: ((1, head_bar),)})
    value = {head_phi: k_phi, head_bar: k_bar}
    resolved = sum(coeff * sum(c * value[head] for c, head in tensors[0][1])
                   for coeff, tensors in pieces)
    assert sp.expand(resolved - direct) == 0
    assert VSS_KAPPA == -1
    # the textbook form: i g Q (p_φ − p_φ†)^μ with Q = −1
    assert sp.expand(direct + sp.I * sqed["e_em"].s * (k_phi - k_bar)) == 0


def test_vss_tensor_guards():
    a, b, S1, S2, S3 = sp.symbols("a b S1 S2 S3")
    combo = ((1, sp.Symbol("k")),)
    with pytest.raises(NotImplementedError, match="same field symbol"):
        vss_tensor(a * momentum(S1), "mu", {S1: combo})
    with pytest.raises(NotImplementedError, match="not linear"):
        vss_tensor(a * momentum(S1) + b * momentum(S3), "mu",
                   {S1: combo, S2: combo})
    with pytest.raises(NotImplementedError, match="not linear"):
        vss_tensor(a * momentum(S1) * momentum(S2), "mu",
                   {S1: combo, S2: combo})
    with pytest.raises(NotImplementedError, match="not linear"):
        vss_tensor(a * momentum(S1) + b, "mu", {S1: combo, S2: combo})


# --------------------------------------------------------------------------
# scalar pair production: the closed form
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rs", [70.0, 200.0, 1000.0])
def test_scalar_pair_production_closed_form(sqed, rs):
    """``σ(e⁺e⁻→φ⁺φ⁻) = πα²β³/(3s)`` for a point-like charged scalar and a
    massless electron (Bystritskiy, Fedotovich, Ignatov, Kuraev, Phys. Rev. D 72, 114019,
    arXiv:hep-ph/0505236; cited in docs/manual/scattering_roadmap.md §17.8)."""
    calc, e, values = sqed["setup"](yukawa=0)
    proc = calc.process((e.particle, e.antiparticle),
                        (sqed["phip"], sqed["phim"]), numeric=True, extra=values)
    assert [(d.kind, str(d.mediator)) for d in proc.diagrams
            if d.kind == "boson"] == [("boson", "A")]
    s = rs**2
    alpha, beta = _E**2 / (4 * math.pi), math.sqrt(1 - 4 * _MPHI**2 / s)
    closed = math.pi * alpha**2 * beta**3 / (3 * s)
    assert proc.numeric_cross_section(s) == pytest.approx(closed, rel=1e-9)


# --------------------------------------------------------------------------
# the sign: Ward identity
# --------------------------------------------------------------------------

def _photoproduction(sqed, m_e=0):
    calc, e, values = sqed["setup"](m_e=m_e)
    return calc.process((e.particle, sqed["A"]),
                        (sqed["chi"].particle, sqed["phim"]),
                        numeric=True, extra=values)


def test_ward_identity_fixes_the_vss_sign(sqed, monkeypatch):
    """``e⁻γ→χφ⁻``: electron exchange plus ``φ`` exchange through ``γφφ``.
    ``k_γ·M = 0`` only if the two have the right relative sign, and the
    engine checks it before using ``−g`` for the photon.  With the tag sign
    flipped the check must fail."""
    proc = _photoproduction(sqed)
    assert sorted((d.kind, d.channel) for d in proc.diagrams) == [
        ("boson", "t"), ("fermion", "s")]
    proc.squared()                                  # Ward check passes

    monkeypatch.setattr(boson_vertices, "VSS_KAPPA", -boson_vertices.VSS_KAPPA)
    flipped = _photoproduction(sqed)
    with pytest.raises(NotImplementedError, match="Ward identity"):
        flipped.squared()


# --------------------------------------------------------------------------
# explicit spinor evaluator
# --------------------------------------------------------------------------

def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


def _dot(a, b):
    return a @ _MET @ b


def _cm(s, cos, masses):
    m1, m2, m3, m4 = masses
    rs = math.sqrt(s)
    E1, E2 = (s + m1**2 - m2**2) / (2 * rs), (s + m2**2 - m1**2) / (2 * rs)
    E3, E4 = (s + m3**2 - m4**2) / (2 * rs), (s + m4**2 - m3**2) / (2 * rs)
    pi, pf = math.sqrt(E1**2 - m1**2), math.sqrt(E3**2 - m3**2)
    sin = math.sqrt(1 - cos**2)
    return [np.array([E1, 0, 0, pi]), np.array([E2, 0, 0, -pi]),
            np.array([E3, pf * sin, 0, pf * cos]),
            np.array([E4, -pf * sin, 0, -pf * cos])]


def _spinors(p, m, anti):
    M = (_slash(p) + (-m if anti else m) * np.eye(4)) @ _G[0]
    w, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    return [math.sqrt(w[j]) * vecs[:, j] for j in range(4) if w[j] > 1e-9 * p[0]]


def _engine_at(proc, s, cos):
    kin = proc.kinematics
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(proc.squared().subs({kin.s: s, kin.t: t}))


_POINTS = ((200.0**2, 0.3), (90.0**2, -0.65), (500.0**2, 0.92))


def _pair_oracle(s, cos, m_e):
    """``Σ|M|²`` for ``e⁻(k1) e⁺(k2) → φ⁺(k3) φ⁻(k4)``.

    Rules (``D = ∂ − i e Q A``, ``Q = −1``; the field ``φ`` annihilates
    ``φ⁻`` and creates ``φ⁺``): ``ēeγ`` is ``−i e γ^μ``; ``γφφ†`` is
    ``−i e (p_φ − p_φ†)^μ`` with momenta into the vertex on the ``φ`` and
    ``φ†`` legs — here ``−k3`` and ``−k4``; the Yukawa vertices are
    ``−i y P_R`` (``χ̄ e φ†``) and ``−i y P_L`` (``ē χ φ``).
    """
    ks = _cm(s, cos, (m_e, m_e, _MPHI, _MPHI))
    total = 0.0
    for u, v in itertools.product(_spinors(ks[0], m_e, False),
                                  _spinors(ks[1], m_e, True)):
        vbar = v.conj() @ _G[0]
        photon = ((vbar @ (-1j * _E * _slash(-ks[2] + ks[3])) @ u)
                  * (-1j / s) * (-1j * _E))
        q = ks[0] - ks[3]                      # φ⁻ leaves at the e⁻ vertex
        chi = (vbar @ (-1j * _Y * _PL) @ (1j * _slash(q))
               @ (-1j * _Y * _PR) @ u) / _dot(q, q)
        total += abs(photon + chi)**2
    return total


@pytest.mark.parametrize("m_e", [0, 4.0])
def test_pair_production_with_yukawa_vs_explicit_spinors(sqed, m_e):
    """γ s channel interfering with the ``χ`` exchange."""
    calc, e, values = sqed["setup"](m_e=m_e)
    proc = calc.process((e.particle, e.antiparticle),
                        (sqed["phip"], sqed["phim"]), numeric=True, extra=values)
    assert sorted((d.kind, str(d.mediator)) for d in proc.diagrams) == [
        ("boson", "A"), ("fermion", "vchi")]
    for s, cos in _POINTS:
        engine = _engine_at(proc, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(_pair_oracle(s, cos, m_e), rel=1e-9)


def _photoproduction_oracle(s, cos, m_e):
    """``Σ|M|²`` for ``e⁻(k1) γ(k2) → χ(k3) φ⁻(k4)``: electron in the
    s channel, and ``φ`` in the t channel through ``γφφ†`` (``φ`` leg
    momentum in ``q = k1 − k3``, ``φ†`` leg ``−k4``)."""
    ks = _cm(s, cos, (m_e, 0.0, 0.0, _MPHI))
    pols = [np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])]
    total = 0.0
    for u, w, eps in itertools.product(_spinors(ks[0], m_e, False),
                                       _spinors(ks[2], 0.0, False), pols):
        wbar = w.conj() @ _G[0]
        p = ks[0] + ks[1]
        electron = (wbar @ (-1j * _Y * _PR)
                    @ (1j * (_slash(p) + m_e * np.eye(4)))
                    @ (-1j * _E * _slash(eps)) @ u) / (s - m_e**2)
        q = ks[0] - ks[2]
        scalar = ((wbar @ (-1j * _Y * _PR) @ u)
                  * (1j / (_dot(q, q) - _MPHI**2))
                  * (-1j * _E * _dot(q + ks[3], eps)))
        total += abs(electron + scalar)**2
    return total


@pytest.mark.parametrize("m_e", [0, 4.0])
def test_photoproduction_vs_explicit_spinors(sqed, m_e):
    """Scalar mediator, one external vector and one external scalar on the
    ``VSS`` vertex — the other shape the same code handles."""
    proc = _photoproduction(sqed, m_e=m_e)
    for s, cos in _POINTS:
        engine = _engine_at(proc, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(
            _photoproduction_oracle(s, cos, m_e), rel=1e-9)


def test_massive_vector_and_unequal_scalar_masses():
    """Hand-built vertices: a massive vector with chiral couplings and a
    width, meeting two *different* real scalars with independent tag
    coefficients.  ``q·(a p₃ + b p₄) ≠ 0`` here, so the propagator's
    ``q_μq_ν/M²`` term matters."""
    f, fbar, V, S1, S2 = sp.symbols("f fbar V S1 S2")
    gl, gr, a, b = 0.7, -0.4, 1.3, 0.6 + 0.2j
    m_f, m_v, width, m1, m2 = 6.0, 80.0, 3.0, 20.0, 45.0
    fermion = type("P", (), {"particle": f, "antiparticle": fbar, "mass": m_f})()
    externals = [ExternalFermion(fermion, False, True),
                 ExternalFermion(fermion, True, True),
                 ExternalBoson(S1, 0, m1, False), ExternalBoson(S2, 0, m2, False)]
    kin = TwoToTwoKinematics(m_f, m_f, m1, m2)
    vertices = [
        DecayVertex((fbar, f, V), "FFV", g_left=sp.I * gl, g_right=sp.I * gr),
        DecayVertex((S1, S2, V), "VSS",
                    coupling=sp.I * (a * momentum(S1) + sp.sympify(b) * momentum(S2))),
    ]
    infos = enumerate_line_diagrams(
        externals, vertices, kin, species=[fermion],
        mass_of=lambda boson: m_v, width_of=lambda boson: width)
    assert [(d.kind, d.channel, d.mediator) for d in infos] == [("boson", "s", V)]
    m2_engine = line_amplitude(externals, infos, kin).squared(kin)

    for s, cos in _POINTS:
        ks = _cm(s, cos, (m_f, m_f, m1, m2))
        q = ks[0] + ks[1]
        # tags: p(S) -> −(momentum into the vertex) = +k for an outgoing S
        w = 1j * (a * ks[2] + b * ks[3])
        current_w = _slash(w) - _dot(q, w) / m_v**2 * _slash(q)
        total = 0.0
        for u, v in itertools.product(_spinors(ks[0], m_f, False),
                                      _spinors(ks[1], m_f, True)):
            amp = (v.conj() @ _G[0] @ current_w
                   @ (1j * gl * _PL + 1j * gr * _PR) @ u
                   * (-1j) / (s - m_v**2 + 1j * m_v * width))
            total += abs(amp)**2
        t = kin.t_of_cos(cos).subs(kin.s, s)
        engine = complex(m2_engine.subs({kin.s: s, kin.t: t}))
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(total, rel=1e-9)


def test_float_zero_mass_does_not_poison_the_epsilon_guard():
    """``0.0`` (not ``0``) for a massless leg used to make the Gram
    determinant ``nan`` — Bareiss elimination pivoting on ``k·k = 0.0`` —
    and the γ₅ guard then refused a perfectly good process."""
    from feynlag.pheno.epsilon import gram_determinant
    kin = TwoToTwoKinematics(0.0, 0.0, 30.0, 30.0)
    det = gram_determinant([kin.k1, kin.k2, kin.k3, kin.k4], kin.dot)
    assert sp.simplify(det) == 0
    # and non-zero float masses still give an exact zero (a division-free
    # determinant leaves a round-off residue the guard would read as ≠ 0)
    kin = TwoToTwoKinematics(0, 0, 80.4, 80.4)
    det = gram_determinant([kin.k1, kin.k2, kin.k3, kin.k4], kin.dot)
    assert sp.simplify(det) == 0

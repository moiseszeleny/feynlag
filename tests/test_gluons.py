"""Tier 5, stage 5b of the 2→2 scattering roadmap: external gluons on one
quark line — ``q q̄→gg``, ``gg→q q̄``, ``qg→qg`` and the mixed ``q q̄→gγ``.

Two things are new relative to Tier 4's line engine: two external gluons are
summed over their **physical** polarizations (``−g`` would count unphysical
ones that only ghosts cancel), and each diagram carries a colour tensor —
``T^aT^b`` for the quark-exchange diagrams, ``T^c f^{cab}`` for the gluon
exchange through the three-gluon vertex.

What is pinned, and against what:

- the stripped ``ggg`` coupling against ``cubic_couplings``;
- the physical polarization sum against explicit polarization vectors;
- the three gluon entries of the parton table of Ellis, Stirling and Webber,
  *QCD and Collider Physics*, Table 7.1 — exactly and symbolically;
- massive quarks against an explicit evaluator: explicit spinors, explicit
  transverse polarization vectors, explicit ``T^a`` and ``f^{abc}``, the
  amplitude summed over diagrams per colour and spin configuration — an
  evaluator that first checks its own Ward identity;
- independence of the reference vector; the Ward check failing with the
  triple-gauge sign flipped, and with ``−g`` on both gluons;
- the abelian limits ``q q̄→γγ`` and ``q q̄→gγ``.
"""

import itertools
import math

import numpy as np
import pytest
import sympy as sp

from feynlag import (
    ExternalParameter, Lagrangian, Model, SU3, U1, WeylFermion,
    cubic_couplings, fermion_gauge_current,
)
from feynlag.dirac import _dirac_rep
from feynlag.export.ufo.vvvv import adjoint_vvv
from feynlag.pheno import (
    DiracParticle, ScatteringCalculator, line_amplitude, line_colour_tensor,
    colour_matrix,
)
from feynlag.pheno import boson_vertices
from feynlag.pheno.line import VectorLeg

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_MET = np.diag([1.0, -1.0, -1.0, -1.0])

_GS, _EE = 1.2, 0.3
_QU = sp.Rational(2, 3)


@pytest.fixture(scope="module")
def qcd():
    gs = ExternalParameter("gs6", _GS, positive=True)
    ee = ExternalParameter("ee6", _EE, positive=True)
    SU3c, U1em = SU3("SU3c6", coupling=gs), U1("U1em6", coupling=ee)
    uL, uR = [WeylFermion(f"gu{chi}", reps={SU3c: 3, U1em: _QU},
                          chirality=chi, nflavors=1,
                          component_names=[f"gu{chi}_{k}" for k in (1, 2, 3)])
              for chi in ("L", "R")]
    G, A = SU3c.bosons("G"), U1em.bosons("A")
    i = sp.Symbol("i", integer=True)
    L = Lagrangian()
    for f in (uL, uR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("QCDgluons", gauge_groups=[SU3c, U1em],
                  fields=[uL, uR, G, A], parameters=[gs, ee], lagrangian=L)
    photon = A.components[0]

    def setup(m_u=0):
        u = DiracParticle("gu", uL.components[0], uR.components[0], m_u,
                          color=3)
        calc = ScatteringCalculator(model, [u], masses={photon: 0},
                                    boson_fields=[photon],
                                    colour_group=SU3c, gluon=G)
        return calc, u

    return dict(SU3c=SU3c, G=G, A=photon, gs=gs.s, ee=ee.s, setup=setup)


@pytest.fixture(scope="module")
def massless(qcd):
    """The three symbolic massless processes, ``Σ|M|²`` computed once."""
    calc, u = qcd["setup"]()
    g = calc.gluon
    procs = {
        "qqbar_gg": calc.process((u.particle, u.antiparticle), (g, g)),
        "gg_qqbar": calc.process((g, g), (u.particle, u.antiparticle)),
        "qg_qg": calc.process((u.particle, g), (u.particle, g)),
    }
    for proc in procs.values():
        proc.squared()
    return procs


# --------------------------------------------------------------------------
# ingredients
# --------------------------------------------------------------------------

def test_stripped_three_gluon_coupling(qcd):
    """``cubic_couplings`` is ``−g f^{abc}`` per component triple; the
    calculator uses ``adjoint_vvv = −g`` on one gluon symbol and puts
    ``f^{abc}`` in the colour tensor."""
    group = qcd["SU3c"]
    f = group.structure_constants()
    components = list(qcd["G"].components)
    table = cubic_couplings(group)
    assert len(table) == len(f)
    for (a, b, c), value in f.items():
        key = (components[a], components[b], components[c])
        assert sp.simplify(table[key] - adjoint_vvv(group) * value) == 0
    calc, _ = qcd["setup"]()
    (ggg,) = [v for v in calc.vertices() if v.vertex_type == "VVV"]
    assert ggg.particles == (calc.gluon,) * 3
    assert sp.simplify(ggg.coupling + qcd["gs"]) == 0


def test_line_colour_matrix_values():
    """``q q̄→gg``: ``Tr(T^aT^bT^bT^a) = N C_F² = 16/3``,
    ``Tr(T^aT^bT^aT^b) = −2/3``, and the three-gluon diagram
    ``Σ|T^c f^{cab}|² = N · ½ · 8 = 12``."""
    group = SU3("SU3linecol")
    T, f = group.generators(3), group.structure_constants()
    dims = (3, 3, 8, 8)
    ab = line_colour_tensor("fermion", dims, order=(0, 1), generators=T)
    ba = line_colour_tensor("fermion", dims, order=(1, 0), generators=T)
    s = line_colour_tensor("gluon", dims, generators=T, structure=f)
    C = colour_matrix([ab, ba, s])
    R = sp.Rational
    assert C[0][0] == C[1][1] == R(16, 3)
    assert C[0][1] == -R(2, 3)
    assert C[2][2] == 12
    # [T^a, T^b] = i f^{abc} T^c  ⟹  K_ab − K_ba = i K_s
    assert C[0][2] - C[1][2] == sp.I * C[2][2]
    with pytest.raises(ValueError, match="singlet mediator"):
        line_colour_tensor("singlet", dims)


def test_physical_polarization_sum_matches_explicit_vectors():
    """``−g + (k n + n k)/(k·n)`` equals ``Σ ε ε`` over the two real
    polarization vectors orthogonal to both ``k`` and ``n``."""
    k = np.array([5.0, 3.0, 0.0, 4.0])
    n = np.array([2.0, 0.0, 2.0, 0.0])
    vectors = {"k": k, "n": n}

    def dot(a, b):
        return float(vectors[a] @ _MET @ vectors[b])

    leg = VectorLeg("mu", "k", 0, reference="n")
    total = np.zeros((4, 4))
    for coeff, tensors in leg.polarization_sum(dot):
        if tensors[0][0] == "g":
            piece = _MET.copy()                      # g^{μν} (upper indices)
        else:
            (_, combo_a, _), (_, combo_b, _) = tensors
            piece = np.outer(vectors[combo_a[0][1]], vectors[combo_b[0][1]])
        total += float(coeff) * piece
    # two unit spacelike vectors orthogonal to k and n
    basis = []
    for seed in (np.array([0.0, 1.0, 0.3, -0.2]), np.array([0.0, -0.4, 1.0, 0.7])):
        e = seed.copy()
        for _ in range(3):
            e = e - (e @ _MET @ n) / (k @ _MET @ n) * k - (e @ _MET @ k) / (k @ _MET @ n) * n
            for other in basis:
                e = e + (e @ _MET @ other) * other   # other·other = −1
        basis.append(e / math.sqrt(-(e @ _MET @ e)))
    explicit = sum(np.outer(e, e) for e in basis)
    assert np.allclose(total, explicit, atol=1e-12)
    assert np.allclose(total @ _MET @ k, 0) and np.allclose(total @ _MET @ n, 0)
    with pytest.raises(ValueError, match="dot"):
        leg.polarization_sum()


# --------------------------------------------------------------------------
# the parton table
# --------------------------------------------------------------------------

def test_parton_table_gluon_entries(qcd, massless):
    """Ellis–Stirling–Webber Table 7.1 (massless; averaged over initial and
    summed over final spins and colours; ``g_s⁴`` removed).  A quark has
    ``2 × 3`` states, a gluon ``2 × 8``."""
    R = sp.Rational
    g4 = qcd["gs"]**4

    proc = massless["qqbar_gg"]
    kin = proc.kinematics
    s, t, u = kin.s, kin.t, kin.u
    assert sorted((d.kind, d.channel) for d in proc.diagrams) == [
        ("boson", "s"), ("fermion", "t"), ("fermion", "u")]
    assert [st.dof() for st in proc.initial_states] == [6, 6]
    assert proc.symmetry_factor == R(1, 2)
    assert sp.simplify(proc.squared() / 36 / g4 - (
        R(32, 27) * (t**2 + u**2) / (t * u)
        - R(8, 3) * (t**2 + u**2) / s**2)) == 0

    proc = massless["gg_qqbar"]
    kin = proc.kinematics
    s, t, u = kin.s, kin.t, kin.u
    assert [st.dof() for st in proc.initial_states] == [16, 16]
    assert sp.simplify(proc.squared() / 256 / g4 - (
        R(1, 6) * (t**2 + u**2) / (t * u)
        - R(3, 8) * (t**2 + u**2) / s**2)) == 0

    proc = massless["qg_qg"]
    kin = proc.kinematics
    s, t, u = kin.s, kin.t, kin.u
    assert sorted((d.kind, d.channel) for d in proc.diagrams) == [
        ("boson", "t"), ("fermion", "s"), ("fermion", "u")]
    assert sp.simplify(proc.squared() / 96 / g4 - (
        (s**2 + u**2) / t**2 - R(4, 9) * (s**2 + u**2) / (s * u))) == 0


def test_result_is_independent_of_the_reference_vector(massless):
    """The gluons' reference vectors are a gauge choice: taking the quark
    momenta instead of the other gluon's must give the same ``Σ|M|²``."""
    proc = massless["qqbar_gg"]
    kin = proc.kinematics
    other = line_amplitude(proc.externals, proc.diagrams, kin,
                           references={2: kin.k1, 3: kin.k2})
    assert sp.simplify(other.squared(kin, colour=proc.colour)
                       - proc.squared()) == 0


def test_minus_g_on_two_gluons_is_refused_and_wrong(massless):
    """With ``−g`` for both gluons the Ward check fails — and forcing it
    gives a different number: the unphysical polarizations do not cancel."""
    proc = massless["qqbar_gg"]
    kin = proc.kinematics
    covariant = line_amplitude(proc.externals, proc.diagrams, kin)
    with pytest.raises(NotImplementedError, match="Ward identity"):
        covariant.squared(kin, colour=proc.colour)
    forced = covariant.squared(kin, ward_check=False, colour=proc.colour)
    assert sp.simplify(forced - proc.squared()) != 0


def test_flipped_three_gluon_sign_breaks_the_ward_identity(qcd, monkeypatch):
    monkeypatch.setattr(boson_vertices, "VVV_KAPPA", -boson_vertices.VVV_KAPPA)
    calc, u = qcd["setup"]()
    proc = calc.process((u.particle, u.antiparticle), (calc.gluon, calc.gluon))
    with pytest.raises(NotImplementedError, match="Ward identity"):
        proc.squared()


# --------------------------------------------------------------------------
# abelian limits
# --------------------------------------------------------------------------

def test_photon_and_mixed_final_states(qcd):
    """``q q̄→γγ = N_c Q⁴ · 8e⁴(u/t + t/u)`` (the QED result of stage 4a
    times the colour of the line) and ``q q̄→gγ`` with ``Tr(T^aT^a) = 4``.
    Neither has a three-boson diagram, and ``−g`` is used."""
    calc, u = qcd["setup"]()
    e, g = qcd["ee"], qcd["gs"]
    qed = calc.process((u.particle, u.antiparticle), (qcd["A"], qcd["A"]))
    kin = qed.kinematics
    shape = 8 * (kin.u / kin.t + kin.t / kin.u)
    assert sp.simplify(qed.squared() - 3 * _QU**4 * e**4 * shape) == 0

    mixed = calc.process((u.particle, u.antiparticle), (calc.gluon, qcd["A"]))
    assert all(d.kind == "fermion" for d in mixed.diagrams)
    kin = mixed.kinematics
    shape = 8 * (kin.u / kin.t + kin.t / kin.u)
    assert sp.simplify(mixed.squared() - 4 * _QU**2 * e**2 * g**2 * shape) == 0


# --------------------------------------------------------------------------
# explicit spinors, polarization vectors, T^a and f^{abc}
# --------------------------------------------------------------------------

def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


def _dot(a, b):
    return a @ _MET @ b


def _spinors(p, m, anti):
    M = (_slash(p) + (-m if anti else m) * np.eye(4)) @ _G[0]
    w, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    return [math.sqrt(w[j]) * vecs[:, j] for j in range(4) if w[j] > 1e-9 * p[0]]


def _transverse(k):
    n = k[1:] / np.linalg.norm(k[1:])
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return [np.concatenate(([0.0], e1)), np.concatenate(([0.0], e2))]


def _three_gluon(q, pa, pb, ea, eb):
    """The three-gluon vertex with legs ``(c, ρ, q)``, ``(a, μ, pa)``,
    ``(b, ν, pb)`` (all momenta incoming), contracted with ``ea``, ``eb``:
    ``W^ρ`` such that the vertex is ``g f^{cab} W^ρ``."""
    return (ea * _dot(q - pa, eb) + _dot(ea, eb) * (pa - pb)
            + eb * _dot(pb - q, ea))


def _colour_arrays():
    group = SU3("SU3gluonoracle")
    T = np.array([np.array(Ta.evalf(), dtype=complex)
                  for Ta in group.generators(3)])
    f = np.zeros((8, 8, 8))
    for (a, b, c), value in group.structure_constants().items():
        f[a, b, c] = float(value)
    return T, f


def _amplitude(bar, spinor, ka, kb, ea, eb, q_first, q_second, q_gluon, m):
    """Colour-dressed amplitude ``[x, y, a, b]`` of a quark line ``bar … spinor``
    with gluons ``a`` and ``b`` (incoming momenta ``ka``, ``kb``).

    ``q_first``: the quark propagator momentum when ``b`` is emitted next to
    ``spinor`` (colour ``T^aT^b``); ``q_second``: when ``a`` is (colour
    ``T^bT^a``); ``q_gluon``: the momentum flowing from the line into the
    three-gluon vertex.  Vertices ``i g γ^μ T^a``, propagators
    ``i(q̸+m)/(q²−m²)`` and ``−i g_{μν}/q²``.
    """
    T, f = _colour_arrays()
    one = np.eye(4)

    def prop(q):
        return 1j * (_slash(q) + m * one) / (_dot(q, q) - m**2)

    vertex = 1j * _GS
    lor_ab = bar @ (vertex * _slash(ea)) @ prop(q_first) @ (vertex * _slash(eb)) @ spinor
    lor_ba = bar @ (vertex * _slash(eb)) @ prop(q_second) @ (vertex * _slash(ea)) @ spinor
    W = _three_gluon(q_gluon, ka, kb, ea, eb)
    lor_s = (bar @ (vertex * _slash(W)) @ spinor) * (-1j / _dot(q_gluon, q_gluon)) * _GS
    return (lor_ab * np.einsum("axz,bzy->xyab", T, T)
            + lor_ba * np.einsum("bxz,azy->xyab", T, T)
            + lor_s * np.einsum("cxy,cab->xyab", T, f))


def _annihilation_oracle(s, cos, m):
    """``Σ|M|²`` for ``q(k1) q̄(k2) → g(k3) g(k4)`` with a quark of mass
    ``m``; also returns the largest Ward-identity violation relative to the
    amplitude, which must be round-off."""
    rs = math.sqrt(s)
    p = math.sqrt(s / 4 - m**2)
    sin = math.sqrt(1 - cos**2)
    k1, k2 = np.array([rs / 2, 0, 0, p]), np.array([rs / 2, 0, 0, -p])
    k3 = rs / 2 * np.array([1.0, sin, 0, cos])
    k4 = rs / 2 * np.array([1.0, -sin, 0, -cos])
    total, ward, scale = 0.0, 0.0, 0.0
    for u, v in itertools.product(_spinors(k1, m, False), _spinors(k2, m, True)):
        vbar = v.conj() @ _G[0]
        args = (vbar, u, -k3, -k4)
        momenta = (k1 - k4, k1 - k3, k1 + k2, m)
        for e3, e4 in itertools.product(_transverse(k3), _transverse(k4)):
            amp = _amplitude(*args, e3, e4, *momenta)
            total += float(np.sum(np.abs(amp)**2))
            scale = max(scale, float(np.max(np.abs(amp))))
        for e4 in _transverse(k4):
            ward = max(ward, float(np.max(np.abs(
                _amplitude(*args, k3, e4, *momenta)))) / rs)
    return total, ward / scale


def _pair_production_oracle(s, cos, m):
    """``Σ|M|²`` for ``g(k1) g(k2) → Q(k3) Q̄(k4)``."""
    rs = math.sqrt(s)
    p = math.sqrt(s / 4 - m**2)
    sin = math.sqrt(1 - cos**2)
    k1, k2 = rs / 2 * np.array([1.0, 0, 0, 1.0]), rs / 2 * np.array([1.0, 0, 0, -1.0])
    k3 = np.array([rs / 2, p * sin, 0, p * cos])
    k4 = np.array([rs / 2, -p * sin, 0, -p * cos])
    total = 0.0
    for u, v in itertools.product(_spinors(k3, m, False), _spinors(k4, m, True)):
        ubar = u.conj() @ _G[0]
        for e1, e2 in itertools.product(_transverse(k1), _transverse(k2)):
            amp = _amplitude(ubar, v, k1, k2, e1, e2,
                             k3 - k1, k3 - k2, -k3 - k4, m)
            total += float(np.sum(np.abs(amp)**2))
    return total


def _engine_at(proc, s, cos):
    kin = proc.kinematics
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(proc.squared().subs({kin.s: s, kin.t: t}))


_POINTS = ((60.0**2, 0.3), (150.0**2, -0.65), (25.0**2, 0.8))


@pytest.mark.parametrize("m", [0, 9.0])
def test_quark_annihilation_to_gluons_vs_explicit_evaluator(qcd, m):
    calc, u = qcd["setup"](m_u=m)
    proc = calc.process((u.particle, u.antiparticle), (calc.gluon, calc.gluon),
                        numeric=True, extra={qcd["gs"]: _GS, qcd["ee"]: _EE})
    for s, cos in _POINTS:
        oracle, ward = _annihilation_oracle(s, cos, float(m))
        assert ward < 1e-9                  # the evaluator is gauge invariant
        engine = _engine_at(proc, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(oracle, rel=1e-8)


def test_heavy_quark_pair_production_vs_explicit_evaluator(qcd):
    """``gg→Q Q̄`` with a massive quark (the top-pair production shape)."""
    m = 9.0
    calc, u = qcd["setup"](m_u=m)
    proc = calc.process((calc.gluon, calc.gluon), (u.particle, u.antiparticle),
                        numeric=True, extra={qcd["gs"]: _GS, qcd["ee"]: _EE})
    for s, cos in _POINTS:
        engine = _engine_at(proc, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(
            _pair_production_oracle(s, cos, m), rel=1e-8)

"""Tier 4, stage 4b of the 2→2 scattering roadmap: boson exchange through a
bosonic three-point vertex — ``e⁺e⁻→Zh`` (``VVS``) and the gauge-cancellation
acid test ``e⁺e⁻→W⁺W⁻`` (``VVV``).

What is pinned, and against what:

- the triple-gauge rule (:data:`feynlag.pheno.boson_vertices.VVV_KAPPA`)
  against a direct differentiation of ``−¼F²`` done here;
- ``e⁺e⁻→W⁺W⁻`` against an explicit spinor + polarization-vector evaluator
  with the Feynman rules written out by hand;
- the **gauge cancellation itself**, exactly, at a rational parameter point:
  the sum of the three diagrams stays bounded at high energy while the ν
  exchange alone grows like ``s²`` — and a flipped triple-gauge sign brings
  the growth back;
- the MadGraph benchmark, 19.498 ± 0.058 pb at √s = 200 GeV
  (``docs/benchmark.md``);
- ``e⁺e⁻→Zh`` against the Higgsstrahlung closed form.
"""

import itertools
import math

import numpy as np
import pytest
import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_scaffold,
    fermion_gauge_current, to_physical_basis,
)
from feynlag.dirac import _dirac_rep
from feynlag.gauge_basis import adjoint_rotation, physical_vector_basis
from feynlag.pheno import (
    DiracParticle, ScatteringCalculator, VVV_KAPPA, line_amplitude,
)
from feynlag.pheno import boson_vertices
from feynlag.vertices.yangmills import structure_constants

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_G5 = np.array(_REP["g5"].evalf(), dtype=complex)
_PL, _PR = (np.eye(4) - _G5) / 2, (np.eye(4) + _G5) / 2
_MET = np.diag([1.0, -1.0, -1.0, -1.0])

# MadGraph's stock electroweak input scheme (docs/benchmark.md)
_AEW, _GF, _MZ, _WZ, _MH = 1 / 132.50698, 1.16639e-5, 91.1876, 2.4952, 125.0
_EE = math.sqrt(4 * math.pi * _AEW)
_MW = math.sqrt(_MZ**2 / 2 + math.sqrt(_MZ**4 / 4 - _AEW * math.pi * _MZ**2
                                       / (_GF * math.sqrt(2))))
_SW2 = 1 - _MW**2 / _MZ**2
_CW = math.sqrt(1 - _SW2)
_GW, _G1 = _EE / math.sqrt(_SW2), _EE / _CW
_VEV = 2 * _MW * math.sqrt(_SW2) / _EE
_GEV2_TO_PB = 0.3893794e9


@pytest.fixture(scope="module")
def sm():
    """SM electroweak sector, one massless lepton generation, the neutrino
    left-handed only."""
    ew = electroweak_scaffold(gw=_GW, g1=_G1, v=_VEV, mh=_MH)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)
    Le = WeylFermion("WLe", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["Wnu", "WeL"])
    eR = WeylFermion("WeR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["WeR"])
    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("Tier4b", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)
    e = DiracParticle("We", Le.components[1], eR.components[0], 0)
    nu = DiracParticle("Wnu_e", Le.components[0], None, 0)

    def calc(masses, widths=None, gauge_groups=(SU2L,)):
        return ScatteringCalculator(
            model, [e, nu], masses=masses, widths=widths,
            boson_fields=[ph.Z, ph.A, ph.Wp, ph.Wm, ph.h],
            conjugate_map=ph.cmap, conjugates={ph.Wp: ph.Wm, ph.Wm: ph.Wp},
            gauge_groups=list(gauge_groups))

    return dict(model=model, ew=ew, ph=ph, e=e, nu=nu, calc=calc)


def _physical(sm):
    ph, ew = sm["ph"], sm["ew"]
    masses = {ph.A: 0, ph.Z: _MZ, ph.Wp: _MW, ph.h: _MH}
    return masses, {ew.gw.s: _GW, ew.g1.s: _G1, ew.v.s: _VEV}


@pytest.fixture(scope="module")
def ww(sm):
    """``e⁺e⁻→W⁺W⁻`` at the benchmark point, ``Σ|M|²`` computed once."""
    masses, values = _physical(sm)
    ph, e = sm["ph"], sm["e"]
    proc = sm["calc"](masses, {ph.Z: _WZ}).process(
        (e.particle, e.antiparticle), (ph.Wp, ph.Wm),
        numeric=True, extra=values)
    proc.squared()
    return proc


# --------------------------------------------------------------------------
# the triple-gauge rule, from −¼F² directly
# --------------------------------------------------------------------------

def _mdot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def test_vvv_rule_from_direct_differentiation(sm):
    """``L₃ = −g f^{abc}(∂_μA^a_ν)A^{bμ}A^{cν}`` (the cubic part of ``−¼F²``
    for ``D = ∂ − igT^aA^a``).  With incoming plane waves ``∂_μ → −ip_μ`` the
    vertex ``i δ³L`` is ``−g Σ_π F_{π₁π₂π₃}(p_{π₁}·ε_{π₂})(ε_{π₁}·ε_{π₃})``
    over the six assignments of the legs to the three field slots, ``F`` the
    structure constants rotated to the physical basis.  It must equal
    ``κ × Vertex.coupling × [g^{μν}(p₁−p₂)^ρ + cyclic]`` with one ``κ`` for
    both ``AWW`` and ``ZWW``."""
    model, SU2L = sm["model"], sm["ew"].SU2L
    basis = physical_vector_basis(model, [SU2L])
    U, _ = adjoint_rotation(model, SU2L, basis)
    f = structure_constants(SU2L)
    R = sp.Rational
    eps = [(R(1, 3), R(2), R(-1, 2), R(5, 7)), (R(3, 2), R(-1, 4), R(2), R(1, 9)),
           (R(-2, 5), R(1), R(3, 4), R(-2))]
    mom = [(R(4), R(1, 2), R(-3), R(2, 3)), (R(-1), R(5, 2), R(1, 6), R(3))]
    mom.append(tuple(-a - b for a, b in zip(*mom)))          # Σp = 0

    vertices = model.gauge_vertices(groups=[SU2L], include=("VVV",))
    assert {tuple(map(str, v.particles)) for v in vertices} == {
        ("A", "Wm", "Wp"), ("Wm", "Wp", "Z")}
    for vertex in vertices:
        cols = [basis.index(sym) for sym in vertex.particles]
        direct = sp.S.Zero
        for perm in itertools.permutations(range(3)):
            F = sum(val * U[a, cols[perm[0]]] * U[b, cols[perm[1]]]
                    * U[c, cols[perm[2]]] for (a, b, c), val in f.items())
            direct += (-SU2L.g * F * _mdot(mom[perm[0]], eps[perm[1]])
                       * _mdot(eps[perm[0]], eps[perm[2]]))
        structure = sum(
            _mdot(eps[a], eps[b]) * _mdot(tuple(x - y for x, y in zip(mom[a], mom[b])),
                                          eps[c])
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
        assert sp.simplify(direct - VVV_KAPPA * vertex.coupling * structure) == 0
    assert VVV_KAPPA == -1


# --------------------------------------------------------------------------
# explicit spinor + polarization evaluator for e+e- -> W+W-
# --------------------------------------------------------------------------

def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


def _cm(s, cos, m3, m4):
    rs = math.sqrt(s)
    E3, E4 = (s + m3**2 - m4**2) / (2 * rs), (s + m4**2 - m3**2) / (2 * rs)
    pi, pf = rs / 2, math.sqrt(E3**2 - m3**2)
    sin = math.sqrt(1 - cos**2)
    return [np.array([rs / 2, 0, 0, pi]), np.array([rs / 2, 0, 0, -pi]),
            np.array([E3, pf * sin, 0, pf * cos]),
            np.array([E4, -pf * sin, 0, -pf * cos])]


def _massless_spinors(p, anti):
    M = _slash(p) @ _G[0]
    w, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    return [math.sqrt(w[j]) * vecs[:, j] for j in range(4) if w[j] > 1e-9 * p[0]]


def _polarizations(k, mass):
    n = k[1:] / np.linalg.norm(k[1:])
    e1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return [np.concatenate(([0.0], e1)), np.concatenate(([0.0], e2)),
            np.concatenate(([np.linalg.norm(k[1:])], k[0] * n)) / mass]


def _lower(v):
    return _MET @ v


def _ww_oracle(s, cos, tgc_sign=1.0, parts=("nu", "A", "Z")):
    """``Σ|M|²`` for ``e⁻(k1) e⁺(k2) → W⁺(k3) W⁻(k4)``, massless electron.

    Rules, written out: ``i(g/√2)γ^μP_L`` for ``ℓνW``; ``i(−e)γ^μ`` and
    ``i(g/c_w)γ^μ[(−½+s_w²)P_L + s_w²P_R]`` for the electron's γ and Z;
    ``i q̸/q²`` and ``−i g_{ρσ}/(q²−M²+iMΓ)`` propagators (the ``q_ρq_σ``
    term drops on a massless current); and the triple-gauge vertex
    ``i g_V [g^{μν}(p₁−p₂)^ρ + cyclic]`` for the field legs
    ``(V, W⁻, W⁺)`` with incoming momenta ``(k1+k2, −k3, −k4)`` — the W⁻
    *field* creates the outgoing W⁺ — and ``g_γ = e``, ``g_Z = g c_w``.
    """
    ks = _cm(s, cos, _MW, _MW)
    q = ks[0] + ks[1]
    total = 0.0
    for u, v, e3, e4 in itertools.product(
            _massless_spinors(ks[0], False), _massless_spinors(ks[1], True),
            _polarizations(ks[2], _MW), _polarizations(ks[3], _MW)):
        vbar = v.conj() @ _G[0]
        amp = 0
        if "nu" in parts:
            qn = ks[0] - ks[3]
            amp += (vbar @ (1j * _GW / math.sqrt(2) * _slash(e3) @ _PL)
                    @ (1j * _slash(qn) / (qn @ _MET @ qn))
                    @ (1j * _GW / math.sqrt(2) * _slash(e4) @ _PL) @ u)
        p1, p2, p3 = q, -ks[2], -ks[3]
        # T^ρ: legs (V: index ρ), (W⁻ field: ε3), (W⁺ field: ε4)
        tensor = (e3 * ((p1 - p2) @ _MET @ e4)
                  + (e3 @ _MET @ e4) * (p2 - p3)
                  + e4 * ((p3 - p1) @ _MET @ e3))
        gz = _GW / _CW
        for name, g_v, gl, gr, mass, width in (
                ("A", _EE, -_EE, -_EE, 0.0, 0.0),
                ("Z", _GW * _CW, gz * (-0.5 + _SW2), gz * _SW2, _MZ, _WZ)):
            if name not in parts:
                continue
            current = np.array([vbar @ (1j * _G[m] @ (gl * _PL + gr * _PR)) @ u
                                for m in range(4)])           # upper index
            prop = -1j / (s - mass**2 + 1j * mass * width)
            amp += tgc_sign * (current @ _MET @ tensor) * prop * 1j * g_v
        total += abs(amp)**2
    return total


def _engine_at(m2, kin, s, cos):
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(m2.subs({kin.s: s, kin.t: t}))


def test_ww_diagrams(ww):
    """The ν exchange and the γ and Z s channel — three diagrams."""
    kinds = sorted((d.kind, str(d.mediator)) for d in ww.diagrams)
    assert kinds == [("boson", "A"), ("boson", "Z"), ("fermion", "Wnu_e")]
    assert {d.channel for d in ww.diagrams if d.kind == "boson"} == {"s"}


@pytest.mark.parametrize("sqrt_s, cos", [(200.0, 0.3), (200.0, -0.8),
                                         (500.0, 0.55)])
def test_ww_matches_explicit_evaluator(ww, sqrt_s, cos):
    engine = _engine_at(ww.squared(), ww.kinematics, sqrt_s**2, cos)
    assert abs(engine.imag) < 1e-9 * abs(engine.real)
    assert engine.real == pytest.approx(_ww_oracle(sqrt_s**2, cos), rel=1e-8)


def test_ww_reproduces_madgraph(ww):
    """**The Tier-4 acceptance benchmark.**  MadGraph gives
    19.498 ± 0.058 pb at √s = 200 GeV (``docs/benchmark.md``; its default
    cuts do not act on W bosons, so this is the full-angle cross section).
    A flipped relative sign between the ν exchange and the triple-gauge
    diagrams gives ≈ 98 pb instead."""
    sigma = ww.numeric_cross_section(200.0**2) * _GEV2_TO_PB
    assert abs(sigma - 19.498) < 2 * 0.058
    assert sigma == pytest.approx(19.447, rel=1e-3)


def test_wrong_triple_gauge_sign_is_the_98_pb_failure(sm, monkeypatch):
    """Negative control at the benchmark point, engine and evaluator alike:
    with the opposite ``κ`` the cross section is several times larger."""
    monkeypatch.setattr(boson_vertices, "VVV_KAPPA", -boson_vertices.VVV_KAPPA)
    masses, values = _physical(sm)
    ph, e = sm["ph"], sm["e"]
    proc = sm["calc"](masses, {ph.Z: _WZ}).process(
        (e.particle, e.antiparticle), (ph.Wp, ph.Wm), numeric=True, extra=values)
    kin = proc.kinematics
    assert _engine_at(proc.squared(), kin, 200.0**2, 0.3).real == pytest.approx(
        _ww_oracle(200.0**2, 0.3, tgc_sign=-1.0), rel=1e-8)
    sigma = proc.numeric_cross_section(200.0**2) * _GEV2_TO_PB
    assert 90 < sigma < 105


def _degree_in_s(expr, s):
    """Asymptotic power of ``s`` of a rational function."""
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    return sp.degree(num, s) - sp.degree(den, s)


def test_gauge_cancellation_is_exact(sm, monkeypatch):
    """At ``g = 4, g' = 3, v = 1`` (so ``s_w = 3/5``, ``M_W = 2``,
    ``M_Z = 5/2``, all rational) and along ``t = −s/3``, ``Σ|M|²`` is an exact
    rational function of ``s``.  The ν exchange alone grows like ``s²``; the
    full amplitude is bounded (degree 0) — four powers of ``s`` cancel
    between the three diagrams' squares and interferences.  Flipping the
    triple-gauge sign loses the cancellation."""
    ph, ew, e = sm["ph"], sm["ew"], sm["e"]
    R = sp.Rational
    masses = {ph.A: 0, ph.Z: R(5, 2), ph.Wp: 2, ph.h: 1}
    values = {ew.gw.s: 4, ew.g1.s: 3, ew.v.s: 1}

    def process():
        return sm["calc"](masses).process(
            (e.particle, e.antiparticle), (ph.Wp, ph.Wm),
            numeric=True, extra=values)

    proc = process()
    kin = proc.kinematics
    line = {kin.t: -kin.s / 3}
    full = proc.squared().subs(line)
    assert not full.has(sp.Float)
    assert _degree_in_s(full, kin.s) == 0
    assert sp.limit(sp.cancel(sp.together(full)), kin.s, sp.oo) > 0

    nu_only = line_amplitude(
        proc.externals, [d for d in proc.diagrams if d.kind == "fermion"],
        kin).squared(kin).subs(line)
    assert _degree_in_s(nu_only, kin.s) == 2

    monkeypatch.setattr(boson_vertices, "VVV_KAPPA", -boson_vertices.VVV_KAPPA)
    flipped = process().squared().subs(line)
    assert _degree_in_s(flipped, kin.s) == 2


def test_charged_vectors_need_the_gauge_vertices(sm):
    """Without ``gauge_groups`` the triple-gauge vertices are not collected,
    and the ν exchange alone must not be returned."""
    masses, values = _physical(sm)
    ph, e = sm["ph"], sm["e"]
    calc = sm["calc"](masses, gauge_groups=())
    with pytest.raises(NotImplementedError, match="gauge_groups"):
        calc.process((e.particle, e.antiparticle), (ph.Wp, ph.Wm),
                     numeric=True, extra=values)


def test_stray_goldstone_vertices_are_dropped(sm):
    """With the Goldstones left out of ``boson_fields`` the extractor reports
    four-leg terms as three-leg "vertices" with a Goldstone in the coupling
    (``A h W⁺ ∝ G⁻``); the calculator evaluates at zero fields."""
    masses, _ = _physical(sm)
    fields = set(sm["model"].spin_map(conjugate_map=sm["ph"].cmap))
    vertices = sm["calc"](masses).vertices()
    assert all(not (sp.sympify(v.coupling).free_symbols & fields)
               for v in vertices)
    legs = {tuple(sorted(map(str, v.particles))) for v in vertices
            if v.vertex_type == "VVS"}
    assert legs == {("H0_r", "Wm", "Wp"), ("H0_r", "Z", "Z")}


# --------------------------------------------------------------------------
# e+e- -> Z h
# --------------------------------------------------------------------------

def test_higgsstrahlung_closed_form(sm):
    """``σ(e⁺e⁻→Zh) = G_F²M_Z⁴/(96πs) (v_e²+a_e²) √λ (λ + 12M_Z²/s)
    /(1 − M_Z²/s)²`` with ``λ = (1 − (M_h+M_Z)²/s)(1 − (M_h−M_Z)²/s)``,
    ``v_e = −1 + 4s_w²``, ``a_e = −1`` and the tree relation
    ``G_F = 1/(√2 v²)`` [Djouadi08, Eq. 4.18]."""
    masses, values = _physical(sm)
    ph, e = sm["ph"], sm["e"]
    proc = sm["calc"](masses).process(
        (e.particle, e.antiparticle), (ph.Z, ph.h), numeric=True, extra=values)
    assert [(d.kind, str(d.mediator), d.channel) for d in proc.diagrams] == [
        ("boson", "Z", "s")]
    s = 250.0**2
    gf = 1 / (math.sqrt(2) * _VEV**2)
    lam = (1 - (_MH + _MZ)**2 / s) * (1 - (_MH - _MZ)**2 / s)
    closed = (gf**2 * _MZ**4 / (96 * math.pi * s)
              * ((-1 + 4 * _SW2)**2 + 1) * math.sqrt(lam)
              * (lam + 12 * _MZ**2 / s) / (1 - _MZ**2 / s)**2)
    assert proc.numeric_cross_section(s) == pytest.approx(closed, rel=1e-7)

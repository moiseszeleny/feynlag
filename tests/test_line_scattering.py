"""Tier 4, stage 4a of the 2→2 scattering roadmap: two external fermions and
two external bosons on one open fermion line, fermion exchange
(:mod:`feynlag.pheno.line`, :mod:`feynlag.pheno.contraction`).

House style: pin physics, not code paths.  Two independent references:

- the textbook closed forms for ``e⁺e⁻→γγ`` and Compton scattering
  (Klein–Nishina) with a massive electron;
- an **explicit spinor + polarization-vector evaluator** (:func:`_oracle`):
  literal 4×4 matrices sandwiched between numerically built spinors,
  contracted with numerically built polarization vectors, amplitudes summed
  and squared.  No traces, no projector algebra, no polarization-sum
  tensors, so it shares nothing with the engine but the Dirac-basis matrices.
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
from feynlag.pheno import (
    DecayVertex, DiracParticle, ExternalBoson, ExternalFermion, ExternalState,
    Leg, LineAmplitude, ScatteringCalculator, TwoToTwoKinematics, VectorLeg,
    average_factor, enumerate_line_diagrams, fermion_exchange_term,
)
from feynlag.pheno.contraction import contract_trace

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_G5 = np.array(_REP["g5"].evalf(), dtype=complex)
_PL, _PR = (np.eye(4) - _G5) / 2, (np.eye(4) + _G5) / 2
_MET = np.diag([1.0, -1.0, -1.0, -1.0])


def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


# --------------------------------------------------------------------------
# contract_trace against explicit matrices
# --------------------------------------------------------------------------

class _FakeLeg:
    def __init__(self, momentum, mass, anti=False):
        self.momentum, self.mass, self.anti = momentum, sp.sympify(mass), anti


def test_contract_trace_against_explicit_matrices():
    """A chiral string with a spin sum, a propagator numerator, two γ's whose
    indices are tied by a loose metric and one γ contracted with a loose
    vector, times a vector·vector pair.  The momenta are confined to a
    three-dimensional subspace (no ``y`` component), so the γ₅ part — which
    :func:`contract_trace` discards — is exactly zero and the full explicit
    trace is the reference."""
    vec = {"p": np.array([3.0, 0.4, 0.0, 1.1]), "q": np.array([2.0, -0.7, 0.0, 0.3]),
           "r": np.array([1.5, 0.2, 0.0, -0.9]), "w": np.array([0.8, 0.1, 0.0, 0.5])}

    def dot(a, b):
        return sp.Float(vec[a] @ _MET @ vec[b])

    gl, gr, hl, hr, m, mq = 0.7, -0.3, 0.2, 1.1, 0.6, 0.9
    combo = ((sp.Float(1.0), "q"), (sp.Float(-2.0), "r"))
    factors = (("S", _FakeLeg("p", m)),
               ("G", "a", gl, gr),
               ("Q", combo, sp.Float(mq)),
               ("G", "b", hl, hr),
               ("G", "c", 1, 1))
    tensors = (("g", "a", "b"), ("v", ((1, "w"),), "c"),
               ("v", ((1, "p"),), "d"), ("v", ((1, "r"),), "d"))
    engine = complex(contract_trace(factors, tensors, dot))

    qq = vec["q"] - 2 * vec["r"]
    explicit = 0
    for a in range(4):
        explicit += _MET[a, a] * np.trace(
            (_slash(vec["p"]) + m * np.eye(4))
            @ _G[a] @ (gl * _PL + gr * _PR)
            @ (_slash(qq) + mq * np.eye(4))
            @ _G[a] @ (hl * _PL + hr * _PR)
            @ _slash(vec["w"]))
    explicit *= vec["p"] @ _MET @ vec["r"]
    assert abs(explicit.imag) < 1e-12
    assert engine.real == pytest.approx(explicit.real, rel=1e-12)
    assert abs(engine.imag) < 1e-12


def test_contract_trace_metric_trace_and_free_index():
    assert contract_trace((), (("g", "a", "a"),), None) == 16   # g^a_a Tr[1]
    with pytest.raises(ValueError, match="exactly twice"):
        contract_trace((("G", "a", 1, 1),), (), None)


# --------------------------------------------------------------------------
# closed forms: e+e- -> γγ and Compton, massive electron
# --------------------------------------------------------------------------

def _photon_vertex(e, index):
    return ("G", index, sp.I * e, sp.I * e)


def _annihilation(mass, e):
    """``e⁻(k1) e⁺(k2) → γ(k3) γ(k4)``."""
    kin = TwoToTwoKinematics(mass, mass, 0, 0)
    t_diagram = (fermion_exchange_term(
        _photon_vertex(e, "x4"), _photon_vertex(e, "x3"),
        ((1, kin.k1), (-1, kin.k3)), kin.t, mass),)
    u_diagram = (fermion_exchange_term(
        _photon_vertex(e, "x3"), _photon_vertex(e, "x4"),
        ((1, kin.k1), (-1, kin.k4)), kin.u, mass),)
    amp = LineAmplitude(Leg(kin.k2, mass, anti=True), Leg(kin.k1, mass),
                        (t_diagram, u_diagram),
                        (VectorLeg("x3", kin.k3, 0), VectorLeg("x4", kin.k4, 0)))
    return kin, amp, (t_diagram, u_diagram)


def test_pair_annihilation_closed_form():
    """``¼Σ|M|² = 2e⁴[p·k₂/p·k₁ + p·k₁/p·k₂ + 2m²(1/p·k₁ + 1/p·k₂)
    − m⁴(1/p·k₁ + 1/p·k₂)²]`` — the crossed Klein–Nishina form [PS95 §5.5];
    massless it is ``2e⁴(u/t + t/u)``."""
    e, m = sp.symbols("e m", positive=True)
    kin, amp, _ = _annihilation(0, e)
    assert sp.simplify(amp.squared(kin)
                       - 8 * e**4 * (kin.u / kin.t + kin.t / kin.u)) == 0

    kin, amp, _ = _annihilation(m, e)
    pk1, pk2 = (m**2 - kin.t) / 2, (m**2 - kin.u) / 2
    closed = 8 * e**4 * (pk2 / pk1 + pk1 / pk2 + 2 * m**2 * (1 / pk1 + 1 / pk2)
                         - m**4 * (1 / pk1 + 1 / pk2)**2)
    assert sp.simplify(amp.squared(kin) - closed) == 0


def test_compton_klein_nishina():
    """``e⁻(k1) γ(k2) → e⁻(k3) γ(k4)``:
    ``¼Σ|M|² = 2e⁴[p·k'/p·k + p·k/p·k' + 2m²(1/p·k − 1/p·k')
    + m⁴(1/p·k − 1/p·k')²]`` [PS95 §5.5]."""
    e, m = sp.symbols("e m", positive=True)
    kin = TwoToTwoKinematics(m, 0, m, 0)
    s_diagram = (fermion_exchange_term(
        _photon_vertex(e, "x4"), _photon_vertex(e, "x2"),
        ((1, kin.k1), (1, kin.k2)), kin.s, m),)
    u_diagram = (fermion_exchange_term(
        _photon_vertex(e, "x2"), _photon_vertex(e, "x4"),
        ((1, kin.k1), (-1, kin.k4)), kin.u, m),)
    amp = LineAmplitude(Leg(kin.k3, m), Leg(kin.k1, m), (s_diagram, u_diagram),
                        (VectorLeg("x2", kin.k2, 0), VectorLeg("x4", kin.k4, 0)))
    pk, pkp = (kin.s - m**2) / 2, (m**2 - kin.u) / 2
    closed = 8 * e**4 * (pkp / pk + pk / pkp + 2 * m**2 * (1 / pk - 1 / pkp)
                         + m**4 * (1 / pk - 1 / pkp)**2)
    assert sp.simplify(amp.squared(kin) - closed) == 0


def test_massless_vector_averages_over_two_polarizations():
    """The Compton initial state is 2 (electron) × 2 (photon) states."""
    electron = ExternalState("e", 1, sp.Rational(1, 2))
    photon = ExternalState("A", 0, 1)
    assert photon.spin_states() == 2
    assert ExternalState("Z", 91, 1).spin_states() == 3
    assert average_factor((electron, photon)) == sp.Rational(1, 4)


def test_ward_identity_guard_refuses_an_incomplete_amplitude():
    """One of the two ``e⁺e⁻→γγ`` diagrams alone is not gauge invariant:
    ``k·M ≠ 0``, so ``−g`` is not its polarization sum and the engine must
    refuse rather than return a number."""
    e, m = sp.symbols("e m", positive=True)
    kin, amp, (t_diagram, _) = _annihilation(m, e)
    broken = LineAmplitude(amp.out, amp.inn, (t_diagram,), amp.vectors)
    with pytest.raises(NotImplementedError, match="Ward identity"):
        broken.squared(kin)
    broken.squared(kin, ward_check=False)       # the escape hatch still runs


# --------------------------------------------------------------------------
# explicit spinor + polarization oracle
# --------------------------------------------------------------------------

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
    """``{u_s}`` (``Σuū = p̸+m``) or ``{v_s}`` (``Σvv̄ = p̸−m``)."""
    M = (_slash(p) + (-m if anti else m) * np.eye(4)) @ _G[0]
    w, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    return [math.sqrt(w[j]) * vecs[:, j] for j in range(4) if w[j] > 1e-9 * p[0]]


def _polarizations(k, mass):
    """Real polarization vectors (upper index): two transverse, plus the
    longitudinal ``(|k|, E k̂)/m`` for a massive vector."""
    n = k[1:] / np.linalg.norm(k[1:])
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    pols = [np.concatenate(([0.0], e1)), np.concatenate(([0.0], e2))]
    if mass != 0:
        pols.append(np.concatenate(([np.linalg.norm(k[1:])], k[0] * n)) / mass)
    return pols


def _vertex(eps, gl, gr):
    """``ε̸ (g_L P_L + g_R P_R)``."""
    return _slash(eps) @ (gl * _PL + gr * _PR)


def _oracle(s, cos, m_f, m_x, vector_masses, a, b):
    """``Σ|M|²`` for ``f(k1) f̄(k2) → V_a(k3) V_b(k4)`` by fermion exchange.

    ``a``/``b``: the ``(g_L, g_R)`` of each vector's vertex; ``m_x`` the
    exchanged fermion's mass.  Diagrams: ``V_a`` emitted next to ``u(k1)``
    (``q = k1 − k3``), and ``V_b`` emitted there (``q = k1 − k4``).
    """
    ks = _cm(s, cos, (m_f, m_f) + tuple(vector_masses))
    us, vs = _spinors(ks[0], m_f, False), _spinors(ks[1], m_f, True)
    total = 0.0
    for u, v, ea, eb in itertools.product(
            us, vs, _polarizations(ks[2], vector_masses[0]),
            _polarizations(ks[3], vector_masses[1])):
        vbar = v.conj() @ _G[0]
        qt, qu = ks[0] - ks[2], ks[0] - ks[3]
        amp = (vbar @ _vertex(eb, *b) @ (_slash(qt) + m_x * np.eye(4))
               @ _vertex(ea, *a) @ u) / (qt @ _MET @ qt - m_x**2)
        amp += (vbar @ _vertex(ea, *a) @ (_slash(qu) + m_x * np.eye(4))
                @ _vertex(eb, *b) @ u) / (qu @ _MET @ qu - m_x**2)
        total += abs(amp)**2
    return total


def _engine_at(m2, kin, s, cos):
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(m2.subs({kin.s: s, kin.t: t}))


_POINTS = ((300.0**2, 0.3), (250.0**2, -0.65))


def test_two_massive_vectors_chiral_massive_matches_oracle():
    """The general fermion-exchange case in one identity: a massive external
    fermion, a *different* exchanged-fermion mass, two distinct massive
    vectors with independent chiral couplings.  Exercises both polarization
    sums' ``k k/m²`` pieces, the propagator mass term and every chirality
    combination."""
    m_f, m_x, masses = 5.0, 17.0, (80.0, 91.0)
    a, b = (0.31, -0.12), (0.07, 0.45)
    kin = TwoToTwoKinematics(m_f, m_f, *masses)

    def G(index, c):
        return ("G", index, c[0], c[1])

    t_diagram = (fermion_exchange_term(G("x4", b), G("x3", a),
                                       ((1, kin.k1), (-1, kin.k3)), kin.t, m_x),)
    u_diagram = (fermion_exchange_term(G("x3", a), G("x4", b),
                                       ((1, kin.k1), (-1, kin.k4)), kin.u, m_x),)
    amp = LineAmplitude(Leg(kin.k2, m_f, anti=True), Leg(kin.k1, m_f),
                        (t_diagram, u_diagram),
                        (VectorLeg("x3", kin.k3, masses[0]),
                         VectorLeg("x4", kin.k4, masses[1])))
    m2 = amp.squared(kin)
    for s, cos in _POINTS:
        engine = _engine_at(m2, kin, s, cos)
        assert abs(engine.imag) < 1e-10 * abs(engine.real)
        assert engine.real == pytest.approx(
            _oracle(s, cos, m_f, m_x, masses, a, b), rel=1e-9)


# --------------------------------------------------------------------------
# from a model, through ScatteringCalculator
# --------------------------------------------------------------------------

_AEW, _GF, _MZ = 1 / 132.50698, 1.16639e-5, 91.1876
_EE = math.sqrt(4 * math.pi * _AEW)
_MW = math.sqrt(_MZ**2 / 2 + math.sqrt(_MZ**4 / 4 - _AEW * math.pi * _MZ**2
                                       / (_GF * math.sqrt(2))))
_SW2 = 1 - _MW**2 / _MZ**2
_GW, _G1 = _EE / math.sqrt(_SW2), _EE / math.sqrt(1 - _SW2)
_VEV = 2 * _MW * math.sqrt(_SW2) / _EE
_GZ = _EE / math.sqrt(_SW2 * (1 - _SW2))
_Z_E = (_GZ * (-0.5 + _SW2), _GZ * _SW2)        # textbook (g_L, g_R) of Z e e
_Z_NU = (_GZ * 0.5, 0.0)


@pytest.fixture(scope="module")
def sm():
    """SM electroweak sector with one massless lepton generation; the
    neutrino is declared **left-handed only** (``right=None``)."""
    ew = electroweak_scaffold(gw=_GW, g1=_G1, v=_VEV, mh=125.0)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)
    Le = WeylFermion("LLe", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=1, component_names=["Lnu", "LeL"])
    eR = WeylFermion("LeR", reps={U1Y: -1}, chirality="R", nflavors=1,
                     component_names=["LeR"])
    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("Tier4a", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR], parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)
    e = DiracParticle("Le", Le.components[1], eR.components[0], 0)
    nu = DiracParticle("Lnu_e", Le.components[0], None, 0)
    values = {ew.gw.s: _GW, ew.g1.s: _G1}

    def calc(bosons, particles=(e, nu)):
        return ScatteringCalculator(
            model, list(particles),
            masses={ph.A: 0, ph.Z: _MZ, ph.Wp: _MW},
            boson_fields=bosons, conjugate_map=ph.cmap,
            conjugates={ph.Wp: ph.Wm, ph.Wm: ph.Wp})

    return dict(ph=ph, e=e, nu=nu, values=values, calc=calc)


def test_left_handed_only_particle_maps_two_legs(sm):
    nu = sm["nu"]
    assert nu.bar_right is None
    assert set(nu.particle_map().values()) == {nu.particle, nu.antiparticle}
    assert len(nu.particle_map()) == 2


def test_ee_to_gamma_gamma_from_the_model(sm):
    """``e⁺e⁻→γγ`` with the photon coupling the model derives:
    ``Σ|M|² = 8e⁴(u/t + t/u)``, two diagrams (t and u), identical-particle
    factor ½, initial average ¼."""
    A, e = sm["ph"].A, sm["e"]
    proc = sm["calc"]([A, sm["ph"].Z]).process(
        (e.particle, e.antiparticle), (A, A), numeric=True, extra=sm["values"])
    assert sorted(d.channel for d in proc.diagrams) == ["t", "u"]
    assert all(d.mediator == e.particle for d in proc.diagrams)
    assert proc.symmetry_factor == sp.Rational(1, 2)
    kin = proc.kinematics
    for s, cos in _POINTS:
        t = float(kin.t_of_cos(cos).subs(kin.s, s))
        u = -s - t
        engine = _engine_at(proc.squared(), kin, s, cos)
        assert engine.real == pytest.approx(8 * _EE**4 * (u / t + t / u), rel=1e-9)
    # dσ/dcosθ = (2πα²/s)(1+cos²)/(1−cos²) × ½ [PS95 §5.5, massless limit]
    dsdcos = proc.differential_cross_section(variable="cos")
    s, cos = _POINTS[0]
    value = complex(dsdcos.subs(kin.t, kin.t_of_cos(cos)).subs(kin.s, s)).real
    assert value == pytest.approx(
        math.pi * _AEW**2 / s * (1 + cos**2) / (1 - cos**2), rel=1e-9)


@pytest.mark.parametrize("final", ["ZZ", "ZA"])
def test_ee_to_neutral_vectors_from_the_model(sm, final):
    """``e⁺e⁻→ZZ`` and ``e⁺e⁻→Zγ`` against the oracle evaluated with the
    textbook Z couplings ``g_{L,R} = (g/c_w)(T₃ − Q s_w²)`` — the model's
    extracted couplings never enter the reference."""
    ph, e = sm["ph"], sm["e"]
    second = ph.Z if final == "ZZ" else ph.A
    proc = sm["calc"]([ph.A, ph.Z]).process(
        (e.particle, e.antiparticle), (ph.Z, second),
        numeric=True, extra=sm["values"])
    assert len(proc.diagrams) == 2
    masses = (_MZ, _MZ if final == "ZZ" else 0.0)
    b = _Z_E if final == "ZZ" else (_EE, _EE)
    kin = proc.kinematics
    m2 = proc.squared()
    for s, cos in _POINTS:
        engine = _engine_at(m2, kin, s, cos)
        assert engine.real == pytest.approx(
            _oracle(s, cos, 0.0, 0.0, masses, _Z_E, b), rel=1e-8)


def test_neutrino_pair_to_zz_uses_the_left_handed_only_species(sm):
    ph, nu = sm["ph"], sm["nu"]
    proc = sm["calc"]([ph.A, ph.Z]).process(
        (nu.particle, nu.antiparticle), (ph.Z, ph.Z),
        numeric=True, extra=sm["values"])
    kin = proc.kinematics
    s, cos = _POINTS[0]
    assert _engine_at(proc.squared(), kin, s, cos).real == pytest.approx(
        _oracle(s, cos, 0.0, 0.0, (_MZ, _MZ), _Z_NU, _Z_NU), rel=1e-8)


def test_compton_from_the_model_initial_average(sm):
    """``e⁻γ→e⁻γ``: s- and u-channel electron, and the initial state averages
    over 2 × 2 states."""
    A, e = sm["ph"].A, sm["e"]
    proc = sm["calc"]([A, sm["ph"].Z]).process(
        (e.particle, A), (e.particle, A), numeric=True, extra=sm["values"])
    assert sorted(d.channel for d in proc.diagrams) == ["s", "u"]
    assert average_factor(proc.initial_states) == sp.Rational(1, 4)
    kin = proc.kinematics
    s, cos = _POINTS[0]
    t = float(kin.t_of_cos(cos).subs(kin.s, s))
    u = -s - t
    # massless Klein–Nishina: Σ|M|² = −8e⁴(u/s + s/u)
    assert _engine_at(proc.squared(), kin, s, cos).real == pytest.approx(
        -8 * _EE**4 * (u / s + s / u), rel=1e-9)


def test_ww_is_refused_without_the_triple_gauge_vertices(sm):
    """``e⁺e⁻→W⁺W⁻`` has the ν exchange *and* the γ/Z s channel; this
    calculator was built without ``gauge_groups``, so the triple-gauge
    vertices are missing and returning the ν exchange alone would be the
    unitarity-violating wrong answer.  (``tests/test_ww.py`` computes it.)"""
    ph, e = sm["ph"], sm["e"]
    calc = sm["calc"]([ph.A, ph.Z, ph.Wp, ph.Wm])
    with pytest.raises(NotImplementedError, match="triple-gauge"):
        calc.process((e.particle, e.antiparticle), (ph.Wp, ph.Wm),
                     numeric=True, extra=sm["values"])


def test_calculator_external_guards(sm):
    ph, e = sm["ph"], sm["e"]
    calc = sm["calc"]([ph.A, ph.Z])
    with pytest.raises(KeyError, match="boson_fields"):
        calc.process((e.particle, e.antiparticle), (sp.Symbol("X"), ph.A))
    with pytest.raises(NotImplementedError, match="external"):
        calc.process((e.particle, ph.A), (ph.A, ph.A))
    with pytest.raises(ValueError, match="fermion number"):
        calc.process((e.particle, ph.A), (e.antiparticle, ph.A))


def test_undeclared_exchanged_fermion_raises():
    """A vertex that would complete a diagram through a fermion no
    ``DiracParticle`` claims must not be dropped silently."""
    e, ebar, X, raw_bar, raw = sp.symbols("e ebar X rawbar raw")
    electron = type("P", (), {"particle": e, "antiparticle": ebar, "mass": 0})()
    externals = [ExternalFermion(electron, False, True),
                 ExternalFermion(electron, True, True),
                 ExternalBoson(X, 1, 10, False), ExternalBoson(X, 1, 10, False)]
    kin = TwoToTwoKinematics(0, 0, 10, 10)
    vertices = [DecayVertex((raw_bar, e, X), "FFV", g_left=1, g_right=1),
                DecayVertex((ebar, raw, X), "FFV", g_left=1, g_right=1)]
    with pytest.raises(ValueError, match="no declared"):
        enumerate_line_diagrams(externals, vertices, kin, species=[electron])

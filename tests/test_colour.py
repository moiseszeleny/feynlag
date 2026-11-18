"""Tier 5, stage 5a of the 2→2 scattering roadmap: coloured four-fermion
processes.

Colour is factorised: every diagram is a colour tensor built from the gauge
group's own generators times the Lorentz amplitude with a colour-stripped
coupling, and the colour-summed ``|M|²`` is ``Σ C_{dd'} L_d L̄_{d'}``
(:mod:`feynlag.pheno.colour`).

What is pinned, and against what:

- the colour matrices against the SU(N) trace identities, for N = 2, 3, 4;
- the four four-quark entries of the parton table of Ellis, Stirling and
  Webber, *QCD and Collider Physics*, Table 7.1 — exactly and symbolically,
  with the couplings extracted from a QCD × QED model;
- QCD and QED together, with massive quarks, against an explicit evaluator
  that sums the amplitude over diagrams for each colour configuration with
  explicit ``T^a`` matrices and explicit spinors (no colour matrix, no trace);
- the colour-singlet limits ``σ(e⁺e⁻→q q̄) = N_c Q_q² · 4πα²/(3s)`` and
  ``σ(q q̄→e⁺e⁻)`` with its ``1/N_c``.
"""

import itertools
import math

import numpy as np
import pytest
import sympy as sp

from feynlag import (
    ExternalParameter, Lagrangian, Model, SU3, SUN, U1, WeylFermion,
    fermion_gauge_current,
)
from feynlag.dirac import _dirac_rep
from feynlag.pheno import DiracParticle, ScatteringCalculator
from feynlag.pheno.engine import (
    Amplitude, colour_matrix, four_fermion_colour_tensor,
)

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_MET = np.diag([1.0, -1.0, -1.0, -1.0])

_GS, _EE = 1.2, 0.3
_QU, _QD = sp.Rational(2, 3), -sp.Rational(1, 3)


@pytest.fixture(scope="module")
def qcd():
    gs = ExternalParameter("gs5", _GS, positive=True)
    ee = ExternalParameter("ee5", _EE, positive=True)
    SU3c, U1em = SU3("SU3c5", coupling=gs), U1("U1em5", coupling=ee)

    def quark(name, charge):
        return [WeylFermion(f"c{name}{chi}", reps={SU3c: 3, U1em: charge},
                            chirality=chi, nflavors=1,
                            component_names=[f"c{name}{chi}_{k}"
                                             for k in (1, 2, 3)])
                for chi in ("L", "R")]

    uL, uR = quark("u", _QU)
    dL, dR = quark("d", _QD)
    eL = WeylFermion("ceL", reps={U1em: -1}, chirality="L", nflavors=1,
                     component_names=["ceL"])
    eR = WeylFermion("ceR", reps={U1em: -1}, chirality="R", nflavors=1,
                     component_names=["ceR"])
    G, A = SU3c.bosons("G"), U1em.bosons("A")
    i = sp.Symbol("i", integer=True)
    L = Lagrangian()
    for f in (uL, uR, dL, dR, eL, eR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("QCDxQED", gauge_groups=[SU3c, U1em],
                  fields=[uL, uR, dL, dR, eL, eR, G, A],
                  parameters=[gs, ee], lagrangian=L)
    photon = A.components[0]
    e = DiracParticle("ce", eL.components[0], eR.components[0], 0)

    def setup(m_u=0, m_d=0, component=0, **kwargs):
        u = DiracParticle("cu", uL.components[component],
                          uR.components[component], m_u, color=3)
        d = DiracParticle("cd", dL.components[component],
                          dR.components[component], m_d, color=3)
        kwargs.setdefault("colour_group", SU3c)
        kwargs.setdefault("gluon", G)
        calc = ScatteringCalculator(model, [u, d, e], masses={photon: 0},
                                    boson_fields=[photon], **kwargs)
        return calc, u, d

    return dict(model=model, SU3c=SU3c, G=G, A=photon, e=e, gs=gs.s, ee=ee.s,
                setup=setup, uL=uL, uR=uR)


# --------------------------------------------------------------------------
# colour algebra
# --------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3, 4])
def test_four_quark_colour_matrix_is_the_trace_identities(N):
    """t-channel × t-channel is ``Tr(T^aT^b)Tr(T^aT^b) = (N²−1)/4``; the
    t × u (or s × t) interference is ``Tr(T^aT^bT^aT^b) = −(N²−1)/(4N)``;
    a singlet exchange gives ``N²`` and ``N``; a singlet s channel times an
    adjoint s channel vanishes (``Tr T^a = 0``).  Nothing is typed in."""
    group = SUN(N, f"SU{N}col")
    T = group.generators(N)
    dims = [N] * 4
    t_lines, u_lines, s_lines = ((2, 0), (3, 1)), ((3, 0), (2, 1)), ((1, 0), (2, 3))
    gt = four_fermion_colour_tensor(t_lines, dims, (T, T))
    gu = four_fermion_colour_tensor(u_lines, dims, (T, T))
    st = four_fermion_colour_tensor(t_lines, dims)
    su = four_fermion_colour_tensor(u_lines, dims)
    C = colour_matrix([gt, gu, st, su])
    R = sp.Rational
    assert C[0][0] == C[1][1] == R(N**2 - 1, 4)
    assert C[0][1] == C[1][0] == -R(N**2 - 1, 4 * N)
    assert C[2][2] == C[3][3] == N**2
    assert C[2][3] == N
    assert C[0][2] == 0                       # Tr(T^a) Tr(T^a)
    assert C[0][3] == R(N**2 - 1, 2)          # Tr(T^aT^a) = C_F N
    gs_ = four_fermion_colour_tensor(s_lines, dims, (T, T))
    ss = four_fermion_colour_tensor(s_lines, dims)
    assert colour_matrix([gs_, ss])[0][1] == 0


def test_colour_tensor_guards():
    T = SU3("SU3guard").generators(3)
    with pytest.raises(ValueError, match="colour dimensions"):
        four_fermion_colour_tensor(((2, 0), (3, 1)), [3, 1, 1, 3])
    with pytest.raises(ValueError, match="generators of shape"):
        four_fermion_colour_tensor(((2, 0), (3, 1)), [1, 3, 1, 3], (T, T))
    with pytest.raises(ValueError, match="one row per diagram"):
        Amplitude(()).squared(None, colour=[[1]])


# --------------------------------------------------------------------------
# the stripped gluon vertex
# --------------------------------------------------------------------------

@pytest.mark.parametrize("component", [0, 1, 2])
def test_gluon_vertex_is_stripped_to_one_mediator(qcd, component):
    """Whichever colour component the species is declared on, the eight
    component vertices collapse to one ``q̄ q g`` vertex with ``i g_s``."""
    calc, u, d = qcd["setup"](component=component)
    gluon = [v for v in calc.vertices()
             if v.vertex_type == "FFV" and v.particles[2] == calc.gluon]
    assert {(str(v.particles[0]), str(v.particles[1])) for v in gluon} == {
        ("cubar", "cu"), ("cdbar", "cd")}
    for v in gluon:
        assert sp.simplify(v.g_left - sp.I * qcd["gs"]) == 0
        assert sp.simplify(v.g_right - sp.I * qcd["gs"]) == 0
    components = set(qcd["G"].components)
    assert not any(v.particles[2] in components for v in calc.vertices())


def test_mismatched_colour_components_raise(qcd):
    """Left and right legs on different colour components do not form
    ``g T^a`` for one component."""
    uL, uR = qcd["uL"], qcd["uR"]
    bad = DiracParticle("cubad", uL.components[0], uR.components[1], 0, color=3)
    calc = ScatteringCalculator(qcd["model"], [bad], masses={qcd["A"]: 0},
                                boson_fields=[qcd["A"]],
                                colour_group=qcd["SU3c"], gluon=qcd["G"])
    with pytest.raises(ValueError, match="single colour component"):
        calc.vertices()


# --------------------------------------------------------------------------
# the parton table
# --------------------------------------------------------------------------

def _averaged_qcd(qcd, incoming, outgoing):
    """``Σ̄|M|²/g_s⁴`` with the photon switched off."""
    calc, u, d = qcd["setup"]()
    names = {"u": u.particle, "ubar": u.antiparticle,
             "d": d.particle, "dbar": d.antiparticle}
    proc = calc.process([names[n] for n in incoming],
                        [names[n] for n in outgoing])
    summed = proc.squared().subs(qcd["ee"], 0)
    assert [s.dof() for s in proc.initial_states] == [6, 6]
    return proc.kinematics, sp.simplify(summed / 36 / qcd["gs"]**4)


def test_parton_table_four_quark_entries(qcd):
    """Ellis–Stirling–Webber Table 7.1 (massless, averaged over initial and
    summed over final spins and colours, ``g_s⁴`` removed)."""
    R = sp.Rational
    kin, value = _averaged_qcd(qcd, ("u", "d"), ("u", "d"))
    s, t, u = kin.s, kin.t, kin.u
    assert sp.simplify(value - R(4, 9) * (s**2 + u**2) / t**2) == 0

    kin, value = _averaged_qcd(qcd, ("u", "dbar"), ("u", "dbar"))
    assert sp.simplify(value - R(4, 9) * (s**2 + u**2) / t**2) == 0

    kin, value = _averaged_qcd(qcd, ("u", "ubar"), ("d", "dbar"))
    assert sp.simplify(value - R(4, 9) * (t**2 + u**2) / s**2) == 0

    kin, value = _averaged_qcd(qcd, ("u", "u"), ("u", "u"))
    assert sp.simplify(value - (
        R(4, 9) * ((s**2 + u**2) / t**2 + (s**2 + t**2) / u**2)
        - R(8, 27) * s**2 / (t * u))) == 0

    kin, value = _averaged_qcd(qcd, ("u", "ubar"), ("u", "ubar"))
    assert sp.simplify(value - (
        R(4, 9) * ((s**2 + u**2) / t**2 + (t**2 + u**2) / s**2)
        - R(8, 27) * u**2 / (s * t))) == 0


def test_colour_matrix_of_same_flavour_scattering(qcd):
    """γ and g in both channels of ``u ū → u ū``: every entry is one of the
    trace identities, and the singlet × octet s channel is exactly zero."""
    calc, u, _ = qcd["setup"]()
    proc = calc.process((u.particle, u.antiparticle),
                        (u.particle, u.antiparticle))
    order = [(d.channel, str(d.mediator)) for d in proc.diagrams]
    assert sorted(order) == [("s", "A"), ("s", "G"), ("t", "A"), ("t", "G")]
    C = {(a, b): proc.colour[i][j]
         for i, a in enumerate(order) for j, b in enumerate(order)}
    R = sp.Rational
    assert C[("s", "A"), ("s", "A")] == 9 and C[("s", "G"), ("s", "G")] == 2
    assert C[("s", "A"), ("s", "G")] == 0
    assert C[("s", "A"), ("t", "A")] == 3
    assert C[("s", "A"), ("t", "G")] == 4
    assert C[("s", "G"), ("t", "G")] == -R(2, 3)


def test_quark_pair_cross_section(qcd):
    """``σ(q q̄→q′q̄′) = 8πα_s²/(27 s)``, the integral of the table entry."""
    calc, u, d = qcd["setup"]()
    proc = calc.process((u.particle, u.antiparticle),
                        (d.particle, d.antiparticle), numeric=True,
                        extra={qcd["gs"]: _GS, qcd["ee"]: 0})
    s = 150.0**2
    alpha_s = _GS**2 / (4 * math.pi)
    assert proc.numeric_cross_section(s) == pytest.approx(
        8 * math.pi * alpha_s**2 / (27 * s), rel=1e-9)


# --------------------------------------------------------------------------
# colour-singlet limits
# --------------------------------------------------------------------------

def test_lepton_quark_processes_carry_nc_once(qcd):
    """``σ(e⁺e⁻→u ū) = N_c Q_u² · 4πα²/(3s)`` (the R ratio) and
    ``σ(u ū→e⁺e⁻) = Q_u²/N_c · 4πα²/(3s)``."""
    calc, u, _ = qcd["setup"]()
    e, ee = qcd["e"], qcd["ee"]
    point = 4 * sp.pi * (ee**2 / (4 * sp.pi))**2 / (3 * sp.Symbol("s", positive=True))

    prod = calc.process((e.particle, e.antiparticle),
                        (u.particle, u.antiparticle))
    assert [(d.channel, str(d.mediator)) for d in prod.diagrams] == [("s", "A")]
    s = prod.kinematics.s
    assert sp.simplify(prod.cross_section()
                       - 3 * _QU**2 * point.subs(sp.Symbol("s", positive=True), s)) == 0

    drell_yan = calc.process((u.particle, u.antiparticle),
                             (e.particle, e.antiparticle))
    s = drell_yan.kinematics.s
    assert sp.simplify(drell_yan.cross_section()
                       - _QU**2 / 3 * point.subs(sp.Symbol("s", positive=True), s)) == 0


# --------------------------------------------------------------------------
# explicit spinors × explicit T^a
# --------------------------------------------------------------------------

def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


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


def _current_product(bar_a, u_a, bar_b, u_b):
    """``[ā γ^μ a][b̄ γ_μ b]``."""
    return sum(_MET[m, m] * (bar_a @ _G[m] @ u_a) * (bar_b @ _G[m] @ u_b)
               for m in range(4))


def _uubar_oracle(s, cos, m, charge):
    """``Σ|M|²`` for ``u(k1,c1) ū(k2,c2) → u(k3,c3) ū(k4,c4)`` with photon
    and gluon in the s and t channels.

    Vertices ``i g_s γ^μ T^a`` and ``i e Q γ^μ`` (``D = ∂ − i g T^a A^a``),
    propagator ``−i g_{μν}/q²``, relative fermion sign ``−1`` between the
    two pairings.  The amplitude is summed over diagrams for each spin and
    colour configuration, then squared.
    """
    T = [np.array(Ta.evalf(), dtype=complex)
         for Ta in SU3("SU3oracle").generators(3)]
    ks = _cm(s, cos, (m, m, m, m))
    t = (ks[0] - ks[2]) @ _MET @ (ks[0] - ks[2])
    delta = np.eye(3)
    TT = sum(np.einsum("ij,kl->ijkl", Ta, Ta) for Ta in T)
    DD = np.einsum("ij,kl->ijkl", delta, delta)
    strong, em = (1j * _GS)**2 * (-1j), (1j * _EE * charge)**2 * (-1j)
    total = 0.0
    for u1, v2, u3, v4 in itertools.product(
            _spinors(ks[0], m, False), _spinors(ks[1], m, True),
            _spinors(ks[2], m, False), _spinors(ks[3], m, True)):
        v2bar, u3bar = v2.conj() @ _G[0], u3.conj() @ _G[0]
        lorentz_s = _current_product(v2bar, u1, u3bar, v4) / s
        lorentz_t = _current_product(u3bar, u1, v2bar, v4) / t
        # colour indices (c1, c2, c3, c4):  s: [c2,c1][c3,c4]   t: [c3,c1][c2,c4]
        amp = (lorentz_s * (strong * np.einsum("bacd->abcd", TT)
                            + em * np.einsum("bacd->abcd", DD))
               - lorentz_t * (strong * np.einsum("cabd->abcd", TT)
                              + em * np.einsum("cabd->abcd", DD)))
        total += float(np.sum(np.abs(amp)**2))
    return total


@pytest.mark.parametrize("m", [0, 7.0])
def test_qcd_plus_qed_vs_explicit_colour_sum(qcd, m):
    """Massive quarks, both mediators, both channels — sixteen weighted
    terms in the engine against a brute-force colour and spin sum."""
    calc, u, _ = qcd["setup"](m_u=m)
    proc = calc.process((u.particle, u.antiparticle),
                        (u.particle, u.antiparticle), numeric=True,
                        extra={qcd["gs"]: _GS, qcd["ee"]: _EE})
    kin = proc.kinematics
    for rs, cos in ((80.0, 0.3), (200.0, -0.65), (35.0, 0.8)):
        s = rs**2
        t = kin.t_of_cos(cos).subs(kin.s, s)
        engine = complex(proc.squared().subs({kin.s: s, kin.t: t}))
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(
            _uubar_oracle(s, cos, float(m), float(_QU)), rel=1e-9)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_coloured_external_needs_the_colour_group(qcd):
    calc, u, d = qcd["setup"](colour_group=None, gluon=None)
    with pytest.raises(NotImplementedError, match="colour_group"):
        calc.process((u.particle, u.antiparticle), (d.particle, d.antiparticle))
    with pytest.raises(ValueError, match="together"):
        qcd["setup"](gluon=None)


def test_gluon_component_is_not_an_external_state(qcd):
    """An external gluon is ``calc.gluon`` (stage 5b, ``test_gluons.py``);
    one colour component of it is refused."""
    calc, u, _ = qcd["setup"]()
    component = qcd["G"].components[0]
    with pytest.raises(KeyError, match="colour component"):
        calc.process((u.particle, u.antiparticle), (component, component))


def test_declared_colour_must_match_the_field(qcd):
    uL, uR = qcd["uL"], qcd["uR"]
    wrong = DiracParticle("cuwrong", uL.components[0], uR.components[0], 0,
                          color=2)
    calc = ScatteringCalculator(qcd["model"], [wrong], masses={qcd["A"]: 0},
                                boson_fields=[qcd["A"]],
                                colour_group=qcd["SU3c"], gluon=qcd["G"])
    with pytest.raises(ValueError, match="color=2"):
        calc.vertices()


def test_float_masses_do_not_trip_the_epsilon_guard():
    """With float masses the Gram determinant of four dependent momenta is a
    round-off residue (``3e-17·s³t²``), not ``0``; the guard must read it as
    singular — and must still see a genuinely non-singular float table."""
    from feynlag.pheno.engine import TwoToTwoKinematics
    from feynlag.pheno.epsilon import (
        _float_singular, assert_epsilon_single_vanishes,
    )
    kin = TwoToTwoKinematics(7.0, 7.0, 7.0, 7.0)
    heads = [kin.k1, kin.k2, kin.k3, kin.k4]
    assert_epsilon_single_vanishes(heads, kin.dot)

    generic = {frozenset((a, b)): 1.0 + 0.37 * i * j + (2.0 if a is b else 0.0)
               for i, a in enumerate(heads, 1) for j, b in enumerate(heads, 1)}
    assert not _float_singular(heads, lambda a, b: sp.Float(generic[frozenset((a, b))]))
    with pytest.raises(NotImplementedError, match="Gram determinant"):
        assert_epsilon_single_vanishes(
            heads, lambda a, b: sp.Float(generic[frozenset((a, b))]))

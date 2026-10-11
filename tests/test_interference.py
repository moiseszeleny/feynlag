"""Tier 3 of the 2→2 scattering roadmap: interference, s/t/u topologies,
:class:`~feynlag.pheno.ScatteringCalculator`.

House style: pin physics, not code paths.  The independent oracle here is an
**explicit-spinor helicity evaluator** (:func:`_oracle`): it builds the
spinors ``u``/``v`` themselves (as the eigenvectors of ``(p̸±m)γ⁰``), sandwiches
literal 4×4 vertex matrices between them, sums the diagrams' *amplitudes*
with explicit propagators, and squares — no traces, no projector algebra, no
ε identities, so it shares nothing with :mod:`feynlag.pheno.diagrams` except
the Dirac-basis matrices themselves.  The Wick sign of each oracle diagram is
written in by hand from the textbook rule (Bhabha s vs t: −1; Møller t vs u:
−1); the closed forms and the Fierz identity below pin those signs from the
physics side as well.
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
    Amplitude, BosonPropagator, ChainVertex, Diagram, DiracParticle, Leg,
    ScatteringCalculator, SpinorChain, TwoToTwoKinematics,
)
from feynlag.pheno.diagrams import (
    _chain_indices, _contract_pairs, _slash_trace, _string_tensor,
)
from feynlag.pheno.lorentz import contract_to_dots, momentum
from feynlag.verify import numeric_equal
from sympy.physics.hep.gamma_matrices import LorentzIndex


# --------------------------------------------------------------------------
# explicit-spinor oracle — shares no code with the covariant engine
# --------------------------------------------------------------------------

_REP = _dirac_rep()
_G = [np.array(_REP[("g", m)].evalf(), dtype=complex) for m in range(4)]
_G5 = np.array(_REP["g5"].evalf(), dtype=complex)
_PL, _PR = (np.eye(4) - _G5) / 2, (np.eye(4) + _G5) / 2
_MET = np.diag([1.0, -1.0, -1.0, -1.0])


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


def _slash(p):
    return sum(_MET[m, m] * p[m] * _G[m] for m in range(4))


def _spinors(p, m, anti):
    """``{u_s}`` (``Σuū = p̸+m``) or ``{v_s}`` (``Σvv̄ = p̸−m``): the
    eigenvectors of the positive-semidefinite ``(p̸±m)γ⁰ = Σ w w†``."""
    M = (_slash(p) + (-m if anti else m) * np.eye(4)) @ _G[0]
    w, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    return [math.sqrt(w[j]) * vecs[:, j] for j in range(4) if w[j] > 1e-9 * p[0]]


def _oracle(s, cos, masses, anti, diagrams):
    """``Σ_spins |Σ_d M_d|²`` at one CM point.

    ``diagrams``: dicts with ``sign``, ``chains`` = two
    ``(bar_leg, field_leg, 'V'|'S', g_L, g_R)``, ``spin``, ``M``, ``W`` and
    ``q`` = ``[(leg, ±1), …]``.  A ``bar_leg`` spinor enters as ``ψ†γ⁰``.
    """
    ks = _cm(s, cos, masses)
    basis = [_spinors(ks[j], masses[j], anti[j]) for j in range(4)]
    total = 0.0
    for helicities in itertools.product(range(2), repeat=4):
        spin = [basis[j][h] for j, h in enumerate(helicities)]
        amp = 0
        for d in diagrams:
            currents = []
            for bar, field, structure, gl, gr in d["chains"]:
                left = spin[bar].conj() @ _G[0]
                proj = gl * _PL + gr * _PR
                if structure == "V":
                    currents.append(np.array(
                        [left @ _G[m] @ proj @ spin[field] for m in range(4)]))
                else:
                    currents.append(left @ proj @ spin[field])
            q = sum(sign * ks[leg] for leg, sign in d["q"])
            den = q @ _MET @ q - d["M"]**2 + 1j * d["M"] * d["W"]
            if d["spin"] == 1:
                ql = _MET @ q
                num = _MET if d["M"] == 0 else _MET - np.outer(ql, ql) / d["M"]**2
                amp += d["sign"] * (currents[0] @ num @ currents[1]) * (-1j) / den
            else:
                amp += d["sign"] * currents[0] * currents[1] * 1j / den
        total += abs(amp)**2
    return total


def _engine_at(m2, kin, s, cos):
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(m2.subs({kin.s: s, kin.t: t}))


def _chain(out, inn, structure, gl, gr, tag):
    if structure == "V":
        mu, mu_bar = _chain_indices(tag)
        idx, bar_idx = (mu,), (mu_bar,)
    else:
        idx = bar_idx = ()
    return SpinorChain(out=out, inn=inn,
                       vertices=(ChainVertex(structure, gl, gr),),
                       indices=idx, bar_indices=bar_idx)


def _assert_matches_oracle(m2, kin, masses, anti, oracle_diagrams,
                           points=((200.0, 0.3), (130.0, -0.7))):
    for s, cos in points:
        engine = _engine_at(m2, kin, s, cos)
        oracle = _oracle(s, cos, masses, anti, oracle_diagrams)
        assert abs(engine.imag) < 1e-12 * abs(engine.real)
        assert engine.real == pytest.approx(oracle, rel=1e-10)


# --------------------------------------------------------------------------
# the generic chiral-string engine reproduces the hand-derived Tier-1/2 trace
# --------------------------------------------------------------------------

@pytest.mark.parametrize("structure", ["V", "S"])
def test_cross_trace_with_itself_is_the_hand_derived_trace(structure):
    """``SpinorChain.cross_trace(self)`` — the mechanical projector algebra
    every interference term uses — equals :meth:`SpinorChain.trace` (the
    hand-derived γ₅-free trace) and :meth:`SpinorChain.epsilon_structure`
    for massive legs and complex, independent chiral couplings."""
    m1, m2 = sp.symbols("m1 m2", positive=True)
    gl, gr = sp.symbols("g_L g_R")
    kin = TwoToTwoKinematics(0, 0, m1, m2)
    chain = _chain(Leg(kin.k3, m1), Leg(kin.k4, m2, anti=True), structure,
                   gl, gr, f"self{structure}")
    plain, eps4 = chain.cross_trace(chain, *(chain.indices + chain.bar_indices
                                             or (None, None)))
    difference = chain.trace() - plain
    if structure == "V":
        mu, mu_bar = chain.indices[0], chain.bar_indices[0]
        difference = contract_to_dots(
            (difference * kin.k1(-mu) * kin.k2(-mu_bar))
            .contract_metric(LorentzIndex.metric), kin.dot)
        (prefactor, slots), = eps4
        legacy_prefactor, legacy_slots = chain.epsilon_structure()
        assert slots == legacy_slots
        values = {gl: 0.3 + 0.7j, gr: -1.1 + 0.2j}
        assert complex((prefactor - legacy_prefactor).subs(values)) == pytest.approx(0)
    else:
        difference = contract_to_dots(difference, kin.dot)
        assert eps4 == []
    for values in ({gl: 0.3 + 0.7j, gr: -1.1 + 0.2j},
                   {gl: 2.0, gr: -0.4 - 0.9j}):
        point = {m1: 1.3, m2: 0.7, kin.s: 11.0, kin.t: -2.5, **values}
        assert complex(sp.sympify(difference).subs(point)) == pytest.approx(0, abs=1e-12)


def test_repeated_momentum_is_refused_before_sympy_traces_it():
    """SymPy 1.14's ``gamma_trace`` is wrong when a momentum is slashed twice:
    ``Tr[q̸γ^a p̸p̸γ^b q̸]g_ab = p²Tr[q̸γ^aγ_a q̸] = 16p²q²``, but it returns
    ``72p²q²``.  The engine refuses such a string instead of tracing it, and
    the scalar route used for the exchange trace gets it right."""
    p, q = momentum("rp_p"), momentum("rp_q")
    P2, Q2, PQ = sp.symbols("P2 Q2 PQ")

    def dot(a, b):
        return {("rp_p", "rp_p"): P2, ("rp_q", "rp_q"): Q2}.get(
            tuple(sorted((a.name, b.name))), PQ)

    from feynlag.pheno.lorentz import index
    rho = index("rp_rho")
    atoms = (("m", q), ("i", rho), ("m", p), ("m", p), ("i", -rho), ("m", q))
    with pytest.raises(ValueError, match="repeated momentum"):
        _string_tensor(atoms)
    reduced = _contract_pairs(atoms)
    value = sum(c * _slash_trace(tuple(h for _, h in r), dot, {}) for c, r in reduced)
    assert sp.expand(value - 16 * P2 * Q2) == 0


def _explicit_trace(atoms, vectors):
    """``Tr[atoms]`` with literal 4×4 matrices: a slash of a numeric vector,
    or ``γ^ρ … γ_ρ`` summed over ``ρ`` with the metric."""
    names = sorted({str(v) for kind, v in atoms if kind == "i" and v.is_up})
    total = 0
    for values in itertools.product(range(4), repeat=len(names)):
        chosen = dict(zip(names, values))
        mat, weight = np.eye(4, dtype=complex), 1.0
        for kind, v in atoms:
            if kind == "m":
                mat = mat @ _slash(vectors[v])
            else:
                mu = chosen[str(v) if v.is_up else str(-v)]
                mat = mat @ _G[mu]
                if not v.is_up:
                    weight *= _MET[mu, mu]
        total += weight * np.trace(mat)
    return total


def test_contraction_identities_against_explicit_matrices():
    """``_contract_pairs`` + ``_slash_trace`` (the exchange route's tracer)
    against literal 4×4 Dirac matrices with random numeric momenta — nested
    and interleaved contracted pairs, including ``γ^aγ^b k̸ γ_a k̸' γ_b``, the
    string SymPy 1.14's ``kahane_simplify`` raises on."""
    rng = np.random.default_rng(11)
    heads = [momentum(f"ci_{n}") for n in range(6)]
    vectors = {h: rng.normal(size=4) for h in heads}

    def dot(a, b):
        return float(vectors[a] @ _MET @ vectors[b])

    from feynlag.pheno.lorentz import index
    r1, r2 = index("ci_r1"), index("ci_r2")
    h = [("m", x) for x in heads]
    strings = [
        (("i", r1), h[0], h[1], ("i", -r1)),
        (("i", r1), h[0], h[1], h[2], ("i", -r1), h[3]),
        (h[0], ("i", r1), h[1], h[2], h[3], h[4], ("i", -r1), h[5]),
        (("i", r1), ("i", r2), h[0], ("i", -r1), h[1], ("i", -r2)),
        (("i", r1), h[0], ("i", r2), h[1], ("i", -r1), h[2], ("i", -r2), h[3]),
        (h[0], ("i", r1), h[1], h[1], ("i", -r1), h[0]),
    ]
    for atoms in strings:
        ours = sum(float(c) * float(_slash_trace(tuple(x for _, x in r), dot, {}))
                   for c, r in _contract_pairs(atoms))
        assert ours == pytest.approx(_explicit_trace(atoms, vectors).real,
                                     rel=1e-9, abs=1e-9), atoms


# --------------------------------------------------------------------------
# same-pairing interference: γ / Z / h in the s channel
# --------------------------------------------------------------------------

def test_gamma_z_higgs_s_channel_interference_matches_spinor_oracle():
    """Massive final-state fermions, a massive Z with a width and independent
    chiral couplings, a photon, and a scalar: every cross term (γZ, γh, Zh —
    the last two vector × scalar, non-zero only through the masses) against
    the explicit-spinor oracle."""
    mf = 1.5
    kin = TwoToTwoKinematics(0, 0, sp.Rational(3, 2), sp.Rational(3, 2))
    l1, l2 = Leg(kin.k1, 0), Leg(kin.k2, 0, anti=True)
    l3, l4 = Leg(kin.k3, sp.Rational(3, 2)), Leg(kin.k4, sp.Rational(3, 2), anti=True)
    e = sp.Rational(3, 10)
    gl, gr = sp.Rational(-2, 10), sp.Rational(15, 100)
    hl, hr = sp.Rational(-25, 100), sp.Rational(17, 100)
    yh, yf = sp.Rational(2, 10), sp.Rational(3, 10)
    q = lambda i: kin.k1(i) + kin.k2(i)   # noqa: E731
    I = sp.I
    diagrams = (
        Diagram(chains=(_chain(l2, l1, "V", -I * e, -I * e, "ia1"),
                        _chain(l3, l4, "V", -I * e, -I * e, "ia2")),
                propagators=(BosonPropagator(1, kin.s, 0),)),
        Diagram(chains=(_chain(l2, l1, "V", I * gl, I * gr, "iz1"),
                        _chain(l3, l4, "V", I * hl, I * hr, "iz2")),
                propagators=(BosonPropagator(1, kin.s, 9, sp.Rational(1, 2),
                                             momentum=q),)),
        Diagram(chains=(_chain(l2, l1, "S", I * yh, I * yh, "ih1"),
                        _chain(l3, l4, "S", -I * yf, -I * yf, "ih2")),
                propagators=(BosonPropagator(0, kin.s, 7),)),
    )
    m2 = Amplitude(diagrams).squared(kin)
    c = complex
    s_line = [(0, 1), (1, 1)]
    oracle = [
        dict(sign=1, chains=[(1, 0, "V", c(-I * e), c(-I * e)),
                             (2, 3, "V", c(-I * e), c(-I * e))],
             spin=1, M=0, W=0, q=s_line),
        dict(sign=1, chains=[(1, 0, "V", c(I * gl), c(I * gr)),
                             (2, 3, "V", c(I * hl), c(I * hr))],
             spin=1, M=9.0, W=0.5, q=s_line),
        dict(sign=1, chains=[(1, 0, "S", c(I * yh), c(I * yh)),
                             (2, 3, "S", c(-I * yf), c(-I * yf))],
             spin=0, M=7.0, W=0, q=s_line),
    ]
    _assert_matches_oracle(m2, kin, [0, 0, mf, mf], [False, True, False, True],
                           oracle)


def test_single_diagram_result_is_unchanged_by_tier_3():
    """A one-diagram amplitude takes exactly the Tier-1/2 path: adding a
    *second* diagram with zero coupling changes nothing."""
    m, e = sp.symbols("m e", positive=True)
    kin = TwoToTwoKinematics(0, 0, m, m)
    l1, l2 = Leg(kin.k1, 0), Leg(kin.k2, 0, anti=True)
    l3, l4 = Leg(kin.k3, m), Leg(kin.k4, m, anti=True)
    photon = Diagram(chains=(_chain(l2, l1, "V", e, e, "u1"),
                             _chain(l3, l4, "V", e, e, "u2")),
                     propagators=(BosonPropagator(1, kin.s, 0),))
    silent = Diagram(chains=(_chain(l2, l1, "V", 0, 0, "u3"),
                             _chain(l3, l4, "V", e, e, "u4")),
                     propagators=(BosonPropagator(1, kin.s, 0),))
    alone = Amplitude((photon,)).squared(kin)
    both = Amplitude((photon, silent)).squared(kin)
    assert sp.simplify(alone - both) == 0


# --------------------------------------------------------------------------
# exchanged-pairing interference: Bhabha and Møller
# --------------------------------------------------------------------------

def _bhabha(m, couplings, mz=None):
    """Hand-built Bhabha ``e⁻(k1)e⁺(k2)→e⁻(k3)e⁺(k4)``: s and t channel, for
    a photon and (optionally) a Z; returns (Σ|M|², kin, oracle diagrams)."""
    kin = TwoToTwoKinematics(m, m, m, m)
    l1, l2 = Leg(kin.k1, m), Leg(kin.k2, m, anti=True)
    l3, l4 = Leg(kin.k3, m), Leg(kin.k4, m, anti=True)
    qs = lambda i: kin.k1(i) + kin.k2(i)   # noqa: E731
    qt = lambda i: kin.k1(i) - kin.k3(i)   # noqa: E731
    width = sp.Rational(1, 2)
    diagrams, oracle = [], []
    for name, (cl, cr), mass in couplings:
        w = width if mass else 0
        diagrams.append(Diagram(
            chains=(_chain(l2, l1, "V", cl, cr, f"b{name}s1"),
                    _chain(l3, l4, "V", cl, cr, f"b{name}s2")),
            propagators=(BosonPropagator(1, kin.s, mass, w,
                                         momentum=qs if mass else None),),
            coefficient=-1))
        diagrams.append(Diagram(
            chains=(_chain(l3, l1, "V", cl, cr, f"b{name}t1"),
                    _chain(l2, l4, "V", cl, cr, f"b{name}t2")),
            propagators=(BosonPropagator(1, kin.t, mass, w,
                                         momentum=qt if mass else None),),
            coefficient=1))
        if sp.sympify(cl).free_symbols or sp.sympify(cr).free_symbols:
            continue                    # symbolic couplings: no numeric oracle
        cc = (complex(cl), complex(cr))
        oracle += [
            dict(sign=-1, chains=[(1, 0, "V", *cc), (2, 3, "V", *cc)], spin=1,
                 M=float(mass), W=float(w), q=[(0, 1), (1, 1)]),
            dict(sign=1, chains=[(2, 0, "V", *cc), (1, 3, "V", *cc)], spin=1,
                 M=float(mass), W=float(w), q=[(0, 1), (2, -1)]),
        ]
    return Amplitude(tuple(diagrams)).squared(kin), kin, oracle


def test_bhabha_qed_closed_form():
    """Massless QED Bhabha: ``Σ|M|² = 8e⁴[(s²+u²)/t² + 2u²/(st) +
    (t²+u²)/s²]`` — the spin-summed form of Peskin & Schroeder Problem 5.2
    [PS95] (four times the averaged ``2e⁴[…]``) — exactly, symbolic ``e``."""
    e = sp.Symbol("e", positive=True)
    m2, kin, _ = _bhabha(0, [("a", (-sp.I * e, -sp.I * e), 0)])
    s, t, u = kin.s, kin.t, kin.u
    closed = 8 * e**4 * ((s**2 + u**2) / t**2 + 2 * u**2 / (s * t)
                         + (t**2 + u**2) / s**2)
    assert sp.simplify(m2 - closed) == 0
    ok, diff = numeric_equal(m2, closed, [s, t, e], sample_range=(-5.0, -1.0),
                             seed=3)
    assert ok, diff


def test_bhabha_relative_sign_is_physical():
    """The s/t relative sign −1 is not a convention: with +1 the result is
    no longer the Bhabha closed form (and changes by the full s–t
    interference ``16e⁴u²/(st)``)."""
    e = sp.Symbol("e", positive=True)
    right, kin, _ = _bhabha(0, [("a", (-sp.I * e, -sp.I * e), 0)])
    s, t, u = kin.s, kin.t, kin.u
    l1, l2 = Leg(kin.k1, 0), Leg(kin.k2, 0, anti=True)
    l3, l4 = Leg(kin.k3, 0), Leg(kin.k4, 0, anti=True)
    c = -sp.I * e
    wrong = Amplitude((
        Diagram(chains=(_chain(l2, l1, "V", c, c, "w1"), _chain(l3, l4, "V", c, c, "w2")),
                propagators=(BosonPropagator(1, kin.s, 0),), coefficient=1),
        Diagram(chains=(_chain(l3, l1, "V", c, c, "w3"), _chain(l2, l4, "V", c, c, "w4")),
                propagators=(BosonPropagator(1, kin.t, 0),), coefficient=1),
    )).squared(kin)
    assert sp.simplify(right - wrong - 32 * e**4 * u**2 / (s * t)) == 0


def test_bhabha_massive_qed_matches_spinor_oracle():
    m2, kin, oracle = _bhabha(sp.Rational(3, 2), [("a", (-0.3j, -0.3j), 0)])
    _assert_matches_oracle(m2, kin, [1.5] * 4, [False, True, False, True], oracle)


def test_bhabha_gamma_z_chiral_massive_matches_spinor_oracle():
    """The hardest case the exchange route meets: massive external fermions,
    a massive Z (so the ``q_aq_b/M²`` numerator puts a second slash of a leg
    momentum into the one trace — the string SymPy mis-traces), chiral
    couplings and γ–Z cross terms in both channels."""
    I = sp.I
    m2, kin, oracle = _bhabha(sp.Rational(3, 2), [
        ("a", (-I * sp.Rational(3, 10), -I * sp.Rational(3, 10)), 0),
        ("z", (I * sp.Rational(-2, 10), I * sp.Rational(15, 100)), sp.Integer(9)),
    ])
    _assert_matches_oracle(m2, kin, [1.5] * 4, [False, True, False, True], oracle)


# --------------------------------------------------------------------------
# ScatteringCalculator end-to-end, couplings extracted from the Lagrangian
# --------------------------------------------------------------------------

# docs/benchmark.md parameter point: MadGraph's (α⁻¹, G_F, M_Z) scheme
_AEWM1, _GF, _MZ, _WZ = 132.50698, 1.16639e-5, 91.1876, 2.4952
_AEW = 1 / _AEWM1
_EE = math.sqrt(4 * math.pi * _AEW)
_MW = math.sqrt(_MZ**2 / 2 + math.sqrt(_MZ**4 / 4 - _AEW * math.pi * _MZ**2
                                       / (_GF * math.sqrt(2))))
_SW2 = 1 - _MW**2 / _MZ**2
_GW, _G1 = _EE / math.sqrt(_SW2), _EE / math.sqrt(1 - _SW2)
_VEV = 2 * _MW * math.sqrt(_SW2) / _EE
_GEV2_TO_PB = 0.3893794e9


@pytest.fixture(scope="module")
def sm_leptons():
    """SM electroweak sector with massless e, μ (plus inert ν_R singlets so
    each neutrino is a declarable DiracParticle)."""
    ew = electroweak_scaffold(gw=_GW, g1=_G1, v=_VEV, mh=125.0)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)

    def doublet(name, comps):
        return WeylFermion(name, reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                           chirality="L", nflavors=1, component_names=comps)

    def singlet(name, comp, y):
        return WeylFermion(name, reps={U1Y: y}, chirality="R", nflavors=1,
                           component_names=[comp])

    Le, eR = doublet("TLe", ["Tnue", "TeL"]), singlet("TeR", "TeR", -1)
    Lmu, muR = doublet("TLmu", ["Tnumu", "TmuL"]), singlet("TmuR", "TmuR", -1)
    nueR = singlet("TnueR", "TnueR", 0)
    L = Lagrangian()
    ew.add_higgs(L)
    for f in (Le, eR, Lmu, muR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("Tier3", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + [Le, eR, Lmu, muR, nueR],
                  parameters=ew.parameters, lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)
    e = DiracParticle("Te", Le.components[1], eR.components[0], 0)
    mu = DiracParticle("Tmu", Lmu.components[1], muR.components[0], 0)
    nu = DiracParticle("Tnu", Le.components[0], nueR.components[0], 0)
    values = {ew.gw.s: _GW, ew.g1.s: _G1}
    return dict(model=model, ph=ph, e=e, mu=mu, nu=nu, values=values)


def _calc(sm, bosons, masses, widths=None, conjugates=None):
    return ScatteringCalculator(
        sm["model"], [sm["e"], sm["mu"], sm["nu"]], masses=masses,
        widths=widths, boson_fields=bosons, conjugate_map=sm["ph"].cmap,
        conjugates=conjugates)


def _helicity_sigma(s, cos_max, photon=True, z=True):
    """Independent closed form for massless ``e⁺e⁻→μ⁺μ⁻`` through γ and/or Z:
    the four helicity amplitudes ``A_ij = e²/s + g_ig_j/(s−M_Z²+iM_ZΓ_Z)``
    with ``dσ/dcosθ = s/(128π)[(|A_LL|²+|A_RR|²)(1+c)² + (|A_LR|²+|A_RL|²)(1−c)²]``
    (the angular structure of [LEPEWWG06] Eq. 1.55), integrated over
    ``|cosθ| < cos_max``."""
    gz = _EE / math.sqrt(_SW2 * (1 - _SW2))
    gl, gr = gz * (-0.5 + _SW2), gz * _SW2
    den = complex(s - _MZ**2, _MZ * _WZ)

    def amp(a, b):
        return abs((_EE**2 / s if photon else 0) + (a * b / den if z else 0))**2

    same, opposite = amp(gl, gl) + amp(gr, gr), amp(gl, gr) + amp(gr, gl)

    def primitive(c):
        return same * (1 + c)**3 / 3 - opposite * (1 - c)**3 / 3

    return s / (128 * math.pi) * (primitive(cos_max) - primitive(-cos_max))


def _mumu_process(sm, bosons):
    Z, A = sm["ph"].Z, sm["ph"].A
    calc = _calc(sm, bosons, {A: 0, Z: _MZ}, {Z: _WZ})
    e, mu = sm["e"], sm["mu"]
    return calc.process((e.particle, e.antiparticle),
                        (mu.particle, mu.antiparticle),
                        numeric=True, extra=sm["values"])


@pytest.fixture(scope="module")
def mumu_gamma_z(sm_leptons):
    """``e⁺e⁻→μ⁺μ⁻`` through γ+Z, ``Σ|M|²`` computed once for the module."""
    proc = _mumu_process(sm_leptons, [sm_leptons["ph"].Z, sm_leptons["ph"].A])
    proc.squared()
    return proc


def test_ee_to_mumu_gamma_z_reproduces_madgraph(mumu_gamma_z):
    """**The Tier-3 acceptance benchmark.**  ``e⁺e⁻→μ⁺μ⁻`` with the γ and Z
    couplings *extracted from the Lagrangian*, both s-channel diagrams found
    by the topology enumerator, at the ``docs/benchmark.md`` point:

    - no cut: 2.8443 pb, against the independent helicity closed form;
    - MadGraph's default lepton acceptance ``|η| < 2.5`` (``|cosθ| <
      tanh 2.5`` for massless leptons): 2.7876 pb, against MadGraph's
      **2.7878 ± 0.0027 pb** — 0.1σ.  The cut is not optional: the no-cut
      number is 2% (20σ) higher, which is how the cut was identified.
    """
    proc = mumu_gamma_z
    assert sorted((d.channel, str(d.mediator)) for d in proc.diagrams) == [
        ("s", "A"), ("s", "Z")]
    s = proc.kinematics.s
    full = float(proc.cross_section().subs(s, 200.0**2)) * _GEV2_TO_PB
    assert full == pytest.approx(_helicity_sigma(200.0**2, 1.0) * _GEV2_TO_PB,
                                 rel=1e-9)
    assert full == pytest.approx(2.8443, abs=5e-4)

    c = math.tanh(2.5)
    cut = float(proc.cross_section(cos_range=(-c, c)).subs(s, 200.0**2)) * _GEV2_TO_PB
    assert cut == pytest.approx(_helicity_sigma(200.0**2, c) * _GEV2_TO_PB, rel=1e-9)
    assert abs(cut - 2.7878) < 0.0027

    # the quadrature route agrees with the symbolic t-integral
    quad = proc.numeric_cross_section(200.0**2, cos_range=(-c, c)) * _GEV2_TO_PB
    assert quad == pytest.approx(cut, rel=1e-8)


def test_ee_to_mumu_interference_small_in_sigma_large_in_afb(sm_leptons, mumu_gamma_z):
    """The γ–Z cross term ``σ(γ+Z) − σ(γ) − σ(Z)`` is only +0.026 pb at
    √s=200 GeV — it goes with the leptons' *vector* couplings, ∝ ``−½+2s_W²
    ≈ 0`` — so Tier 1's photon-only 2.322 pb grows to 2.844 pb mostly through
    ``|M_Z|²``.  In ``A_FB`` the cross term goes with the large *axial*
    couplings instead: 0.565, against ``¾A_eA_μ ≈ 0.016`` for the Z alone on
    the pole.  Each piece is pinned against the helicity closed form."""
    sm = sm_leptons
    s_val = 200.0**2
    sigma = {}
    for label, bosons in (("A", [sm["ph"].A]), ("Z", [sm["ph"].Z])):
        proc = _mumu_process(sm, bosons)
        sigma[label] = (float(proc.cross_section().subs(proc.kinematics.s, s_val))
                        * _GEV2_TO_PB)
    both = mumu_gamma_z
    sigma["AZ"] = float(both.cross_section().subs(both.kinematics.s, s_val)) * _GEV2_TO_PB
    assert sigma["A"] == pytest.approx(
        _helicity_sigma(s_val, 1.0, z=False) * _GEV2_TO_PB, rel=1e-9)
    assert sigma["Z"] == pytest.approx(
        _helicity_sigma(s_val, 1.0, photon=False) * _GEV2_TO_PB, rel=1e-9)
    assert sigma["A"] == pytest.approx(2.3225, abs=5e-4)
    interference = sigma["AZ"] - sigma["A"] - sigma["Z"]
    assert interference == pytest.approx(0.0261, abs=5e-4)
    assert interference / sigma["AZ"] < 0.01
    afb = float(both.forward_backward_asymmetry().subs(both.kinematics.s, s_val))
    assert afb == pytest.approx(0.5653, abs=5e-4)


def test_bhabha_from_the_model_matches_spinor_oracle(sm_leptons):
    """``e⁺e⁻→e⁺e⁻`` from the extracted couplings: the enumerator finds the
    four diagrams (s, t) × (γ, Z) with the s channel carrying the relative
    −1, and the full ``Σ|M|²`` matches the spinor oracle fed the same
    couplings and the textbook Wick signs."""
    sm = sm_leptons
    Z, A = sm["ph"].Z, sm["ph"].A
    calc = _calc(sm, [Z, A], {A: 0, Z: _MZ}, {Z: _WZ})
    e = sm["e"]
    proc = calc.process((e.particle, e.antiparticle),
                        (e.particle, e.antiparticle),
                        numeric=True, extra=sm["values"])
    by_key = {(d.channel, str(d.mediator)): d for d in proc.diagrams}
    assert sorted(by_key) == [("s", "A"), ("s", "Z"), ("t", "A"), ("t", "Z")]
    assert by_key[("s", "A")].sign == -by_key[("t", "A")].sign
    assert proc.symmetry_factor == 1

    oracle = []
    for (channel, _), info in by_key.items():
        va, vb = info.vertices
        chains = [(1, 0) if channel == "s" else (2, 0),
                  (2, 3) if channel == "s" else (1, 3)]
        oracle.append(dict(
            sign=-1 if channel == "s" else 1,
            chains=[(*chains[0], "V", complex(va.g_left), complex(va.g_right)),
                    (*chains[1], "V", complex(vb.g_left), complex(vb.g_right))],
            spin=1, M=float(info.diagram.propagators[0].mass),
            W=float(info.diagram.propagators[0].width),
            q=[(0, 1), (1, 1)] if channel == "s" else [(0, 1), (2, -1)]))
    _assert_matches_oracle(proc.squared(), proc.kinematics, [0.0] * 4,
                           [False, True, False, True], oracle,
                           points=((150.0**2, 0.4), (200.0**2, -0.6)))


def test_moller_qed_closed_form_and_identical_particle_factor(sm_leptons):
    """``e⁻e⁻→e⁻e⁻`` through the photon: t and u channel, relative −1, and
    ``Σ|M|² = 8e⁴[(s²+u²)/t² + 2s²/(tu) + (s²+t²)/u²]`` — the Bhabha form of
    [PS95] Problem 5.2 under the crossing ``s↔u``.  The calculator sets the
    identical-final-state factor ½ itself."""
    sm = sm_leptons
    A = sm["ph"].A
    calc = _calc(sm, [A], {A: 0})
    e = sm["e"]
    proc = calc.process((e.particle, e.particle), (e.particle, e.particle),
                        numeric=True, extra=sm["values"])
    assert sorted(d.channel for d in proc.diagrams) == ["t", "u"]
    assert proc.diagrams[0].sign == -proc.diagrams[1].sign
    assert proc.symmetry_factor == sp.Rational(1, 2)
    kin = proc.kinematics
    s, t, u = kin.s, kin.t, kin.u
    closed = 8 * _EE**4 * ((s**2 + u**2) / t**2 + 2 * s**2 / (t * u)
                           + (s**2 + t**2) / u**2)
    for sv, tv in ((100.0, -30.0), (2500.0, -900.0)):
        point = {s: sv, t: tv}
        assert complex(proc.squared().subs(point)).real == pytest.approx(
            float(closed.subs(point)), rel=1e-10)


def test_nu_e_scattering_w_z_relative_sign_from_fierz(sm_leptons):
    """``ν_e e⁻→ν_e e⁻``: a t-channel Z and a u-channel W, joined through
    the explicit ``{W⁺: W⁻}`` antiparticle map.  Deep below both masses the
    W exchange Fierz-rearranges (``(ν̄γ^μP_Le)(ēγ_μP_Lν) =
    (ν̄γ^μP_Lν)(ēγ_μP_Le)`` for anticommuting fields) into a shift of the
    electron's left-handed Z coupling, ``g_L → g_L + g_W²M_Z²/(g_L^ν M_W²)``
    — in the SM ``(g/c_W)(−½+s_W²) → (g/c_W)(+½+s_W²)``, the familiar
    "``g_L+1``" of ν_e e scattering.  This pins the W–Z relative sign the
    topology's permutation parity assigns: the opposite sign would give
    ``−3/2+s_W²`` instead."""
    sm = sm_leptons
    ph = sm["ph"]
    heavy = 1e5                       # √s ≪ M: the contact (Fermi) limit
    scale_w = heavy * _MW / _MZ
    calc = _calc(sm, [ph.Z, ph.Wp, ph.Wm],
                 {ph.Z: heavy, ph.Wp: scale_w, ph.Wm: scale_w},
                 conjugates={ph.Wp: ph.Wm, ph.Wm: ph.Wp})
    e, nu = sm["e"], sm["nu"]
    proc = calc.process((e.particle, nu.particle), (e.particle, nu.particle),
                        numeric=True, extra=sm["values"])
    channels = sorted((d.channel, str(d.mediator)) for d in proc.diagrams)
    assert channels == [("t", "Z"), ("u", "Wm")] or channels == [("t", "Z"), ("u", "Wp")]
    zinfo = next(d for d in proc.diagrams if d.channel == "t")
    winfo = next(d for d in proc.diagrams if d.channel == "u")
    vz_e, vz_nu = zinfo.vertices
    gw_a, gw_b = winfo.vertices
    gz_nu = complex(vz_nu.g_left) / 1j
    shift = (complex(gw_a.g_left) * complex(gw_b.g_left) / (1j * 1j)
             * heavy**2 / (gz_nu * scale_w**2))

    kin = TwoToTwoKinematics(0, 0, 0, 0)
    l1, l2 = Leg(kin.k1, 0), Leg(kin.k2, 0)
    l3, l4 = Leg(kin.k3, 0), Leg(kin.k4, 0)
    gz_el, gz_er = complex(vz_e.g_left) / 1j, complex(vz_e.g_right) / 1j
    effective = Amplitude((Diagram(
        chains=(_chain(l3, l1, "V", 1j * (gz_el + shift), 1j * gz_er, "nf1"),
                _chain(l4, l2, "V", 1j * gz_nu, 0, "nf2")),
        propagators=(BosonPropagator(1, kin.t, heavy,
                                     momentum=lambda i: kin.k1(i) - kin.k3(i)),)),
    )).squared(kin)
    sw2 = _SW2
    assert ((gz_el + shift) / gz_er).real == pytest.approx((0.5 + sw2) / sw2, rel=1e-9)
    assert (gz_el / gz_er).real == pytest.approx((-0.5 + sw2) / sw2, rel=1e-9)
    for sv, cos in ((50.0, 0.2), (400.0, -0.5)):
        full = _engine_at(proc.squared(), proc.kinematics, sv, cos).real
        eff = _engine_at(effective, kin, sv, cos).real
        assert full == pytest.approx(eff, rel=1e-5)


def test_calculator_guards(sm_leptons):
    sm = sm_leptons
    Z, A = sm["ph"].Z, sm["ph"].A
    e, mu = sm["e"], sm["mu"]
    calc = _calc(sm, [Z, A], {A: 0})                 # no Z mass declared
    with pytest.raises(KeyError, match="mediator"):
        calc.process((e.particle, e.antiparticle), (mu.particle, mu.antiparticle))
    calc = _calc(sm, [Z, A], {A: 0, Z: _MZ})
    with pytest.raises(ValueError, match="bar"):     # fermion number violated
        calc.process((e.particle, e.particle), (mu.particle, mu.antiparticle))
    with pytest.raises(KeyError, match="DiracParticle"):
        calc.process((e.particle, sp.Symbol("W")), (e.particle, sp.Symbol("W")))
    quark = DiracParticle("Tq", sm["e"].left, sm["e"].right, 0, color=3)
    qcalc = ScatteringCalculator(sm["model"], [quark], masses={A: 0},
                                 boson_fields=[A])
    with pytest.raises(NotImplementedError, match="Tier 5"):
        qcalc.process((quark.particle, quark.antiparticle),
                      (quark.particle, quark.antiparticle))


def test_mismatched_diagrams_raise():
    m = sp.Integer(0)
    kin = TwoToTwoKinematics(m, m, m, m)
    l1, l2 = Leg(kin.k1, 0), Leg(kin.k2, 0, anti=True)
    l3, l4 = Leg(kin.k3, 0), Leg(kin.k4, 0, anti=True)
    good = Diagram(chains=(_chain(l2, l1, "V", 1, 1, "mm1"),
                           _chain(l3, l4, "V", 1, 1, "mm2")),
                   propagators=(BosonPropagator(1, kin.s, 0),))
    # bar slot k1 instead of k2: not the same set of external slots
    bad = Diagram(chains=(_chain(l1, l2, "V", 1, 1, "mm3"),
                          _chain(l3, l4, "V", 1, 1, "mm4")),
                  propagators=(BosonPropagator(1, kin.s, 0),))
    with pytest.raises(NotImplementedError, match="same external legs"):
        Amplitude((good, bad)).squared(kin)

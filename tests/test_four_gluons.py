"""Tier 5, stage 5c of the 2→2 scattering roadmap: ``gg→gg``.

The first amplitude with **no fermion line** (:mod:`feynlag.pheno.boson`) and
the first diagram with a four-point contact vertex.  Three colour structures
``f^{ije}f^{ekl}``, each an exchange diagram through two three-gluon vertices
plus its share of the four-gluon vertex; all four gluons summed over physical
polarizations.

What is pinned, and against what:

- the trace-free contraction against explicit numpy tensors;
- the contact vertex against a direct fourth derivative of ``−¼F²``;
- the colour matrix against ``N²(N²−1)`` and the Jacobi identity;
- ``gg→gg`` of Ellis, Stirling and Webber, *QCD and Collider Physics*,
  Table 7.1 — exactly and symbolically;
- SU(3) and SU(2) against an explicit evaluator written from the textbook
  Feynman rules (Peskin & Schroeder, Chapter 16): explicit transverse
  polarization
  vectors and ``f^{abc}``, the amplitude summed over diagrams per colour and
  helicity configuration — an evaluator that first checks its own Ward
  identity;
- independence of the reference vectors;
- the sign that this process fixes: the **contact** term relative to the
  exchange diagrams.  Flipping it, or dropping the contact term, breaks the
  Ward identity; flipping the cubic coupling's sign changes nothing (two
  cubic vertices per diagram), which is pinned too.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import sympy as sp

from feynlag import (
    ExternalParameter, Lagrangian, Model, SU2, SU3, U1, WeylFermion,
    fermion_gauge_current,
)
from feynlag.export.ufo.vvvv import adjoint_vvvv
from feynlag.pheno import (
    BosonAmplitude, BosonTerm, DiracParticle, ScatteringCalculator,
    boson_amplitude, colour_matrix, contract_tensors,
    enumerate_boson_diagrams, four_boson_colour_tensor, vvvv_tensor,
)
from feynlag.pheno import boson_vertices

_MET = np.diag([1.0, -1.0, -1.0, -1.0])
_GS = 1.2
_POINTS = ((60.0**2, 0.3), (150.0**2, -0.65), (25.0**2, 0.8))


def _yang_mills(group_cls, tag):
    """Pure Yang–Mills: a gauge group, its bosons, nothing else."""
    g = ExternalParameter(f"g{tag}", _GS, positive=True)
    group = group_cls(f"G{tag}", coupling=g)
    bosons = group.bosons(f"V{tag}")
    model = Model(f"YM{tag}", gauge_groups=[group], fields=[bosons],
                  parameters=[g], lagrangian=Lagrangian())
    calc = ScatteringCalculator(model, [], masses={}, colour_group=group,
                                gluon=bosons)
    return dict(group=group, g=g.s, calc=calc)


@pytest.fixture(scope="module")
def su3():
    return _yang_mills(SU3, "c7")


@pytest.fixture(scope="module")
def gggg(su3):
    """Symbolic ``gg→gg``, ``Σ|M|²`` computed once."""
    gluon = su3["calc"].gluon
    proc = su3["calc"].process((gluon, gluon), (gluon, gluon))
    proc.squared()
    return proc


def _closed_form(kin, n):
    """``Σ|M|²/g⁴`` summed over all spins and colours for SU(N):
    ``16 N²(N²−1)(3 − tu/s² − su/t² − st/u²)``; for ``N = 3`` this is
    ``256 ×`` the ESW entry."""
    s, t, u = kin.s, kin.t, kin.u
    return 16 * n**2 * (n**2 - 1) * (
        3 - t * u / s**2 - s * u / t**2 - s * t / u**2)


# --------------------------------------------------------------------------
# ingredients
# --------------------------------------------------------------------------

def test_contract_tensors_matches_explicit_contraction():
    vectors = {"p": np.array([3.0, 1.0, -2.0, 0.5]),
               "q": np.array([1.5, -0.5, 0.25, 2.0])}

    def dot(a, b):
        return sp.nsimplify(vectors[a] @ _MET @ vectors[b])

    p, q = ((sp.S.One, "p"),), ((sp.S.One, "q"),)
    both = ((2, "p"), (-3, "q"))
    assert contract_tensors((("g", "a", "a"),), dot) == 4
    assert contract_tensors((("g", "a", "b"), ("g", "b", "a")), dot) == 4
    assert contract_tensors(
        (("g", "a", "b"), ("g", "a", "c"), ("v", p, "b"), ("v", q, "c")),
        dot) == dot("p", "q")
    explicit = (2 * vectors["p"] - 3 * vectors["q"]) @ _MET @ vectors["q"]
    assert float(contract_tensors((("v", both, "a"), ("v", q, "a")), dot)) \
        == pytest.approx(explicit)
    with pytest.raises(ValueError, match="exactly twice"):
        contract_tensors((("v", p, "a"),), dot)
    with pytest.raises(ValueError, match="unknown tensor"):
        contract_tensors((("x", "a", "a"),), dot)


def test_contact_vertex_from_the_lagrangian():
    """``i ∂⁴L/∂A⁴`` of the quartic term of ``−¼F²``,
    ``L₄ = −¼ g² f^{abe}f^{cde} A^a_μ A^b_ν A^{cμ} A^{dν}``, equals
    ``Σ_n (i g²) · f f_n · VVVV_n`` with the structures of
    :func:`vvvv_tensor` and the colour tensors of
    :func:`four_boson_colour_tensor` (SU(2), component by component)."""
    g = sp.Symbol("g", positive=True)
    group = SU2("SU2contact", coupling=g)
    f = group.structure_constants()
    A = [[sp.Symbol(f"A_{a}_{m}") for m in range(4)] for a in range(3)]
    metric = [1, -1, -1, -1]
    L4 = sp.S.Zero
    for (a, b, e), f1 in f.items():
        for (c, d, e2), f2 in f.items():
            if e2 != e:
                continue
            L4 += -sp.Rational(1, 4) * g**2 * f1 * f2 * sum(
                metric[m] * metric[n] * A[a][m] * A[b][n] * A[c][m] * A[d][n]
                for m in range(4) for n in range(4))
    L4 = sp.expand(L4)

    structures = adjoint_vvvv(group)
    assert set(structures.values()) == {sp.I * g**2}
    names = ("x1", "x2", "x3", "x4")
    pairings = {"VVVV1": ((0, 1), (2, 3)), "VVVV2": ((0, 2), (1, 3)),
                "VVVV3": ((0, 3), (1, 2))}
    colour = {name: four_boson_colour_tensor(pairing, 3, f)
              for name, pairing in pairings.items()}

    def engine(colours, lorentz):
        total = sp.S.Zero
        for name, coupling in structures.items():
            k = colour[name].get(colours, 0)
            if k == 0:
                continue
            for coeff, tensors in vvvv_tensor(name, coupling, names):
                value = coeff
                for _, x, y in tensors:
                    m, n = lorentz[names.index(x)], lorentz[names.index(y)]
                    value *= metric[m] if m == n else 0   # g^{μν}
                total += k * value
        return total

    # every colour assignment, with the Lorentz indices in each of the three
    # pairings (anything else vanishes on both sides)
    checked = 0
    for colours in itertools.product(range(3), repeat=4):
        for m, n in ((0, 1), (2, 2)):
            for lorentz in ((m, m, n, n), (m, n, m, n), (m, n, n, m)):
                fields = [A[c][mu] for c, mu in zip(colours, lorentz)]
                # ∂/∂A_μ carries an upper index: raise it with the metric
                direct = sp.I * sp.diff(L4, *fields) * math.prod(
                    metric[mu] for mu in lorentz)
                assert sp.expand(direct - engine(colours, lorentz)) == 0
                checked += direct != 0
    assert checked > 50                     # the sample is not all zeros
    assert sp.diff(L4, A[0][0], A[1][1], A[0][2], A[1][3]) == 0


@pytest.mark.parametrize("group_cls, n", [(SU2, 2), (SU3, 3)])
def test_four_gluon_colour_matrix(group_cls, n):
    """``Σ(f f)² = N²(N²−1)`` on the diagonal, half of it off the diagonal,
    and the Jacobi identity ``K_{(01|23)} − K_{(02|13)} + K_{(03|12)} = 0``."""
    group = group_cls(f"SU{n}colourmatrix")
    f, dim = group.structure_constants(), n**2 - 1
    tensors = [four_boson_colour_tensor(pairing, dim, f) for pairing in (
        ((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))]
    C = colour_matrix(tensors)
    full = n**2 * (n**2 - 1)
    assert [C[i][i] for i in range(3)] == [full] * 3
    assert (C[0][1], C[0][2], C[1][2]) == (full // 2, -full // 2, full // 2)
    for row in C:
        assert row[0] - row[1] + row[2] == 0
    keys = set().union(*tensors)
    assert all(tensors[0].get(k, 0) - tensors[1].get(k, 0)
               + tensors[2].get(k, 0) == 0 for k in keys)


def test_calculator_supplies_the_stripped_self_couplings(su3):
    calc, g = su3["calc"], su3["g"]
    gluon = calc.gluon
    (ggg,) = [v for v in calc.vertices() if v.vertex_type == "VVV"]
    (contact,) = [v for v in calc.vertices() if v.vertex_type == "VVVV"]
    assert ggg.coupling == -g
    assert contact.particles == (gluon,) * 4
    assert contact.meta["structures"] == {
        name: sp.I * g**2 for name in ("VVVV1", "VVVV2", "VVVV3")}


# --------------------------------------------------------------------------
# the parton table
# --------------------------------------------------------------------------

def test_parton_table_gg_to_gg(su3, gggg):
    """Ellis–Stirling–Webber Table 7.1:
    ``Σ̄|M|²/g⁴ = 9/2 (3 − tu/s² − su/t² − st/u²)``, averaged over the
    ``(2 × 8)²`` initial states."""
    kin = gggg.kinematics
    s, t, u = kin.s, kin.t, kin.u
    assert [(d.channel, d.structure) for d in gggg.diagrams] == [
        ("s", "VVVV1"), ("t", "VVVV2"), ("u", "VVVV3")]
    assert [len(d.terms) for d in gggg.diagrams] == [11, 11, 11]
    assert [st.dof() for st in gggg.initial_states] == [16, 16]
    assert gggg.symmetry_factor == sp.Rational(1, 2)
    assert gggg.colour == [[72, 36, -36], [36, 72, 36], [-36, 36, 72]]
    table = sp.Rational(9, 2) * (
        3 - t * u / s**2 - s * u / t**2 - s * t / u**2)
    assert sp.simplify(gggg.squared() / 256 / su3["g"]**4 - table) == 0
    assert sp.simplify(256 * table - _closed_form(kin, 3)) == 0


def test_quarks_in_the_model_do_not_change_gg_to_gg():
    """The same process from a QCD × QED model with a declared quark."""
    gs = ExternalParameter("gs7", _GS, positive=True)
    ee = ExternalParameter("ee7", 0.3, positive=True)
    SU3c, U1em = SU3("SU3c7q", coupling=gs), U1("U1em7q", coupling=ee)
    uL, uR = [WeylFermion(f"hu{chi}", reps={SU3c: 3, U1em: sp.Rational(2, 3)},
                          chirality=chi, nflavors=1,
                          component_names=[f"hu{chi}_{k}" for k in (1, 2, 3)])
              for chi in ("L", "R")]
    G, A = SU3c.bosons("H"), U1em.bosons("B")
    i = sp.Symbol("i", integer=True)
    L = Lagrangian()
    for f in (uL, uR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("QCDfour", gauge_groups=[SU3c, U1em],
                  fields=[uL, uR, G, A], parameters=[gs, ee], lagrangian=L)
    photon = A.components[0]
    u = DiracParticle("hu", uL.components[0], uR.components[0], 0, color=3)
    calc = ScatteringCalculator(model, [u], masses={photon: 0},
                                boson_fields=[photon], colour_group=SU3c,
                                gluon=G)
    g = calc.gluon
    proc = calc.process((g, g), (g, g), numeric=True,
                        extra={gs.s: _GS, ee.s: 0.3})
    assert len(proc.diagrams) == 3
    for s, cos in _POINTS:
        engine = _engine_at(proc, s, cos)
        expected = _GS**4 * _closed_at(proc.kinematics, 3, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(expected, rel=1e-9)
    # the only four-boson process: anything else is refused, not guessed
    for bad in (((photon, photon), (photon, photon)),
                ((g, g), (g, photon)), ((g, photon), (g, photon))):
        with pytest.raises(NotImplementedError, match="four external gluons"):
            calc.process(*bad)


# --------------------------------------------------------------------------
# explicit polarization vectors and f^{abc}
# --------------------------------------------------------------------------

def _dot(a, b):
    return a @ _MET @ b


def _transverse(k):
    n = k[1:] / np.linalg.norm(k[1:])
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return [np.concatenate(([0.0], e1)), np.concatenate(([0.0], e2))]


def _three_gluon(pa, pb, pc, ea, eb):
    """``W^ρ`` such that the three-gluon vertex with all momenta incoming,
    ``g f^{abc}[g^{μν}(p_a−p_b)^ρ + g^{νρ}(p_b−p_c)^μ + g^{ρμ}(p_c−p_a)^ν]``,
    contracted with ``ε_a``, ``ε_b`` is ``g f^{abc} W^ρ``."""
    return (_dot(ea, eb) * (pa - pb) + eb * _dot(pb - pc, ea)
            + ea * _dot(pc - pa, eb))


def _amplitude(f, g, momenta, eps):
    """The colour-dressed ``gg→gg`` amplitude ``[a, b, c, d]`` for four
    *incoming* momenta and polarization vectors, from the textbook rules:
    the vertex above, the propagator ``−i g_{μν}δ^{ab}/q²`` and the contact
    vertex ``−i g²[f^{abe}f^{cde}(g^{μρ}g^{νσ} − g^{μσ}g^{νρ}) + …]``."""
    amp = np.zeros((f.shape[0],) * 4, dtype=complex)
    labels = "abcd"
    for (i, j), (k, l) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        q = momenta[i] + momenta[j]
        w1 = _three_gluon(momenta[i], momenta[j], -q, eps[i], eps[j])
        w2 = _three_gluon(momenta[k], momenta[l], q, eps[k], eps[l])
        # vertex 1 is f^{ij e}; vertex 2, legs (k, l, e), is f^{kl e}
        lorentz = g**2 * _dot(w1, w2) * (-1j / _dot(q, q))
        contact = -1j * g**2 * (
            _dot(eps[i], eps[k]) * _dot(eps[j], eps[l])
            - _dot(eps[i], eps[l]) * _dot(eps[j], eps[k]))
        colour = np.einsum(f"{labels[i]}{labels[j]}e,{labels[k]}{labels[l]}e->abcd",
                           f, f)
        amp += (lorentz + contact) * colour
    return amp


def _oracle(group, g, s, cos):
    """``Σ|M|²`` for ``g(k1) g(k2) → g(k3) g(k4)``, and the largest Ward
    violation relative to the amplitude (must be round-off)."""
    n = group.n_generators
    f = np.zeros((n, n, n))
    for (a, b, c), value in group.structure_constants().items():
        f[a, b, c] = float(value)
    rs = math.sqrt(s)
    sin = math.sqrt(1 - cos**2)
    k1, k2 = rs / 2 * np.array([1.0, 0, 0, 1.0]), rs / 2 * np.array([1.0, 0, 0, -1.0])
    k3 = rs / 2 * np.array([1.0, sin, 0, cos])
    k4 = rs / 2 * np.array([1.0, -sin, 0, -cos])
    momenta = (k1, k2, -k3, -k4)
    total, ward, scale = 0.0, 0.0, 0.0
    for eps in itertools.product(*(_transverse(k) for k in (k1, k2, k3, k4))):
        amp = _amplitude(f, g, momenta, eps)
        total += float(np.sum(np.abs(amp)**2))
        scale = max(scale, float(np.max(np.abs(amp))))
        for leg in range(4):
            probe = list(eps)
            probe[leg] = momenta[leg] / rs
            ward = max(ward, float(np.max(np.abs(
                _amplitude(f, g, momenta, probe)))))
    return total, ward / scale


def _engine_at(proc, s, cos):
    kin = proc.kinematics
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return complex(proc.squared().subs({kin.s: s, kin.t: t}))


def _closed_at(kin, n, s, cos):
    t = kin.t_of_cos(cos).subs(kin.s, s)
    return float(_closed_form(kin, n).subs({kin.s: s, kin.t: t}))


def test_explicit_evaluator_agrees_with_the_engine_su3(su3, gggg):
    for s, cos in _POINTS:
        oracle, ward = _oracle(su3["group"], _GS, s, cos)
        assert ward < 1e-9                  # the evaluator is gauge invariant
        kin = gggg.kinematics
        value = complex(gggg.squared().subs({
            su3["g"]: _GS, kin.s: s,
            kin.t: kin.t_of_cos(cos).subs(kin.s, s)}))
        assert abs(value.imag) < 1e-9 * abs(value.real)
        assert value.real == pytest.approx(oracle, rel=1e-9)


def test_su2_yang_mills_vs_explicit_evaluator():
    """Nothing is SU(3)-specific: pure SU(2) Yang–Mills against the same
    evaluator, and the ``N²(N²−1)`` scaling of the closed form."""
    ym = _yang_mills(SU2, "w7")
    boson = ym["calc"].gluon
    proc = ym["calc"].process((boson, boson), (boson, boson), numeric=True,
                              extra={ym["g"]: _GS})
    assert proc.colour == [[12, 6, -6], [6, 12, 6], [-6, 6, 12]]
    assert [st.dof() for st in proc.initial_states] == [6, 6]
    for s, cos in _POINTS:
        oracle, ward = _oracle(ym["group"], _GS, s, cos)
        assert ward < 1e-9
        engine = _engine_at(proc, s, cos)
        assert abs(engine.imag) < 1e-9 * abs(engine.real)
        assert engine.real == pytest.approx(oracle, rel=1e-9)
        assert engine.real == pytest.approx(
            _GS**4 * _closed_at(proc.kinematics, 2, s, cos), rel=1e-9)


# --------------------------------------------------------------------------
# gauge independence and the sign controls
# --------------------------------------------------------------------------

def test_result_is_independent_of_the_reference_vectors(gggg):
    """The reference vectors are a gauge choice: pairing each incoming gluon
    with an outgoing one instead gives the same ``Σ|M|²``."""
    kin = gggg.kinematics
    other = boson_amplitude(gggg.externals, gggg.diagrams, kin, references={
        0: kin.k3, 2: kin.k1, 1: kin.k4, 3: kin.k2})
    assert sp.simplify(other.squared(kin, ward_check=False,
                                     colour=gggg.colour)
                       - gggg.squared()) == 0


def test_minus_g_on_the_gluons_is_refused_and_wrong(gggg):
    kin = gggg.kinematics
    covariant = boson_amplitude(gggg.externals, gggg.diagrams, kin)
    with pytest.raises(NotImplementedError, match="Ward identity"):
        covariant.squared(kin, colour=gggg.colour)
    forced = covariant.squared(kin, ward_check=False, colour=gggg.colour)
    assert sp.simplify(forced - gggg.squared()) != 0


def _with_contact(su3, gggg, factor):
    """The amplitude rebuilt with the contact couplings scaled by ``factor``."""
    calc = su3["calc"]
    vertices = [
        replace(v, meta=dict(v.meta, structures={
            name: factor * c for name, c in v.meta["structures"].items()}))
        if v.vertex_type == "VVVV" else v
        for v in calc.vertices()]
    kin = gggg.kinematics
    diagrams = enumerate_boson_diagrams(gggg.externals, vertices, kin)
    return boson_amplitude(gggg.externals, diagrams, kin,
                           references=calc._references(gggg.externals, kin))


@pytest.mark.parametrize("factor", [-1, 0])
def test_wrong_contact_term_breaks_the_ward_identity(su3, gggg, factor):
    """The contact vertex with the opposite sign, or left out: the exchange
    diagrams alone are not gauge invariant."""
    amplitude = _with_contact(su3, gggg, factor)
    assert [len(d) for d in amplitude.diagrams] == (
        [11] * 3 if factor else [9] * 3)
    with pytest.raises(NotImplementedError, match="Ward identity"):
        amplitude.squared(gggg.kinematics, colour=gggg.colour)


def test_cubic_sign_is_invisible_here(su3, gggg, monkeypatch):
    """Two cubic vertices per exchange diagram: ``gg→gg`` cannot see
    ``VVV_KAPPA`` (``q q̄→gg`` and ``e⁺e⁻→W⁺W⁻`` are what fix it)."""
    monkeypatch.setattr(boson_vertices, "VVV_KAPPA", -boson_vertices.VVV_KAPPA)
    gluon = su3["calc"].gluon
    flipped = su3["calc"].process((gluon, gluon), (gluon, gluon))
    assert sp.simplify(flipped.squared() - gggg.squared()) == 0


# --------------------------------------------------------------------------
# guards and the cross section
# --------------------------------------------------------------------------

def test_enumeration_guards(su3, gggg):
    kin = gggg.kinematics
    cubic_only = [v for v in su3["calc"].vertices() if v.vertex_type != "VVVV"]
    with pytest.raises(ValueError, match="only gauge invariant together"):
        enumerate_boson_diagrams(gggg.externals, cubic_only, kin)
    colourless = [replace(ext, color=1) for ext in gggg.externals]
    with pytest.raises(NotImplementedError, match="four external gluons"):
        enumerate_boson_diagrams(colourless, su3["calc"].vertices(), kin)
    with pytest.raises(ValueError, match="one row per diagram"):
        BosonAmplitude((), ()).squared(kin, colour=[[1]])
    # a term with no tensors is a scalar amplitude
    assert BosonAmplitude(((BosonTerm(sp.Integer(3)),),)).squared(kin) == 9


def test_cross_section_inside_an_angular_cut(su3):
    """``dσ/dt = Σ̄|M|²/(16π s²)`` with the identical-particle ``½``; the
    t- and u-channel poles need a cut."""
    gluon = su3["calc"].gluon
    proc = su3["calc"].process((gluon, gluon), (gluon, gluon), numeric=True,
                               extra={su3["g"]: _GS})
    kin = proc.kinematics
    s_value, cut = 100.0**2, 0.8
    sigma = proc.numeric_cross_section(s_value, cos_range=(-cut, cut))
    t = kin.t
    density = (_GS**4 * _closed_form(kin, 3) / 256
               / (16 * sp.pi * kin.s**2) / 2).subs(kin.s, s_value)
    expected = sp.Integral(density, (t, -s_value * (1 + cut) / 2,
                                     -s_value * (1 - cut) / 2)).evalf()
    assert sigma == pytest.approx(float(expected), rel=1e-6)

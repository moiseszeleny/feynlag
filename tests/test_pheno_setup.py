"""Model-derived calculator setup: ``from_model``, ``ParameterPoint`` and the
``(α, G_F, M_Z)`` input scheme.

What is pinned:

- the boson masses read off the physical Lagrangian are the textbook tree
  values (``m_Z² = (g²+g'²)v²/4``, ``m_W² = g²v²/4``, ``m_A = 0``,
  ``m_h² = 2λv²``), and a field that is not a mass eigenstate gets **no**
  mass rather than a wrong one;
- ``ScatteringCalculator.from_model`` reproduces the explicitly configured
  calculator: the same vertices, the same ``Σ|M|²`` for ``e⁺e⁻→μ⁺μ⁻`` (γ+Z)
  and ``σ(e⁺e⁻→W⁺W⁻) = 19.447 pb`` at √s = 200 GeV (``test_ww.py``; MadGraph
  19.498 ± 0.058 pb), with *derived* boson masses;
- ``DecayCalculator.from_model`` gives ``Γ(Z→νν̄) = m_Z(g²+g'²)/96π`` with
  ``m_Z`` derived, not typed;
- ``ParameterPoint`` re-derives internals under an override, never mutates
  the parameters, and refuses overrides of internal or unknown parameters;
- ``electroweak_inputs`` reproduces the hand-written input block of the
  examples and makes the derived masses equal the inputs.
"""

import math

import pytest
import sympy as sp

from feynlag import (
    Lagrangian, Model, ParameterPoint, ParameterSet, Rotation, WeylFermion,
    broken_nonabelian_groups, electroweak_inputs, electroweak_scaffold,
    fermion_gauge_current, physical_boson_masses, rotation_2x2,
    standard_model, to_physical_basis,
)
from feynlag.pheno import DecayCalculator, DiracParticle, ScatteringCalculator
from feynlag.verify import numeric_equal

_INPUTS = electroweak_inputs()          # docs/benchmark.md point
_WZ, _MH = 2.4952, 125.0
_GEV2_TO_PB = 0.3893794e9


def _two_lepton_model():
    """SM electroweak sector at the benchmark point with massless e and μ."""
    ew = electroweak_scaffold(**_INPUTS.couplings, mh=_MH)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)
    fields, leptons = [], {}
    for name in ("e", "mu"):
        Ll = WeylFermion(f"PL{name}", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                         chirality="L", nflavors=1,
                         component_names=[f"Pnu{name}", f"P{name}L"])
        lR = WeylFermion(f"P{name}R", reps={U1Y: -1}, chirality="R",
                         nflavors=1, component_names=[f"P{name}R"])
        fields += [Ll, lR]
        leptons[name] = (
            DiracParticle(f"P{name}", Ll.components[1], lR.components[0], 0),
            DiracParticle(f"Pnu_{name}", Ll.components[0], None, 0))
    L = Lagrangian()
    ew.add_higgs(L)
    for f in fields:
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("PhenoSetup", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + fields, parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)
    return dict(model=model, ew=ew, ph=ph, leptons=leptons)


@pytest.fixture(scope="module")
def sm2():
    return _two_lepton_model()


# --------------------------------------------------------------------------
# derived boson masses
# --------------------------------------------------------------------------

def test_physical_boson_masses_are_the_tree_values():
    sm = standard_model()
    sc, ph = sm.scaffold, sm.physical
    g, gp, v, lam = sc.gw.s, sc.g1.s, sc.v.s, sc.lam.s
    m2 = physical_boson_masses(sm.model)
    expected = {ph.Z: (g**2 + gp**2) * v**2 / 4, ph.A: 0,
                ph.Wp: g**2 * v**2 / 4, ph.Wm: g**2 * v**2 / 4,
                ph.h: 2 * lam * v**2, ph.G0: 0, ph.Gp: 0, ph.Gm: 0}
    assert set(m2) == set(expected)
    for boson, value in expected.items():
        assert sp.simplify(m2[boson] - value) == 0, boson
        assert numeric_equal(m2[boson], value, [g, gp, v, lam]), boson
    # the registry itself is not modified by the derivation
    assert sm.model.physical.masses == {}


def test_non_eigenstate_gets_no_mass():
    """A rotation by an arbitrary angle leaves an X·Y mass term: neither
    state is a mass eigenstate, so neither gets a (wrong) mass."""
    ew = electroweak_scaffold()
    L = Lagrangian()
    ew.add_higgs(L)
    model = Model("NotEigen", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ew.parameters, lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    W3, B = ew.W.components[2], ew.B.components[0]
    X, Y = sp.symbols("X Y", real=True)
    model.rotate(Rotation([W3, B], [X, Y],
                          rotation_2x2(sp.Symbol("theta", real=True))))
    model.physical.add_bosons([X, Y])
    assert physical_boson_masses(model) == {}


def test_unitary_bosons_and_broken_groups(sm2):
    ph, model = sm2["ph"], sm2["model"]
    assert set(model.physical.unitary_bosons) == {ph.h, ph.Z, ph.A, ph.Wp,
                                                  ph.Wm}
    assert not set(model.physical.unitary_bosons) & {ph.G0, ph.Gp, ph.Gm}
    assert [g.name for g in broken_nonabelian_groups(model)] == ["SU2L"]


# --------------------------------------------------------------------------
# electroweak input scheme
# --------------------------------------------------------------------------

def test_electroweak_inputs_reproduce_the_hand_written_block():
    aew, gf, mz = 1 / 132.50698, 1.16639e-5, 91.1876
    ee = math.sqrt(4 * math.pi * aew)
    mw = math.sqrt(mz**2 / 2 + math.sqrt(mz**4 / 4 - aew * math.pi * mz**2
                                         / (gf * math.sqrt(2))))
    sw2 = 1 - mw**2 / mz**2
    assert _INPUTS.MW == pytest.approx(mw, rel=1e-14)
    assert _INPUTS.sw2 == pytest.approx(sw2, rel=1e-14)
    assert _INPUTS.gw == pytest.approx(ee / math.sqrt(sw2), rel=1e-14)
    assert _INPUTS.g1 == pytest.approx(ee / math.sqrt(1 - sw2), rel=1e-14)
    assert _INPUTS.v == pytest.approx(2 * mw * math.sqrt(sw2) / ee, rel=1e-14)
    # tree-level consistency: v² = 1/(√2 G_F), M_Z = √(g²+g'²) v/2
    assert _INPUTS.v**2 == pytest.approx(1 / (math.sqrt(2) * gf), rel=1e-12)
    assert math.hypot(_INPUTS.gw, _INPUTS.g1) * _INPUTS.v / 2 == \
        pytest.approx(mz, rel=1e-12)


def test_electroweak_inputs_rejects_impossible_point():
    with pytest.raises(ValueError, match="no real M_W"):
        electroweak_inputs(alpha_inv=1.0)


def test_derived_masses_equal_the_inputs(sm2):
    calc = ScatteringCalculator.from_model(sm2["model"], [])
    ph, values = sm2["ph"], calc.values()

    def mass(b):
        return float(calc.mass_of(b).subs(values))

    assert mass(ph.Z) == pytest.approx(_INPUTS.MZ, rel=1e-12)
    assert mass(ph.Wp) == pytest.approx(_INPUTS.MW, rel=1e-12)
    assert mass(ph.Wm) == pytest.approx(_INPUTS.MW, rel=1e-12)
    assert mass(ph.h) == pytest.approx(_MH, rel=1e-12)
    assert mass(ph.A) == 0


# --------------------------------------------------------------------------
# ScatteringCalculator.from_model
# --------------------------------------------------------------------------

def _explicit(sm2, particles):
    ph = sm2["ph"]
    return ScatteringCalculator(
        sm2["model"], particles,
        masses={ph.A: 0, ph.Z: _INPUTS.MZ, ph.Wp: _INPUTS.MW, ph.h: _MH},
        widths={ph.Z: _WZ},
        boson_fields=[ph.Z, ph.A, ph.Wp, ph.Wm, ph.h],
        conjugate_map=ph.cmap, conjugates={ph.Wp: ph.Wm, ph.Wm: ph.Wp},
        gauge_groups=[sm2["ew"].SU2L])


def test_from_model_matches_explicit_ee_to_mumu(sm2):
    (e, _), (mu, _) = sm2["leptons"]["e"], sm2["leptons"]["mu"]
    ph, ew = sm2["ph"], sm2["ew"]
    explicit = _explicit(sm2, [e, mu])
    derived = ScatteringCalculator.from_model(sm2["model"], [e, mu],
                                              widths={ph.Z: _WZ})
    key = lambda v: (tuple(map(str, v.particles)), v.vertex_type)
    assert sorted(map(key, derived.vertices())) == \
        sorted(map(key, explicit.vertices()))

    values = {ew.gw.s: _INPUTS.gw, ew.g1.s: _INPUTS.g1, ew.v.s: _INPUTS.v}
    ee, mumu = (e.particle, e.antiparticle), (mu.particle, mu.antiparticle)
    old = explicit.process(ee, mumu, numeric=True, extra=values)
    new = derived.process(ee, mumu)                 # numeric by default
    kin_o, kin_n = old.kinematics, new.kinematics
    for rs, cos in [(200.0, 0.3), (91.0, -0.7)]:
        at_o = {kin_o.s: rs**2, kin_o.t: kin_o.t_of_cos(cos).subs(kin_o.s, rs**2)}
        at_n = {kin_n.s: rs**2, kin_n.t: kin_n.t_of_cos(cos).subs(kin_n.s, rs**2)}
        a = complex(old.squared().subs(at_o))
        b = complex(new.squared().subs(at_n))
        assert b == pytest.approx(a, rel=1e-10)


def test_from_model_reproduces_ww_benchmark(sm2):
    e, nu = sm2["leptons"]["e"]
    ph = sm2["ph"]
    calc = ScatteringCalculator.from_model(sm2["model"], [e, nu],
                                           widths={ph.Z: _WZ})
    proc = calc.process((e.particle, e.antiparticle), (ph.Wp, ph.Wm))
    sigma = proc.numeric_cross_section(200.0**2) * _GEV2_TO_PB
    assert sigma == pytest.approx(19.447, abs=2e-3)      # test_ww.py value
    assert abs(sigma - 19.498) < 2 * 0.058               # MadGraph, 2σ


def test_process_point_overrides_parameters(sm2):
    """``point=`` replaces the calculator's parameters for one call: doubling
    the VEV at fixed couplings doubles m_Z and moves the Z pole."""
    e, _ = sm2["leptons"]["e"]
    mu, _ = sm2["leptons"]["mu"]
    calc = ScatteringCalculator.from_model(sm2["model"], [e, mu],
                                           widths={sm2["ph"].Z: _WZ})
    doubled =ParameterPoint(sm2["model"].parameters, {"v": 2 * _INPUTS.v})
    m_z = calc.mass_of(sm2["ph"].Z)
    assert float(m_z.subs(calc.values())) == pytest.approx(_INPUTS.MZ)
    assert float(m_z.subs(calc.values(point=doubled))) == \
        pytest.approx(2 * _INPUTS.MZ)

    ee, mumu = (e.particle, e.antiparticle), (mu.particle, mu.antiparticle)

    def squared_at(proc, rs):
        kin = proc.kinematics
        at = {kin.s: rs**2, kin.t: kin.t_of_cos(0.0).subs(kin.s, rs**2)}
        return complex(proc.squared().subs(at)).real

    base, moved = calc.process(ee, mumu), calc.process(ee, mumu, point=doubled)
    # each process peaks at its own Z pole
    assert squared_at(base, _INPUTS.MZ) > 10 * squared_at(moved, _INPUTS.MZ)
    assert squared_at(moved, 2 * _INPUTS.MZ) > \
        10 * squared_at(base, 2 * _INPUTS.MZ)


def test_explicit_calculator_stays_symbolic_by_default(sm2):
    """No ``parameters`` and no ``extra``: ``numeric=None`` keeps the old
    symbolic behaviour."""
    e, mu = sm2["leptons"]["e"][0], sm2["leptons"]["mu"][0]
    ph, ew = sm2["ph"], sm2["ew"]
    calc = ScatteringCalculator(
        sm2["model"], [e, mu], masses={ph.A: 0, ph.Z: sp.Symbol("mZ")},
        boson_fields=[ph.Z, ph.A], conjugate_map=ph.cmap)
    proc = calc.process((e.particle, e.antiparticle),
                        (mu.particle, mu.antiparticle))
    assert ew.gw.s in proc.squared().free_symbols


def test_from_model_without_physical_basis_raises():
    ew = electroweak_scaffold()
    L = Lagrangian()
    ew.add_higgs(L)
    model = Model("Unrotated", gauge_groups=ew.gauge_groups, fields=ew.fields,
                  parameters=ew.parameters, lagrangian=L)
    with pytest.raises(ValueError, match="no physical basis"):
        ScatteringCalculator.from_model(model, [])
    with pytest.raises(ValueError, match="no physical basis"):
        DecayCalculator.from_model(model)


# --------------------------------------------------------------------------
# DecayCalculator.from_model
# --------------------------------------------------------------------------

def test_decay_from_model_z_to_nunu():
    sm = standard_model()
    (Ll, eR), = sm.leptons
    ph, sc = sm.physical, sm.scaffold
    nu = DiracParticle("nu", Ll.components[0], None, 0)
    e = DiracParticle("e", Ll.components[1], eR.components[0], 0)
    calc = DecayCalculator.from_model(sm.model, particles=[nu, e],
                                      fermion_sectors=("gauge",))
    g, gp, v = sc.gw.s, sc.g1.s, sc.v.s
    m_z = sp.sqrt(g**2 + gp**2) * v / 2
    assert sp.simplify(calc.mass_of(ph.Z) - m_z) == 0
    width = calc.partial_widths(ph.Z)[(nu.antiparticle, nu.particle)]
    assert sp.simplify(width - m_z * (g**2 + gp**2) / (96 * sp.pi)) == 0
    numbers = calc.numeric_partial_widths(ph.Z)
    expected = float((m_z * (g**2 + gp**2) / (96 * sp.pi)).subs(
        sm.model.parameters.numeric()))
    assert numbers[(nu.antiparticle, nu.particle)] == pytest.approx(expected)
    # a point passed per call wins over the calculator's parameters
    moved = calc.numeric_partial_widths(
        ph.Z, point=ParameterPoint(sm.model.parameters, {"v": 2 * 246.0}))
    assert moved[(nu.antiparticle, nu.particle)] == pytest.approx(
        2 * expected, rel=1e-12)


# --------------------------------------------------------------------------
# ParameterPoint
# --------------------------------------------------------------------------

@pytest.fixture
def params():
    P = ParameterSet()
    a = P.external("a", 2.0)
    b = P.external("b", 3.0)
    c = P.internal("c", a.s * b.s)
    P.internal("d", c.s + 1)
    return P


def test_numeric_without_overrides_unchanged(params):
    values = params.numeric()
    assert values[params["c"].symbol] == 6.0
    assert values[params["d"].symbol] == 7.0


def test_point_rederives_internals_without_mutating(params):
    point = ParameterPoint(params, {"a": 5.0})
    assert point["c"] == 15.0 and point["d"] == 16.0
    assert params["a"].value == 2.0                      # untouched
    scanned = point.replace(b=1.0)
    assert scanned["c"] == 5.0 and point["c"] == 15.0    # original unchanged
    assert ParameterPoint(params, {params["a"]: 1.0})["c"] == 3.0
    assert ParameterPoint(params, {params["a"].symbol: 1.0})["c"] == 3.0


def test_point_extra_and_numeric_alias(params):
    m = sp.Symbol("m_b")
    point = ParameterPoint(params, extra={m: 4.18})
    assert point[m] == 4.18 and point["m_b"] == 4.18
    assert point.numeric() == point.values()
    assert "| `c` | 6 | derived |" in point._repr_markdown_()


def test_point_refuses_internal_and_unknown(params):
    with pytest.raises(ValueError, match="internal"):
        ParameterPoint(params, {"c": 1.0})
    with pytest.raises(KeyError, match="no such parameter"):
        ParameterPoint(params, {"zz": 1.0})
    with pytest.raises(ValueError, match="internal"):
        params.numeric({"d": 0.0})

"""Process names, units, acceptance cuts, and readable results.

What is pinned:

- a process string (``"e ebar > Wp Wm"``, MadGraph-style ``"e- e+ -> w+ w-"``
  through aliases) gives the same diagrams and ``Σ|M|²`` as the symbol
  tuples; unknown names, wrong multiplicities and colliding aliases raise;
- ``σ(e⁺e⁻→W⁺W⁻, √s = 200 GeV) = 19.447 pb`` through ``unit="pb"``
  (``test_ww.py``; MadGraph 19.498 ± 0.058 pb), ``σ(e⁺e⁻→μ⁺μ⁻)`` = 2.7876 pb
  inside ``|η| < 2.5`` (``test_interference.py``; MadGraph 2.7878 ± 0.0027
  pb), and ``A_FB(200 GeV) = 0.565``;
- the barn conversion uses the PDG ``(ħc)² = 0.3893793721 GeV² mb``;
- ``summary()`` reports the numbers ``numeric_cross_section`` gives;
- ``decay_table`` reproduces ``Γ(Z→νν̄) = m_Z(g²+g'²)/96π``, its BRs sum to 1,
  and a channel closed at the point is ``0``, not an imaginary width.
"""

import math

import pytest
import sympy as sp

from feynlag import (
    Lagrangian, Model, WeylFermion, electroweak_inputs, electroweak_scaffold,
    fermion_gauge_current, standard_model, to_physical_basis,
)
from feynlag.pheno import (
    HBARC2_GEV2_MB, DecayCalculator, DiracParticle, ParticleNames,
    ScatteringCalculator, cross_section_in, diagram_row, energy_in,
    eta_acceptance,
)
from feynlag.pheno.engine import BosonDiagramInfo

_INPUTS = electroweak_inputs()          # docs/benchmark.md point
_WZ, _MH = 2.4952, 125.0


@pytest.fixture(scope="module")
def sm():
    """SM electroweak sector with massless e, ν_e, μ at the benchmark point."""
    ew = electroweak_scaffold(**_INPUTS.couplings, mh=_MH)
    SU2L, U1Y = ew.SU2L, ew.U1Y
    i = sp.Symbol("i", integer=True)
    fields, lep = [], {}
    for name in ("e", "mu"):
        Ll = WeylFermion(f"UL{name}", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                         chirality="L", nflavors=1,
                         component_names=[f"Unu{name}L", f"U{name}L"])
        lR = WeylFermion(f"U{name}R", reps={U1Y: -1}, chirality="R",
                         nflavors=1, component_names=[f"U{name}R"])
        fields += [Ll, lR]
        lep[name] = DiracParticle(name, Ll.components[1], lR.components[0], 0)
        lep["nu_" + name] = DiracParticle(f"nu_{name}", Ll.components[0],
                                          None, 0)
    L = Lagrangian()
    ew.add_higgs(L)
    for f in fields:
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("Usability", gauge_groups=ew.gauge_groups,
                  fields=ew.fields + fields, parameters=ew.parameters,
                  lagrangian=L)
    model.solve_tadpoles([ew.mu2])
    ph = to_physical_basis(model, ew)
    e, mu = lep["e"], lep["mu"]
    aliases = {"e-": e.particle, "e+": e.antiparticle,
               "mu-": mu.particle, "mu+": mu.antiparticle}
    calc = ScatteringCalculator.from_model(
        model, [e, lep["nu_e"], mu], widths={ph.Z: _WZ}, aliases=aliases)
    return dict(model=model, ph=ph, lep=lep, calc=calc)


@pytest.fixture(scope="module")
def ww(sm):
    """``e⁺e⁻→W⁺W⁻`` from a string, squared once for the module."""
    proc = sm["calc"].process("e- e+ > w+ w-")
    proc.squared()
    return proc


def _squared_at(proc, rs, cos):
    kin = proc.kinematics
    at = {kin.s: rs**2, kin.t: kin.t_of_cos(cos).subs(kin.s, rs**2)}
    return complex(proc.squared().subs(at))


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------

def test_process_string_matches_symbol_tuples(sm, ww):
    """Same diagrams for every spelling of ``e⁺e⁻→W⁺W⁻`` (squared once, in
    the fixture), and the same ``Σ|M|²`` for ``e⁺e⁻→μ⁺μ⁻``, which is cheap."""
    calc, ph, e, mu = sm["calc"], sm["ph"], sm["lep"]["e"], sm["lep"]["mu"]
    by_symbol = calc.process((e.particle, e.antiparticle), (ph.Wp, ph.Wm))
    rows = [diagram_row(d) for d in by_symbol.diagrams]
    for proc in (calc.process("e ebar > Wp Wm"),
                 calc.process(("e", "ebar"), ("W+", "W-")), ww):
        assert [diagram_row(d) for d in proc.diagrams] == rows
    assert ww.label == "e- e+ → w+ w-"
    assert by_symbol.label == "e ebar → Wp Wm"

    reference = calc.process((e.particle, e.antiparticle),
                             (mu.particle, mu.antiparticle))
    for proc in (calc.process("e ebar -> mu mubar"),
                 calc.process("E- e+ → MU- mu+")):
        for rs, cos in [(200.0, 0.3), (91.0, -0.7)]:
            assert _squared_at(proc, rs, cos) == pytest.approx(
                _squared_at(reference, rs, cos), rel=1e-12)


def test_default_aliases_come_from_to_physical_basis(sm):
    ph, names = sm["ph"], sm["calc"].particle_names()
    for alias, boson in [("a", ph.A), ("gamma", ph.A), ("z", ph.Z),
                         ("w+", ph.Wp), ("w-", ph.Wm), ("h", ph.h)]:
        assert names[alias] == boson
    explicit = ScatteringCalculator(
        sm["model"], [sm["lep"]["e"]], masses={ph.A: 0}, boson_fields=[ph.A])
    assert "gamma" not in explicit.particle_names()


def test_name_errors(sm):
    calc = sm["calc"]
    with pytest.raises(KeyError, match="did you mean.*'Wp'"):
        calc.process("e ebar > Wplus Wm")
    with pytest.raises(ValueError, match="only 2→2"):
        calc.process("e ebar > Wp Wm h")
    with pytest.raises(ValueError, match="in1 in2 > out1 out2"):
        calc.process("e ebar Wp Wm")
    with pytest.raises(TypeError):
        calc.process("e ebar > Wp Wm", ("Wp", "Wm"))


def test_alias_collisions_raise():
    a, b, c = sp.symbols("a b c")
    with pytest.raises(ValueError, match="collides with the symbol name"):
        ParticleNames([a, b], {"a": b})
    with pytest.raises(ValueError, match="ambiguous"):
        ParticleNames([a, b], {"X": a, "x": b})
    with pytest.raises(ValueError, match="not an external"):
        ParticleNames([a], {"x": c})
    names = ParticleNames([a, b], {"a": a, "Bee": b})
    assert names.resolve("bee") == b and names.resolve("a") == a


# --------------------------------------------------------------------------
# units, energy, cuts, A_FB
# --------------------------------------------------------------------------

def test_conversion_constants():
    assert HBARC2_GEV2_MB == 0.3893793721           # PDG, exact
    assert cross_section_in(1.0, "pb") == pytest.approx(0.3893793721e9,
                                                        rel=1e-15)
    assert cross_section_in(2.0, "fb") == pytest.approx(
        1e3 * cross_section_in(2.0, "pb"), rel=1e-15)
    assert cross_section_in(1.0, "PB") == cross_section_in(1.0, "pb")
    assert energy_in(0.0025, "MeV") == pytest.approx(2.5)
    with pytest.raises(ValueError, match="unknown cross-section unit"):
        cross_section_in(1.0, "barn")
    assert eta_acceptance(2.5) == (-math.tanh(2.5), math.tanh(2.5))


def test_ww_in_picobarn(ww):
    proc = ww
    sigma = proc.numeric_cross_section(sqrt_s=200, unit="pb")
    assert sigma == pytest.approx(19.447, abs=2e-3)       # test_ww.py value
    assert abs(sigma - 19.498) < 2 * 0.058                # MadGraph, 2σ
    natural = proc.numeric_cross_section(200.0**2)       # old call shape
    assert sigma == pytest.approx(cross_section_in(natural, "pb"), rel=1e-12)
    assert proc.numeric_cross_section(sqrt_s=200, unit="fb") == \
        pytest.approx(1e3 * sigma, rel=1e-12)
    with pytest.raises(TypeError, match="exactly one"):
        proc.numeric_cross_section(200.0**2, sqrt_s=200)
    with pytest.raises(TypeError, match="exactly one"):
        proc.numeric_cross_section()


def test_eta_cut_reproduces_madgraph(sm):
    proc = sm["calc"].process("e- e+ > mu- mu+")
    sigma = proc.numeric_cross_section(sqrt_s=200, unit="pb", eta_max=2.5)
    assert sigma == pytest.approx(2.7876, abs=2e-4)       # test_interference
    assert abs(sigma - 2.7878) < 2 * 0.0027               # MadGraph, 2σ
    with pytest.raises(TypeError, match="not both"):
        proc.numeric_cross_section(sqrt_s=200, eta_max=2.5,
                                   cos_range=(-0.5, 0.5))


def test_numeric_forward_backward_asymmetry(sm):
    proc = sm["calc"].process("e- e+ > mu- mu+")
    afb = proc.numeric_forward_backward_asymmetry(sqrt_s=200)
    symbolic = complex(proc.forward_backward_asymmetry().subs(
        proc.kinematics.s, 200.0**2)).real
    assert afb == pytest.approx(symbolic, rel=1e-12)
    assert afb == pytest.approx(0.565, abs=1e-3)
    zz = sm["calc"].process("e- e+ > z z")
    with pytest.raises(ValueError, match="identical final-state"):
        zz.numeric_forward_backward_asymmetry(sqrt_s=200)


# --------------------------------------------------------------------------
# summaries and reprs
# --------------------------------------------------------------------------

def test_summary_reports_the_calculator_numbers(ww):
    summary = ww.summary(sqrt_s=200, unit="pb")
    assert summary.sigma == ww.numeric_cross_section(sqrt_s=200, unit="pb")
    assert summary.initial_dof == 4                 # two massless fermions
    assert sorted(summary.diagrams) == [
        ("s", "A", "boson", ""), ("s", "Z", "boson", ""),
        ("u", "nu_e", "fermion", "")]
    text = str(summary)
    assert "e- e+ → w+ w-" in text and "3 diagrams" in text
    assert "sigma = 19.44" in text and "pb" in text
    md = ww._repr_markdown_()
    assert md.startswith("**e- e+ → w+ w-**")
    assert "| channel | mediator | exchange | note |" in md
    assert repr(ww).startswith("ScatteringProcess(e- e+ → w+ w-; 3 diagrams:")


def test_summary_without_energy_evaluates_nothing(sm):
    bhabha = sm["calc"].process("e- e+ > e- e+")
    summary = bhabha.summary()
    assert summary.sigma is None and summary.afb is None
    rows = sorted(summary.diagrams)
    assert [r[:2] for r in rows] == [("s", "A"), ("s", "Z"), ("t", "A"),
                                     ("t", "Z")]
    # the s and t channels carry the opposite fermion sign
    assert {r[0]: r[3] for r in rows}["s"] != {r[0]: r[3] for r in rows}["t"]


def test_summary_of_identical_final_states_has_no_afb(sm):
    summary = sm["calc"].process("e- e+ > z z").summary(sqrt_s=250)
    assert summary.afb is None and summary.symmetry_factor == sp.Rational(1, 2)
    assert "symmetry factor 1/2" in str(summary)


def test_diagram_row_of_a_four_gluon_structure():
    g = sp.Symbol("G")
    info = BosonDiagramInfo(channel="s", mediator=g, pairing=((0, 1), (2, 3)),
                            structure="VVVV1", vertices=(), terms=())
    assert diagram_row(info) == ("s", "G", "boson", "+ VVVV1 contact")


# --------------------------------------------------------------------------
# decay tables
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zdecay():
    model = standard_model()
    (Ll, eR), = model.leptons
    nu = DiracParticle("nu", Ll.components[0], None, 0)
    e = DiracParticle("e", Ll.components[1], eR.components[0], 0)
    calc = DecayCalculator.from_model(model.model, particles=[nu, e],
                                      fermion_sectors=("gauge",))
    return dict(sm=model, calc=calc, nu=nu, e=e)


def test_decay_table_z(zdecay):
    calc, sm, nu = zdecay["calc"], zdecay["sm"], zdecay["nu"]
    sc = sm.scaffold
    g, gp, v = sc.gw.s, sc.g1.s, sc.v.s
    expected = float((sp.sqrt(g**2 + gp**2) * v / 2 * (g**2 + gp**2)
                      / (96 * sp.pi)).subs(sm.model.parameters.numeric()))
    table = calc.decay_table("z")                          # an alias
    assert table[("nu", "nubar")].width == pytest.approx(expected)
    assert table[(nu.antiparticle, nu.particle)].width == \
        pytest.approx(expected)
    assert sum(table.branching_ratios.values()) == pytest.approx(1.0)
    widths = [r.width for r in table.rows]
    assert widths == sorted(widths, reverse=True)
    assert table.total == pytest.approx(
        calc.numeric_total_width(sm.physical.Z))
    in_mev = calc.decay_table(sm.physical.Z, unit="MeV")
    assert in_mev[("nu", "nubar")].width == pytest.approx(1e3 * expected)
    assert "width [MeV]" in str(in_mev) and "total width" in str(in_mev)
    assert "| channel | width [MeV] | BR |" in in_mev._repr_markdown_()
    with pytest.raises(KeyError):
        table[("e", "nubar")]


def test_decay_table_folds_small_channels(zdecay):
    calc = zdecay["calc"]
    full = calc.decay_table("Z")
    nunu = full[("nu", "nubar")].branching_ratio
    assert full.other is None
    folded = calc.decay_table("Z", min_br=0.5)        # only νν̄ (2/3) stays
    assert [r.children for r in folded.rows] == [full.rows[0].children]
    assert folded.other[1] == pytest.approx(1 - nunu)
    assert folded.total == pytest.approx(full.total)
    assert "other" in str(folded)


def test_from_model_decays_have_no_stray_goldstone_channels(zdecay):
    """With the Goldstones left out of ``boson_fields`` the extractor used to
    read ``h A W⁺ G⁻`` as a three-leg ``h → A W⁺`` with coupling ``∝ G⁻``
    (and ``Z → h W±``); a vertex is taken at zero fields, so these are gone
    and only the genuine SM couplings remain."""
    calc, ph = zdecay["calc"], zdecay["sm"].physical
    h_children = {frozenset(c.children) for c in calc.channels(ph.h)}
    assert h_children == {frozenset({ph.Wp, ph.Wm}), frozenset({ph.Z}),
                          frozenset({ph.h})}
    z_children = {frozenset(c.children) for c in calc.channels(ph.Z)}
    # Z → Zh is the genuine (closed) ZZh coupling; Z → h W± was the fake
    assert z_children == {frozenset(map(sp.Symbol, ("nu", "nubar"))),
                          frozenset(map(sp.Symbol, ("e", "ebar"))),
                          frozenset({ph.Z, ph.h})}
    goldstones = {ph.G0, ph.Gp, ph.Gm}
    for vertex in calc.vertices():
        assert not sp.sympify(vertex.coupling).free_symbols & goldstones


def test_decay_table_closed_channel_is_zero(zdecay):
    """h → W⁺W⁻ and ZZ are closed at m_h = 125 GeV: exactly 0, never an
    imaginary width from √λ < 0 (the documented threshold trap)."""
    table = zdecay["calc"].decay_table("h")
    assert table.rows and all(r.width == 0.0 for r in table.rows)
    assert table.total == 0.0

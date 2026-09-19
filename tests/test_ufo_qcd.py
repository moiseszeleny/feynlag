"""UFO color-tensor export for the QCD (SU(3)) sector.

Checks that add_fermion_vertex/add_vvv_vertex/add_vvvv_vertex emit real
color-tensor strings (not the color-singlet '1' default) for a
quark-gluon (qqg), 3-gluon (ggg), and 4-gluon (gggg) vertex, with numeric
coupling values matching the already-pinned physics in tests/test_qcd.py
and tests/test_yangmills.py.
"""

import sympy as sp
import pytest

from feynlag import ExternalParameter, ParameterSet, SU3
from feynlag.export.ufo import UFOParticle, write_ufo
from feynlag.export.ufo.vvvv import (ADJOINT_VVV_COLOR,
                                     ADJOINT_VVVV_COLORS,
                                     adjoint_vvv, adjoint_vvvv)

import importlib
import sys


_UFO_SUBMODULES = ("object_library", "function_library", "coupling_orders",
                   "parameters", "couplings", "lorentz", "particles",
                   "vertices")


def _import_ufo(path):
    """Load a UFO directory the MadGraph way (dir on sys.path, absolute
    imports); return object_library holding the all_* registries."""
    sys.path.insert(0, str(path))
    for mod in _UFO_SUBMODULES:
        sys.modules.pop(mod, None)
    try:
        import object_library
        for mod in _UFO_SUBMODULES[1:]:
            importlib.import_module(mod)
        return object_library
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def qcd_ufo(tmp_path_factory):
    gs = ExternalParameter("gs", 1.22, positive=True)

    q = sp.Symbol("q")
    qbar = sp.Symbol("qbar")
    g = sp.Symbol("g")

    particles = [
        UFOParticle(q, 1, "q", antiname="q~", spin=2, color=3,
                    antisymbol=qbar),
        UFOParticle(g, 21, "g", spin=3, color=8),
    ]

    params = ParameterSet(gs)

    # qqg: T^1_{0,1} = 1/2 (Gell-Mann lambda^1/2) => coefficient +gs/2,
    # matching tests/test_qcd.py::test_qqg_coupling_pinned.
    fermion_vertices = [
        dict(bar=qbar, field=q, bosons=(g,), left=gs.s / 2,
             color="T(3,2,1)"),
    ]
    SU3c = SU3("SU3c", coupling=gs)

    # ggg: ONE physical gluon repeated three times, with the adjoint index
    # carried by the color tensor — so the coupling is COLOR-STRIPPED, like
    # the gggg below. This used to be the hardcoded literal -gs.s, which is
    # cubic_couplings' value for the (G_1,G_2,G_3) triple and contains the
    # color factor; it was right only because f^123 = 1.
    vvv = {(g, g, g): adjoint_vvv(SU3c)}
    vvv_colors = {(g, g, g): ADJOINT_VVV_COLOR}

    # gggg: ONE physical gluon repeated four times, with the adjoint index
    # carried by the color tensors — so the coupling must be COLOR-STRIPPED.
    # This used to pass assemble_vvvv's raw values for the specific quadruple
    # (G_1,G_2,G_4,G_5), which already contain sum_e f_{ije} f_{kle}, and so
    # multiplied by color twice. adjoint_vvvv divides it back out.
    vvvv = {(g, g, g, g): adjoint_vvvv(SU3c)}
    vvvv_colors = {(g, g, g, g): dict(ADJOINT_VVVV_COLORS)}

    out = tmp_path_factory.mktemp("ufo") / "QCD_UFO"
    write_ufo(out, "QCD", params, particles, vvv=vvv, vvv_colors=vvv_colors,
              vvvv=vvvv, vvvv_colors=vvvv_colors,
              fermion_vertices=fermion_vertices)
    return out, dict(gs=1.22)


def test_qqg_color_string(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    names = {p.name for p in ufo.all_particles}
    assert {"q", "q~", "g"} <= names
    for vert in ufo.all_vertices:
        pnames = sorted(p.name for p in vert.particles)
        if pnames == sorted(["q", "q~", "g"]):
            # T(a,i,j): i is the FUNDAMENTAL index, so with the
            # [bar, field, g] leg order the field leg comes first.
            # T(3,1,2) transposes a hermitian T^a (= conjugates it)
            # and flipped the ggg interference; u u~ > g g failed
            # MadGraph's gauge check until this was corrected.
            assert vert.color == ["T(3,2,1)"]
            break
    else:
        pytest.fail("qqg vertex not found")


def test_ggg_color_string_and_coupling(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    for vert in ufo.all_vertices:
        if [p.name for p in vert.particles] == ["g", "g", "g"]:
            assert vert.color == ["f(1,2,3)"]
            coupling = list(vert.couplings.values())[0]
            value = complex(eval(coupling.value, {"gs": num["gs"]}))
            assert abs(value - (-num["gs"])) < 1e-9
            break
    else:
        pytest.fail("ggg vertex not found")


def test_gggg_color_strings_and_couplings(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    expected_colors = dict(ADJOINT_VVVV_COLORS)
    # MadGraph stock sm V_37 [g,g,g,g] carries GC_12 = i*G**2 on all three
    # structures with these same (cyclically identical) color tensors.
    expected_values = {name: 1j * num["gs"] ** 2 for name in expected_colors}
    for vert in ufo.all_vertices:
        if [p.name for p in vert.particles] == ["g", "g", "g", "g"]:
            lorentz_names = [l.name for l in vert.lorentz]
            assert set(lorentz_names) == set(expected_colors)
            assert vert.color == [expected_colors[n] for n in lorentz_names]
            for slot, lname in enumerate(lorentz_names):
                coupling = vert.couplings[(slot, slot)]
                value = complex(eval(coupling.value, {"gs": num["gs"]}))
                assert abs(value - expected_values[lname]) < 1e-9
            break
    else:
        pytest.fail("gggg vertex not found")


def test_gluon_particle_is_color_octet_self_conjugate(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    (gluon,) = [p for p in ufo.all_particles if p.name == "g"]
    assert gluon.color == 8
    assert gluon.selfconjugate


def test_quark_antiquark_color_conjugate(qcd_ufo):
    path, num = qcd_ufo
    ufo = _import_ufo(path)
    quark = next(p for p in ufo.all_particles if p.name == "q")
    antiquark = next(p for p in ufo.all_particles if p.name == "q~")
    assert quark.color == 3
    assert antiquark.color == -3


def test_coupling_order_is_qcd_only_for_coloured_tensors():
    """The order decides which diagrams MadGraph builds.

    Tagging gluon vertices QED makes MG reject the model outright
    (CRITICAL: Model with non QCD emission of gluon). But the mirror mistake
    is just as bad: Identity(i,j) is the colour SINGLET for a coloured
    fermion pair with a colourless boson (q qbar gamma), so treating every
    non-'1' tensor as QCD would tag the whole electroweak quark sector QCD.
    """
    from feynlag.export.ufo.writer import _order_name
    for singlet in ("1", "", " 1 ", "Identity(1,2)"):
        assert _order_name(singlet) == "QED", singlet
    for coloured in ("T(3,2,1)", "f(1,2,3)", "f(1,2,-1)*f(3,4,-1)",
                     "d(1,2,3)"):
        assert _order_name(coloured) == "QCD", coloured


def test_same_value_different_colour_gets_distinct_couplings():
    """A coupling's order is part of its identity: two vertices sharing a
    VALUE but differing in colour must not collapse onto one GC_n, or
    whichever registered first would silently decide the order for both."""
    import sympy as sp
    from feynlag.export.ufo.writer import _UFOBuilder, UFOParticle

    q, qbar, g = sp.symbols("q qbar g")
    builder = _UFOBuilder("X", None, [
        UFOParticle(q, 1, "q", antiname="q~", spin=2, color=3,
                    antisymbol=qbar),
        UFOParticle(g, 21, "g", spin=3, color=8),
    ])
    gs = sp.Symbol("gs", positive=True)
    singlet = builder._coupling(sp.I * gs, 3, "1")
    coloured = builder._coupling(sp.I * gs, 3, "T(3,2,1)")
    assert singlet != coloured, "same GC_n reused across colour classes"
    orders = {name: order for (name, order) in builder.couplings.values()}
    assert orders[singlet] == {"QED": 1}
    assert orders[coloured] == {"QCD": 1}

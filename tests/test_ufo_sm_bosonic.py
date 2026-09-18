"""The exported SM UFO's bosonic vertices, against MadGraph's stock ``sm``.

``scripts/export_sm_ufo.py`` writes a MadGraph-importable electroweak UFO at
the stock ``(aEWM1, Gf, MZ)`` parameter point, so its couplings are directly
comparable — numerically, no basis guesswork — to the model MadGraph ships.
This pins the two things that were missing from it entirely:

- the four quartic gauge couplings (WWWW / WWZZ / WWAA / WWAZ), which the
  script never exported at all (it passed ``vvv=`` but never ``vvvv=``);
- the VVS/VVSS couplings, which used to be two HAND-WRITTEN ``Vertex``
  objects (hWW, hZZ) rather than the extractor's own output, so hhWW/hhZZ
  were absent and nothing checked the hand-written pair against the pipeline.

MadGraph's VVVV basis differs from feynlag's, so the comparison is done in the
convention-free metric-pair basis ``c12*M12M34 + c13*M13M24 + c14*M14M23``:

    VVVV2_MG = M14M23 + M13M24 - 2*M12M34
    VVVV5_MG = M14M23 - M13M24/2 - M12M34/2

Stock values (models/sm/{vertices,couplings}.py):
``GC_5 = i ee^2`` [a,a,W-,W+] · ``GC_35 = -i ee^2/sw^2`` [W-,W-,W+,W+] ·
``GC_36 = i cw^2 ee^2/sw^2`` [W-,W+,Z,Z] · ``GC_57 = -2i cw ee^2/sw``
[a,W-,W+,Z] · ``GC_72`` hWW · ``GC_81`` hZZ · ``GC_34`` hhWW · ``GC_65`` hhZZ.
"""

import cmath
import importlib
import math
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]
                       / "scripts"))

METRIC_PAIRS = {            # (c12, c13, c14) per feynlag structure
    "VVVV1": (0, -1, 1),
    "VVVV2": (-1, 0, 1),
    "VVVV3": (-1, 1, 0),
}


def _stock_point():
    """MadGraph's stock SM electroweak input scheme."""
    aEWM1, Gf, MZ = 132.50698, 1.16639e-5, 91.1876
    aEW = 1 / aEWM1
    ee = math.sqrt(4 * math.pi * aEW)
    MW = math.sqrt(MZ ** 2 / 2
                   + math.sqrt(MZ ** 4 / 4 - aEW * math.pi * MZ ** 2
                               / (Gf * math.sqrt(2))))
    sw = math.sqrt(1 - MW ** 2 / MZ ** 2)
    cw = math.sqrt(1 - sw ** 2)
    return dict(ee=ee, sw=sw, cw=cw, v=2 * MW * sw / ee,
                gw=ee / sw, g1=ee / cw)


@pytest.fixture(scope="module")
def sm_ufo(tmp_path_factory):
    export_sm_ufo = importlib.import_module("export_sm_ufo")
    path = tmp_path_factory.mktemp("ufo") / "FEYNLAG_SM"
    export_sm_ufo.export(path)

    saved = list(sys.path)
    sys.path.insert(0, str(path))
    for mod in ("object_library", "particles", "parameters", "couplings",
                "lorentz", "vertices"):
        sys.modules.pop(mod, None)
    try:
        ol = importlib.import_module("object_library")
        for mod in ("particles", "parameters", "couplings", "lorentz",
                    "vertices"):
            importlib.import_module(mod)
        yield ol, _stock_point()
    finally:
        sys.path[:] = saved
        for mod in ("object_library", "particles", "parameters", "couplings",
                    "lorentz", "vertices"):
            sys.modules.pop(mod, None)


def _vertex(ol, names):
    for vt in ol.all_vertices:
        if sorted(p.name for p in vt.particles) == sorted(names):
            return vt
    raise AssertionError(
        f"no vertex {names}; have "
        f"{[sorted(p.name for p in v.particles) for v in ol.all_vertices]}")


def _evaluate(vt, point):
    """``{lorentz name: complex value}`` at the stock parameter point."""
    env = dict(point, complex=complex, cmath=cmath, math=math)
    lor = [l.name for l in vt.lorentz]
    return {lor[j]: complex(eval(cp.value, dict(env)))
            for (i, j), cp in vt.couplings.items()}


def _metric_pairs(vt, point):
    """``(c12, c13, c14)`` of a VVVV vertex at the stock point."""
    c12 = c13 = c14 = 0j
    for name, val in _evaluate(vt, point).items():
        d12, d13, d14 = METRIC_PAIRS[name]
        c12 += d12 * val
        c13 += d13 * val
        c14 += d14 * val
    return c12, c13, c14


def _assert_close(got, want, label):
    assert abs(got - want) < 1e-9 * max(1.0, abs(want)), \
        f"{label}: {got} != {want}"


class TestQuarticGaugeCouplings:
    """Never exported before — the script passed vvv= but never vvvv=."""

    def test_aaww(self, sm_ufo):
        ol, p = sm_ufo
        c12, c13, c14 = _metric_pairs(_vertex(ol, ["a", "a", "W-", "W+"]), p)
        want = 1j * p["ee"] ** 2                       # GC_5
        _assert_close(c14, want, "AAWW M14M23")
        _assert_close(c13, want, "AAWW M13M24")
        _assert_close(c12, -2 * want, "AAWW M12M34")

    def test_wwww(self, sm_ufo):
        ol, p = sm_ufo
        c12, c13, c14 = _metric_pairs(
            _vertex(ol, ["W-", "W-", "W+", "W+"]), p)
        want = -1j * p["ee"] ** 2 / p["sw"] ** 2       # GC_35
        _assert_close(c14, want, "WWWW M14M23")
        _assert_close(c13, want, "WWWW M13M24")
        _assert_close(c12, -2 * want, "WWWW M12M34")

    def test_wwzz(self, sm_ufo):
        ol, p = sm_ufo
        c12, c13, c14 = _metric_pairs(_vertex(ol, ["W-", "W+", "Z", "Z"]), p)
        want = 1j * p["cw"] ** 2 * p["ee"] ** 2 / p["sw"] ** 2   # GC_36
        _assert_close(c14, want, "WWZZ M14M23")
        _assert_close(c13, want, "WWZZ M13M24")
        _assert_close(c12, -2 * want, "WWZZ M12M34")

    def test_awwz(self, sm_ufo):
        """MG's VVVV5 shape — all three metric-pair coefficients differ, so
        this is the one that cannot be matched by a lucky overall factor."""
        ol, p = sm_ufo
        c12, c13, c14 = _metric_pairs(
            _vertex(ol, ["a", "W-", "W+", "Z"]), p)
        want = -2j * p["cw"] * p["ee"] ** 2 / p["sw"]            # GC_57
        _assert_close(c14, want, "AWWZ M14M23")
        _assert_close(c13, -want / 2, "AWWZ M13M24")
        _assert_close(c12, -want / 2, "AWWZ M12M34")

    def test_no_four_photon_vertex(self, sm_ufo):
        ol, _ = sm_ufo
        for vt in ol.all_vertices:
            assert sorted(p.name for p in vt.particles) != ["a"] * 4


class TestScalarVectorCouplings:
    """hWW/hZZ used to be hand-written; hhWW/hhZZ were missing entirely."""

    def test_hww(self, sm_ufo):
        ol, p = sm_ufo
        got = _evaluate(_vertex(ol, ["W-", "W+", "h"]), p)["VVS1"]
        _assert_close(got, 1j * p["ee"] ** 2 * p["v"] / (2 * p["sw"] ** 2),
                      "hWW")                                     # GC_72

    def test_hzz(self, sm_ufo):
        ol, p = sm_ufo
        got = _evaluate(_vertex(ol, ["Z", "Z", "h"]), p)["VVS1"]
        _assert_close(got, 1j * p["ee"] ** 2 * p["v"]
                      / (2 * p["sw"] ** 2 * p["cw"] ** 2), "hZZ")  # GC_81

    def test_hhww(self, sm_ufo):
        ol, p = sm_ufo
        got = _evaluate(_vertex(ol, ["W-", "W+", "h", "h"]), p)["VVSS1"]
        _assert_close(got, 1j * p["ee"] ** 2 / (2 * p["sw"] ** 2),
                      "hhWW")                                    # GC_34

    def test_hhzz(self, sm_ufo):
        ol, p = sm_ufo
        got = _evaluate(_vertex(ol, ["Z", "Z", "h", "h"]), p)["VVSS1"]
        _assert_close(got, 1j * p["ee"] ** 2
                      / (2 * p["sw"] ** 2 * p["cw"] ** 2), "hhZZ")  # GC_65

    def test_no_photon_higgs_coupling(self, sm_ufo):
        """Forbidden at tree level — the extractor must not leak one in."""
        ol, _ = sm_ufo
        for vt in ol.all_vertices:
            names = [p.name for p in vt.particles]
            assert not ("a" in names and "h" in names)

    def test_goldstones_are_not_exported(self, sm_ufo):
        """Unitary gauge: the Goldstones must be in the extraction field list
        (or the coefficients are wrong) but filtered out of the UFO."""
        ol, _ = sm_ufo
        exported = {p.name for p in ol.all_particles}
        assert not exported & {"G0", "G+", "G-", "Gp", "Gm"}

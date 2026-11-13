"""Field declaration tests."""

import sympy as sp
import pytest

from feynlag import (
    DiracFermion, ExternalParameter, GaugeBoson, Scalar, SU2, U1, WeylFermion,
    dag,
)


@pytest.fixture
def groups():
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    return SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)


def test_doublet_components(groups):
    SU2L, U1Y = groups
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Hp", "H0"])
    assert len(H.components) == 2
    assert H.components[0].name == "Hp"
    assert not H.components[0].is_real  # complex scalar
    assert H.mat.shape == (2, 1)


def test_singlet_default_name(groups):
    SU2L, U1Y = groups
    s = Scalar("sigma", reps={U1Y: 0}, real=True)
    assert len(s.components) == 1
    assert s.components[0].name == "sigma"
    assert s.components[0].is_real
    assert s.self_conjugate  # real => self-conjugate by default


def test_dag_row(groups):
    SU2L, U1Y = groups
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)})
    HdH = (dag(H) * H.mat)[0]
    expected = sum(sp.conjugate(c) * c for c in H.components)
    assert sp.expand(HdH - expected) == 0


def test_generators_doublet(groups):
    SU2L, U1Y = groups
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)})
    gens = H.generators(SU2L)
    assert len(gens) == 3
    # T3 = sigma3/2
    assert gens[2] == sp.Matrix([[sp.Rational(1, 2), 0],
                                 [0, -sp.Rational(1, 2)]])
    # U(1) generator on the full space: q * identity
    u1_gens = H.generators(U1Y)
    assert u1_gens[0] == sp.eye(2) / 2


def test_generators_singlet_field_trivial(groups):
    SU2L, U1Y = groups
    s = Scalar("s", reps={U1Y: 1})
    assert s.generators(SU2L) == [sp.zeros(1, 1)] * 3


def test_vev_expansion(groups):
    SU2L, U1Y = groups
    v = ExternalParameter("v", 246.0, positive=True)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Hp", "H0"])
    H.expand_vev({H.components[1]: v})
    shift = H.shift_map
    H0 = H.components[1]
    h_r, h_i = sp.Symbol("H0_r", real=True), sp.Symbol("H0_i", real=True)
    assert sp.expand(shift[H0] - (v.s + h_r + sp.I * h_i) / sp.sqrt(2)) == 0
    assert set(H.fluctuations) == {h_r, h_i}


def test_vev_on_wrong_component_raises(groups):
    SU2L, U1Y = groups
    v = ExternalParameter("v", 246.0, positive=True)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)})
    with pytest.raises(ValueError):
        H.expand_vev({sp.Symbol("nope"): v})


def test_gauge_boson_from_group(groups):
    SU2L, U1Y = groups
    W = SU2L.bosons("W")
    assert len(W.components) == 3
    assert all(c.is_real for c in W.components)
    assert SU2L.bosons() is W  # created once
    B = U1Y.bosons("B")
    assert len(B.components) == 1


def test_weyl_fermion_declaration(groups):
    SU2L, U1Y = groups
    L = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                    chirality="L", nflavors=3,
                    component_names=["nuL", "eL"])
    assert L.chirality == "L"
    assert isinstance(L.components[0], sp.IndexedBase)
    assert L.dim == 2


def test_dirac_fermion_fails_fast(groups):
    SU2L, U1Y = groups
    with pytest.raises(NotImplementedError, match="two WeylFermions"):
        DiracFermion("E", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)})


def test_named_component_access(groups):
    """``field.<component name>`` returns the stored component objects."""
    SU2L, U1Y = groups
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"], component_tex=["G^+", "H^0"])
    assert H.Gp is H.components[0] and H.H0 is H.components[1]
    # the tex-carrying symbol itself, not a same-named plain Symbol
    assert H.H0 != sp.Symbol("H0") and sp.latex(H.H0) == "H^0"

    v = ExternalParameter("v", 246.0, positive=True)
    with pytest.raises(AttributeError):
        H.H0_r                                   # no VEV expanded yet
    H.expand_vev({H.H0: v}, tex={"H0_r": "h"})
    re, im = H.fluctuations
    assert H.H0_r is re and H.H0_i is im
    assert sp.latex(H.H0_r) == "h"

    PsiL = WeylFermion("PsiL", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                       chirality="L", component_names=["NL", "EL"])
    assert PsiL.EL is PsiL.components[1]
    assert PsiL.ELbar is PsiL.bar_components[1]
    assert PsiL.bar(PsiL.EL) is PsiL.ELbar       # the method is untouched
    assert set(PsiL.named()) == {"NL", "EL", "NLbar", "ELbar"}


def test_named_access_errors_and_does_not_shadow(groups):
    import copy

    SU2L, U1Y = groups
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    with pytest.raises(AttributeError, match="components: Gp, H0"):
        H.nope
    # real attributes win, and optional-attribute probes still get defaults
    assert H.name == "H" and H.dim == 2
    assert getattr(H, "chirality", None) is None
    assert not hasattr(H, "_sympy_")
    assert copy.copy(H).components == H.components

"""The weak -> physical gauge-boson basis bridge (``feynlag/gauge_basis.py``).

Pinned physics — the electroweak quartics in the PHYSICAL basis, against
MadGraph's stock ``sm`` model (models/sm/vertices.py + couplings.py), all four
at MG's own leg ordering:

- [a, a, W-, W+]      GC_5  = i*ee**2                 (VVVV2_MG)
- [W-, W-, W+, W+]    GC_35 = -i*ee**2/sw**2          (VVVV2_MG)
- [W-, W+, Z, Z]      GC_36 = i*cw**2*ee**2/sw**2     (VVVV2_MG)
- [a, W-, W+, Z]      GC_57 = -2i*cw*ee**2/sw         (VVVV5_MG)

plus: the derived rotation matrix equals the matrix that used to be hand-typed
at every call site, and there is no 4-photon and no 4-Z vertex.

``tests/test_yangmills.py`` verifies the same quartics against a direct
functional differentiation of -1/4 F F; this file checks they survive the trip
through ``Model`` (rotations registered the ordinary way, ``U`` derived rather
than supplied).
"""

import sympy as sp
import pytest

from feynlag import (
    Dmu, ExternalParameter, InternalParameter, Lagrangian, Model, Rotation,
    SU2, SU3, Scalar, U1, adjoint_rotation, dag, gauge_self_couplings,
    physical_vector_basis, rotation_2x2,
)
from feynlag.export.ufo.vvvv import metric_pair_coefficients


@pytest.fixture(scope="module")
def ew():
    """SM electroweak sector with the physical-basis rotations registered."""
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.129)
    mu2 = InternalParameter("mu2", unit_dim=2)

    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.components[1]: v})
    HdH = (dag(H) * H.mat)[0]
    DH = Dmu(H)
    L = Lagrangian()
    L.add((dag(DH) * DH)[0], sector="kinetic")
    L.add(-(-mu2.s * HdH + lam.s * HdH ** 2), sector="potential")

    model = Model("SM-EW", gauge_groups=[SU2L, U1Y],
                  fields=[H, SU2L.bosons("W"), U1Y.bosons("B")],
                  parameters=[gw, g1, v, lam, mu2], lagrangian=L)
    model.solve_tadpoles([mu2])

    g, gp = gw.s, g1.s
    W1, W2, W3 = SU2L.bosons().components
    B = U1Y.bosons().components[0]
    Z, A = sp.symbols("Z A", real=True)
    model.rotate(Rotation([W3, B], [Z, A], rotation_2x2(-sp.atan(gp / g))))
    Wp, Wm = sp.symbols("Wp Wm")
    model.rotate(Rotation([W1, W2], [Wp, Wm],
                          sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2),
                          kind="unitary"))
    return model, SU2L, dict(g=g, gp=gp, Z=Z, A=A, Wp=Wp, Wm=Wm)


def _quartic(vertices, legs):
    key = tuple(sorted(legs, key=sp.default_sort_key))
    for v in vertices:
        if v.particles == key:
            return v
    raise AssertionError(f"no quartic {key} in {[v.particles for v in vertices]}")


class TestAdjointRotation:
    def test_basis_is_the_four_physical_bosons(self, ew):
        model, SU2L, s = ew
        assert physical_vector_basis(model, [SU2L]) == \
            [s["A"], s["Wm"], s["Wp"], s["Z"]]

    def test_matches_the_hand_typed_matrix(self, ew):
        """The matrix this replaces, previously retyped at three call sites."""
        model, SU2L, s = ew
        g, gp = s["g"], s["gp"]
        cw, sw = g / sp.sqrt(g ** 2 + gp ** 2), gp / sp.sqrt(g ** 2 + gp ** 2)
        U, basis = adjoint_rotation(model, SU2L)
        order = [basis.index(s[n]) for n in ("Wp", "Wm", "Z", "A")]
        got = sp.Matrix([[sp.simplify(U[a, i]) for i in order]
                         for a in range(3)])
        expected = sp.Matrix([[1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],
                              [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],
                              [0, 0, cw, sw]])
        assert sp.simplify(got - expected) == sp.zeros(3, 4)

    def test_unbroken_group_gives_identity(self):
        """No rotation registered — every gluon is its own basis element."""
        gs = ExternalParameter("gs", 1.22, positive=True)
        SU3c = SU3("SU3c", coupling=gs)
        model = Model("QCD", gauge_groups=[SU3c], fields=[SU3c.bosons("G")],
                      parameters=[gs])
        U, basis = adjoint_rotation(model, SU3c)
        assert U == sp.eye(8)
        assert basis == sorted(SU3c.bosons().components,
                               key=sp.default_sort_key)

    def test_abelian_group_refused(self, ew):
        model, SU2L, s = ew
        U1Y = model.gauge_groups[1]
        with pytest.raises(ValueError, match="abelian"):
            adjoint_rotation(model, U1Y)

    def test_incomplete_basis_raises(self, ew):
        """A basis missing a physical boson must fail loudly, not silently
        drop the component (the chained Z-Z' trap)."""
        model, SU2L, s = ew
        with pytest.raises(ValueError, match="not spanned"):
            adjoint_rotation(model, SU2L, basis=[s["Wp"], s["Wm"], s["Z"]])


class TestElectroweakQuarticsVsMadGraph:
    """The four EW quartics, straight out of Model, in MG's leg ordering."""

    @pytest.fixture(scope="class")
    def vertices(self, ew):
        model, SU2L, s = ew
        return model.gauge_vertices(groups=[SU2L]), s

    def _pairs(self, vertices, legs):
        v = _quartic(vertices, legs)
        return [sp.simplify(c)
                for c in metric_pair_coefficients(v.structures)]

    def test_aaww(self, vertices):
        verts, s = vertices
        e = s["g"] * s["gp"] / sp.sqrt(s["g"] ** 2 + s["gp"] ** 2)
        c12, c13, c14 = self._pairs(verts, (s["A"], s["A"], s["Wm"], s["Wp"]))
        assert sp.simplify(c14 - sp.I * e ** 2) == 0
        assert sp.simplify(c13 - sp.I * e ** 2) == 0
        assert sp.simplify(c12 + 2 * sp.I * e ** 2) == 0

    def test_wwww(self, vertices):
        verts, s = vertices
        g = s["g"]
        c12, c13, c14 = self._pairs(verts,
                                    (s["Wm"], s["Wm"], s["Wp"], s["Wp"]))
        assert sp.simplify(c14 + sp.I * g ** 2) == 0
        assert sp.simplify(c13 + sp.I * g ** 2) == 0
        assert sp.simplify(c12 - 2 * sp.I * g ** 2) == 0

    def test_wwzz(self, vertices):
        verts, s = vertices
        g, gp = s["g"], s["gp"]
        val = sp.I * g ** 4 / (g ** 2 + gp ** 2)      # i g^2 cw^2
        c12, c13, c14 = self._pairs(verts, (s["Wm"], s["Wp"], s["Z"], s["Z"]))
        assert sp.simplify(c14 - val) == 0
        assert sp.simplify(c13 - val) == 0
        assert sp.simplify(c12 + 2 * val) == 0

    def test_awwz(self, vertices):
        """MG's VVVV5 shape — the only EW quartic whose three metric-pair
        coefficients are all different."""
        verts, s = vertices
        g, gp = s["g"], s["gp"]
        val = -2 * sp.I * g ** 3 * gp / (g ** 2 + gp ** 2)   # -2i g^2 cw sw
        c12, c13, c14 = self._pairs(verts,
                                    (s["A"], s["Wm"], s["Wp"], s["Z"]))
        assert sp.simplify(c14 - val) == 0
        assert sp.simplify(c13 + val / 2) == 0
        assert sp.simplify(c12 + val / 2) == 0

    def test_no_four_photon_and_no_four_z(self, vertices):
        """QED is abelian and the Z has no self-coupling — both must be
        absent, not merely small."""
        verts, s = vertices
        present = {v.particles for v in verts}
        assert tuple([s["A"]] * 4) not in present
        assert tuple([s["Z"]] * 4) not in present
        assert len(verts) == 4

    def test_coupling_slot_is_zero(self, vertices):
        """A VVVV vertex has no single scalar coupling — the slot must be
        zero so nothing downstream reads a plausible-looking wrong number."""
        verts, s = vertices
        for v in verts:
            assert v.coupling == 0
            assert set(v.structure_couplings) == set(v.structures)


def test_cubic_is_refused_not_guessed(ew):
    """The UFO sign convention for VVV is unresolved (unflipped for gluons,
    flipped for the EW vertices) — it must raise rather than guess."""
    model, SU2L, s = ew
    with pytest.raises(NotImplementedError, match="sign convention"):
        gauge_self_couplings(model, groups=[SU2L], include=("VVV", "VVVV"))


def test_gauge_vertices_cached_and_invalidated(ew):
    model, SU2L, s = ew
    first = model.gauge_vertices(groups=[SU2L])
    assert model.gauge_vertices(groups=[SU2L]) is first
    model._invalidate()
    assert model.gauge_vertices(groups=[SU2L]) is not first

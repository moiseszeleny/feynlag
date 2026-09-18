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
    ufo_leg_sign,
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
        verts = model.gauge_vertices(groups=[SU2L], include=("VVVV",))
        return verts, s

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


def test_unknown_vertex_type_rejected(ew):
    model, SU2L, s = ew
    with pytest.raises(ValueError, match="unknown vertex types"):
        gauge_self_couplings(model, groups=[SU2L], include=("VVV", "FFV"))


def test_gauge_vertices_cached_and_invalidated(ew):
    model, SU2L, s = ew
    first = model.gauge_vertices(groups=[SU2L], include=("VVVV",))
    assert model.gauge_vertices(groups=[SU2L], include=("VVVV",)) is first
    model._invalidate()
    assert model.gauge_vertices(groups=[SU2L], include=("VVVV",)) is not first


def test_hermiticity_check_refuses_multi_structure(ew):
    """A VVVV vertex's `coupling` slot is zero, so the scalar hermiticity
    comparison would pass vacuously — it must refuse instead."""
    from feynlag import check_hermiticity_pairing
    model, SU2L, s = ew
    verts = model.gauge_vertices(groups=[SU2L], include=("VVVV",))
    with pytest.raises(NotImplementedError, match="multi-structure"):
        check_hermiticity_pairing(bosonic_vertices=verts,
                                  conjugates={s["Wp"]: s["Wm"],
                                              s["Wm"]: s["Wp"]})


class TestUfoLegSign:
    """feynlag's symbols label FIELDS; a UFO leg labels a PARTICLE, and the
    field W+ carries the W- leg.  Emitting legs under naive names therefore
    transposes each conjugate pair, and VVV1 is totally antisymmetric.

    This is the entire content of what used to be documented as an
    unresolved "cubic sign convention": one conjugate pair -> -1 (the
    electroweak cubics), none -> +1 (the gluon).
    """

    def test_conjugate_pair_gives_minus_one(self, ew):
        model, SU2L, s = ew
        Wp, Wm, A = s["Wp"], s["Wm"], s["A"]
        conj = {Wp: Wm, Wm: Wp}
        assert ufo_leg_sign((A, Wm, Wp), conj) == -1

    def test_no_conjugate_pair_gives_plus_one(self):
        """ggg — why the gluon never needed a flip."""
        g1_, g2_, g3_ = sp.symbols("G_1 G_2 G_3")
        assert ufo_leg_sign((g1_, g2_, g3_), {}) == 1
        assert ufo_leg_sign((g1_, g2_, g3_), None) == 1

    def test_unpaired_charged_leg_raises(self, ew):
        """A charged leg whose partner is not also a leg means the
        relabelling is not a permutation — refuse rather than return a sign."""
        model, SU2L, s = ew
        Wp, Wm, A, Z = s["Wp"], s["Wm"], s["A"], s["Z"]
        with pytest.raises(ValueError, match="not.*a leg|permutation"):
            ufo_leg_sign((A, Z, Wp), {Wp: Wm, Wm: Wp})


class TestCubicInFeynlagConvention:
    """The electroweak cubics in **feynlag's own convention** — i.e. plain
    ``cubic_couplings`` values, `-i e` and `-i g cw` at the canonical leg
    ordering.

    These are NOT MadGraph's numbers, and that is deliberate: the
    field->particle leg sign a UFO needs is applied at export by the writer
    (``feynlag.export.ufo.legs``), the only layer that knows the
    particle/antiparticle pairing.  Do not "fix" these to MG's ``GC_4``/
    ``GC_53`` — those are pinned on the EXPORTED UFO, in
    ``tests/test_ufo_sm_bosonic.py`` and ``tests/test_ufo_export.py``.
    """

    @pytest.fixture(scope="class")
    def cubics(self, ew):
        model, SU2L, s = ew
        verts = model.gauge_vertices(groups=[SU2L])
        return {v.particles: v.coupling
                for v in verts if v.vertex_type == "VVV"}, s

    def test_aww(self, cubics):
        verts, s = cubics
        e = s["g"] * s["gp"] / sp.sqrt(s["g"] ** 2 + s["gp"] ** 2)
        got = verts[tuple(sorted((s["A"], s["Wm"], s["Wp"]),
                                 key=sp.default_sort_key))]
        assert sp.simplify(got + sp.I * e) == 0, got

    def test_zww(self, cubics):
        verts, s = cubics
        g, gp = s["g"], s["gp"]
        cw = g / sp.sqrt(g ** 2 + gp ** 2)
        got = verts[tuple(sorted((s["Wm"], s["Wp"], s["Z"]),
                                 key=sp.default_sort_key))]
        assert sp.simplify(got + sp.I * g * cw) == 0, got

    def test_is_the_raw_cubic_tensor(self, cubics, ew):
        """gauge_vertices applies no export convention of its own: the
        coupling is cubic_couplings' value at the canonical ordering."""
        from feynlag import cubic_couplings
        model, SU2L, s = ew
        verts, _ = cubics
        U, basis = adjoint_rotation(model, SU2L,
                                    basis=[s["Wp"], s["Wm"], s["Z"], s["A"]])
        raw = cubic_couplings(SU2L, physical=basis, U=U)
        key = tuple(sorted((s["A"], s["Wm"], s["Wp"]),
                           key=sp.default_sort_key))
        assert sp.simplify(verts[key] - raw[key]) == 0

    def test_unrotated_group_is_refused(self):
        """An unbroken group's "physical" basis IS its weak-basis adjoint
        components, so cubic_couplings/quartic_couplings return one
        colour-CARRYING value per component.  Emitting those as per-component
        vertices is the double-count trap, and the (G_1,G_2,G_3) triple would
        even look right because f^123 = 1 — so refuse, naming the helpers
        that do it properly.

        This test replaces one that asserted the opposite (that
        gauge_vertices returns the raw colour-carrying value); that was the
        footgun, not the contract.
        """
        gs = ExternalParameter("gs", 1.22, positive=True)
        SU3c = SU3("SU3c", coupling=gs)
        model = Model("QCD", gauge_groups=[SU3c], fields=[SU3c.bosons("G")],
                      parameters=[gs])
        for include in (("VVV",), ("VVVV",), ("VVV", "VVVV")):
            with pytest.raises(NotImplementedError, match="adjoint_vvv"):
                model.gauge_vertices(groups=[SU3c], include=include)

    def test_broken_group_still_works(self, ew):
        """The guard keys on "no rotation touches this group", not on colour,
        so the electroweak path is unaffected."""
        model, SU2L, s = ew
        verts = model.gauge_vertices(groups=[SU2L])
        assert {v.vertex_type for v in verts} == {"VVV", "VVVV"}

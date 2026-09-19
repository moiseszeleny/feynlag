"""Ground-truth verification of the Yang-Mills quartic (VVVV) self-coupling
assembly.

The cross-check here is independent of ``export.ufo.vvvv.assemble_vvvv``'s own
implementation: it differentiates ``-1/4 F^a_{mu nu} F^{a mu nu}`` directly (F
built straight from ``structure_constants``, no reliance on
``quartic_couplings`` at all) and reads off the vertex in the **metric-pair
basis** ``V = c12*M12M34 + c13*M13M24 + c14*M14M23`` via diagonal-metric-
component probing (picking (mu1..mu4) values that isolate exactly one pair
structure at a time, since eta is diagonal).

Why the metric-pair basis and not the VVVV1/2/3 catalog
-------------------------------------------------------
The catalog structures are linearly dependent (``VVVV2 = VVVV1 + VVVV3``), so
a vertex has a 1-parameter family of decompositions.  The previous version of
this file built its reference as ``{VVVV1: c14-c13, VVVV2: c14-c12,
VVVV3: c13-c12}`` -- the *same* over-complete convention ``assemble_vvvv``
used -- and compared coefficient lists.  That is circular: it cannot see a
common factor, and in fact both sides were 3x the true vertex for years.  The
metric-pair coefficients are convention-free, so ``TestVVVVReconstruction``
below (reconstruct ``sum(coeff * structure)`` and compare) is the check that
actually pins the normalization.

Covered: SU(3) (gluon, 4 distinct adjoint indices, including a case where all
three structures are simultaneously nonzero), SU(2) (W self-coupling, with
repeated adjoint indices), and the physical electroweak basis reached through
a complex W+- rotation, pinned against MadGraph's stock ``sm`` model.
"""

import itertools

import sympy as sp

from feynlag import SU2, SU3, quartic_couplings, structure_constants
from feynlag.export.ufo.vvvv import (assemble_vvvv, metric_pair_coefficients,
                                     permute_vvvv,
                                     structures_from_metric_pairs)


def _direct_metric_pairs(group, legs_idx, U=None):
    """Differentiate -1/4 F^a F^a directly and return ``(c12, c13, c14)``.

    Args:
        group: the gauge group.
        legs_idx: ``(i,j,k,l)`` — indices of the four legs (repeats allowed),
            into the adjoint components when ``U`` is None, else into the
            physical basis (the columns of ``U``).
        U: optional ``n_generators x M`` rotation with ``A^a = sum_i U[a,i] V_i``.
    """
    n = group.n_generators
    f = structure_constants(group)
    g = group.g
    eta = sp.diag(1, -1, -1, -1)

    if U is None:
        legs = [[sp.Symbol(f"A{a}_{mu}") for mu in range(4)] for a in range(n)]
        A = legs
    else:
        U = sp.Matrix(U)
        m = U.shape[1]
        legs = [[sp.Symbol(f"V{i}_{mu}") for mu in range(4)] for i in range(m)]
        A = [[sum(U[a, i] * legs[i][mu] for i in range(m)) for mu in range(4)]
             for a in range(n)]

    def Fint(a, mu, nu):
        s = sp.S.Zero
        for b in range(n):
            for c in range(n):
                val = f.get((a, b, c), sp.S.Zero)
                if val != 0:
                    s += g * val * A[b][mu] * A[c][nu]
        return s

    L4 = sp.S.Zero
    for a in range(n):
        for mu in range(4):
            for nu in range(4):
                L4 += (sp.Rational(-1, 4) * eta[mu, mu] * eta[nu, nu]
                       * Fint(a, mu, nu) * Fint(a, mu, nu))
    L4 = sp.expand(L4)

    def vertex(probe):
        expr = L4
        for (leg, mu) in probe:
            expr = sp.diff(expr, legs[leg][mu])
        return sp.expand(expr)

    i, j, k, l = legs_idx
    # diagonal-component probing: at these specific (mu1..mu4) choices the
    # OTHER two metric-pair structures vanish identically (off-diagonal eta),
    # isolating exactly one coefficient per probe.
    c12 = -vertex([(i, 0), (j, 0), (k, 1), (l, 1)])   # Metric(1,2)Metric(3,4)
    c13 = -vertex([(i, 0), (j, 1), (k, 0), (l, 1)])   # Metric(1,3)Metric(2,4)
    c14 = -vertex([(i, 0), (j, 1), (k, 1), (l, 0)])   # Metric(1,4)Metric(2,3)
    return sp.expand(c12), sp.expand(c13), sp.expand(c14)


def _assert_reconstructs(got, expected_pairs, label):
    """``got`` (a VVVV1/2/3 dict) must reconstruct ``expected_pairs``."""
    c12, c13, c14 = (sp.simplify(c) for c in metric_pair_coefficients(got))
    e12, e13, e14 = (sp.simplify(c) for c in expected_pairs)
    assert sp.simplify(c12 - e12) == 0, f"{label}: M12M34 {c12} != {e12}"
    assert sp.simplify(c13 - e13) == 0, f"{label}: M13M24 {c13} != {e13}"
    assert sp.simplify(c14 - e14) == 0, f"{label}: M14M23 {c14} != {e14}"


# --------------------------------------------------------------- weak basis

class TestVVVVReconstruction:
    """The non-circular check: sum(coeff * structure) == the differentiated
    vertex, in the convention-free metric-pair basis.  This is what pins the
    overall normalization (and what a coefficient-list comparison cannot)."""

    def test_su3_gluon_quartic_four_distinct_colors(self):
        g = sp.Symbol("g")
        SU3c = SU3("SU3c", coupling=g)
        comps = list(SU3c.bosons().components)
        qc = quartic_couplings(SU3c)
        for idx in [(0, 1, 3, 4), (0, 3, 5, 7)]:
            quad = tuple(comps[a] for a in idx)
            _assert_reconstructs(assemble_vvvv(qc, quad),
                                 _direct_metric_pairs(SU3c, idx), str(idx))

    def test_su2_w_self_coupling_repeated_indices(self):
        """Only 3 generators for 4 legs — necessarily repeated adjoint
        indices, the physically relevant WWWW-type case."""
        g = sp.Symbol("g")
        SU2L = SU2("SU2L", coupling=g)
        comps = list(SU2L.bosons().components)
        qc = quartic_couplings(SU2L)
        quad = (comps[0], comps[1], comps[0], comps[1])
        _assert_reconstructs(assemble_vvvv(qc, quad),
                             _direct_metric_pairs(SU2L, (0, 1, 0, 1)), "0101")

    def test_structures_are_linearly_dependent(self):
        """The reason the normalization needs pinning at all: the catalog is
        over-complete, so a coefficient-list comparison is not a check."""
        one = structures_from_metric_pairs(0, -1, 1)
        assert metric_pair_coefficients({"VVVV1": 1}) == (0, -1, 1)
        # VVVV2 = VVVV1 + VVVV3
        lhs = metric_pair_coefficients({"VVVV2": 1})
        rhs = metric_pair_coefficients({"VVVV1": 1, "VVVV3": 1})
        assert lhs == rhs
        assert one  # the symmetric representative of VVVV1 is nonempty

    def test_round_trip_metric_pairs(self):
        a, b = sp.symbols("a b")
        structures = structures_from_metric_pairs(a + b, -a, -b)
        assert metric_pair_coefficients(structures) == \
            (sp.expand(a + b), sp.expand(-a), sp.expand(-b))


# ------------------------------------------------------ physical (EW) basis

def _ew_setup():
    """SU(2)_L rotated to the physical (W+, W-, Z, A) basis."""
    g = sp.Symbol("g", positive=True)
    cw, sw = sp.symbols("cw sw", positive=True)
    SU2L = SU2("SU2L", coupling=g)
    Wp, Wm, Z, A = sp.symbols("Wp Wm Z A")
    U = sp.Matrix([[1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0],
                   [sp.I / sp.sqrt(2), -sp.I / sp.sqrt(2), 0, 0],
                   [0, 0, cw, sw]])
    basis = [Wp, Wm, Z, A]
    return SU2L, basis, U, dict(g=g, cw=cw, sw=sw,
                                Wp=Wp, Wm=Wm, Z=Z, A=A)


class TestPhysicalBasisQuartic:
    """The electroweak quartics WWWW / WWZZ / WWAA / WWAZ, in the physical
    basis reached through the COMPLEX W+- rotation."""

    def test_reconstructs_direct_differentiation(self):
        SU2L, basis, U, s = _ew_setup()
        qc = quartic_couplings(SU2L, physical=basis, U=U)
        for idx in [(0, 0, 1, 1), (0, 1, 2, 2), (0, 1, 3, 3), (0, 1, 2, 3)]:
            quad = tuple(basis[a] for a in idx)
            _assert_reconstructs(assemble_vvvv(qc, quad),
                                 _direct_metric_pairs(SU2L, idx, U=U),
                                 str(idx))

    def test_ordering_covariance(self):
        """No ordering is privileged and no multiplicity factor is needed:
        every permutation of a repeated-leg multiset reproduces the directly
        differentiated vertex for THAT ordering."""
        SU2L, basis, U, s = _ew_setup()
        qc = quartic_couplings(SU2L, physical=basis, U=U)
        for idx in set(itertools.permutations((0, 0, 1, 1))) | \
                set(itertools.permutations((0, 1, 2, 3))):
            quad = tuple(basis[a] for a in idx)
            _assert_reconstructs(assemble_vvvv(qc, quad),
                                 _direct_metric_pairs(SU2L, idx, U=U),
                                 str(idx))

    def test_permute_vvvv_matches_reassembly(self):
        """permute_vvvv relabels legs consistently with re-assembling at the
        permuted ordering (structures MIX under a permutation)."""
        SU2L, basis, U, s = _ew_setup()
        qc = quartic_couplings(SU2L, physical=basis, U=U)
        idx = (3, 0, 1, 2)                     # (A, Wp, Wm, Z)
        base = assemble_vvvv(qc, tuple(basis[a] for a in idx))
        for perm in [(0, 2, 1, 3), (1, 0, 3, 2), (0, 1, 3, 2)]:
            moved = tuple(idx[p] for p in perm)
            direct = assemble_vvvv(qc, tuple(basis[a] for a in moved))
            got = permute_vvvv(base, perm)
            _assert_reconstructs(
                got, metric_pair_coefficients(direct), str(perm))


class TestMadGraphOracle:
    """Pinned against MadGraph's stock ``sm`` model (models/sm/vertices.py +
    couplings.py), converted into the metric-pair basis.  MG's structures:

        VVVV2_MG = M14M23 + M13M24 - 2*M12M34
        VVVV5_MG = M14M23 - M13M24/2 - M12M34/2

    The UFO coupling carries the Feynman-rule ``i``, so these are compared
    against ``assemble_vvvv(..., feynman_rule=True)``.
    """

    def _pairs(self, quad_idx):
        SU2L, basis, U, s = _ew_setup()
        qc = quartic_couplings(SU2L, physical=basis, U=U)
        quad = tuple(basis[a] for a in quad_idx)
        got = assemble_vvvv(qc, quad, feynman_rule=True)
        return [sp.simplify(c) for c in metric_pair_coefficients(got)], s

    def test_aaww(self):
        """MG V_53 [a, a, W-, W+], VVVV2, GC_5 = i*ee**2."""
        (c12, c13, c14), s = self._pairs((3, 3, 1, 0))
        e = s["g"] * s["sw"]
        assert sp.simplify(c14 - sp.I * e ** 2) == 0
        assert sp.simplify(c13 - sp.I * e ** 2) == 0
        assert sp.simplify(c12 + 2 * sp.I * e ** 2) == 0

    def test_wwww(self):
        """MG V_55 [W-, W-, W+, W+], VVVV2, GC_35 = -i*ee**2/sw**2 = -i g^2."""
        (c12, c13, c14), s = self._pairs((1, 1, 0, 0))
        g = s["g"]
        assert sp.simplify(c14 + sp.I * g ** 2) == 0
        assert sp.simplify(c13 + sp.I * g ** 2) == 0
        assert sp.simplify(c12 - 2 * sp.I * g ** 2) == 0

    def test_wwzz(self):
        """MG V_70 [W-, W+, Z, Z], VVVV2, GC_36 = i*cw**2*ee**2/sw**2."""
        (c12, c13, c14), s = self._pairs((1, 0, 2, 2))
        val = sp.I * (s["cw"] * s["g"]) ** 2
        assert sp.simplify(c14 - val) == 0
        assert sp.simplify(c13 - val) == 0
        assert sp.simplify(c12 + 2 * val) == 0

    def test_awwz(self):
        """MG V_65 [a, W-, W+, Z], VVVV5, GC_57 = -2i*cw*ee**2/sw.

        The only EW quartic needing MG's two-structure VVVV5 shape — the
        sharpest test of the decomposition, since the three metric-pair
        coefficients are all different.
        """
        (c12, c13, c14), s = self._pairs((3, 1, 0, 2))
        val = -2 * sp.I * s["cw"] * s["sw"] * s["g"] ** 2
        assert sp.simplify(c14 - val) == 0
        assert sp.simplify(c13 + val / 2) == 0
        assert sp.simplify(c12 + val / 2) == 0

    def test_gluon_quartic_matches_GC_12(self):
        """MG V_37 [g,g,g,g] carries GC_12 = i*G**2 on every structure, with
        the colour tensor separate.  feynlag's raw output contains the colour
        contraction sum_e f_{ije} f_{kle}; stripped of it, the coupling must
        be i*g_s^2 on all three structures.
        """
        g = sp.Symbol("g", positive=True)
        SU3c = SU3("SU3c", coupling=g)
        comps = list(SU3c.bosons().components)
        f = structure_constants(SU3c)
        qc = quartic_couplings(SU3c)

        def colour(a, b, c, d):
            return sum(f.get((a, b, e), 0) * f.get((c, d, e), 0)
                       for e in range(8))

        for (i, j, k, l) in [(0, 1, 3, 4), (0, 3, 5, 7)]:
            quad = tuple(comps[a] for a in (i, j, k, l))
            got = assemble_vvvv(qc, quad, feynman_rule=True)
            cols = {"VVVV1": colour(i, j, k, l),
                    "VVVV2": colour(i, k, j, l),
                    "VVVV3": colour(i, l, j, k)}
            for name, coeff in got.items():
                assert cols[name] != 0, (name, i, j, k, l)
                assert sp.simplify(coeff / cols[name] - sp.I * g ** 2) == 0, \
                    (name, i, j, k, l)

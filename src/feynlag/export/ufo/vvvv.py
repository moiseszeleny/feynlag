"""Gauge self-coupling export helpers.

Assembles the 4-boson (VVVV) self-coupling into the 3 UFO Lorentz structures,
and supplies the colour-stripped couplings an UNBROKEN non-abelian group needs
(:func:`adjoint_vvv`, :func:`adjoint_vvvv`) — since ``cubic_couplings`` and
``quartic_couplings`` both return weak-basis values with the colour factor
already inside, which must not be multiplied by a colour tensor a second time.

For a representative external ordering ``(i, j, k, l)`` of 4 physical bosons,

    VVVV1 = -4 * quartic_couplings(...)[(i, j, k, l)]
    VVVV2 = -4 * quartic_couplings(...)[(i, k, j, l)]
    VVVV3 = -4 * quartic_couplings(...)[(i, l, j, k)]

matching lorentz_map.py's ``VVVV1/2/3`` definitions.

Why ``-4`` and not ``-12``
--------------------------
The three catalog structures are **linearly dependent**::

    VVVV1 = M14M23 - M13M24
    VVVV2 = M14M23 - M12M34      = VVVV1 + VVVV3
    VVVV3 = M13M24 - M12M34

so they span a 2-dimensional space and the decomposition of a vertex is a
1-parameter family.  Writing the vertex in the metric-pair basis as
``V = c12·M12M34 + c13·M13M24 + c14·M14M23``, the Yang-Mills quartic always
satisfies the traceless identity ``c12 + c13 + c14 = 0``, and the symmetric
representative is

    (a, b, c) = ((c14 - c13)/3, (c14 - c12)/3, (c13 - c12)/3)

which reconstructs exactly: ``a + b = c14``, ``-a + c = c13``,
``-b - c = c12``.  The ``-12`` this module shipped with omitted that ``/3``
and therefore reconstructed **3x** the true vertex.  The error was invisible
because tests/test_yangmills.py's "independent" ground truth built its
reference in the *same* over-complete convention and compared coefficient
lists, never reconstructing ``sum(coeff * structure)``.  Both are fixed
together; see ``TestVVVVReconstruction`` there.

Feynman-rule factor
-------------------
``assemble_vvvv`` returns raw Lagrangian-level coefficients by default (no
``i``), matching this module's original contract and the ground-truth tests.
Pass ``feynman_rule=True`` for the UFO-ready coupling ``i x coefficient`` --
that is what an exported UFO needs, and with it the SU(3) 4-gluon coupling
comes out ``i g_s^2``, matching MadGraph's stock ``sm`` ``GC_12``.
"""

import sympy as sp

__all__ = ["assemble_vvvv", "metric_pair_coefficients",
           "structures_from_metric_pairs", "permute_vvvv",
           "adjoint_vvv", "adjoint_vvvv",
           "ADJOINT_VVV_COLOR", "ADJOINT_VVVV_COLORS"]

#: UFO colour tensor pairing with VVV1 for an unbroken non-abelian group's
#: 3-boson self-coupling (ONE physical particle repeated three times).
ADJOINT_VVV_COLOR = "f(1,2,3)"

#: UFO colour tensors pairing with VVVV1/2/3 for an unbroken non-abelian
#: group's 4-boson self-coupling (ONE physical particle repeated four times).
ADJOINT_VVVV_COLORS = {
    "VVVV1": "f(1,2,-1)*f(3,4,-1)",
    "VVVV2": "f(1,3,-1)*f(2,4,-1)",
    "VVVV3": "f(1,4,-1)*f(2,3,-1)",
}

_K = -4

#: (c12, c13, c14) coefficients of each catalog structure in the metric-pair
#: basis ``V = c12*M12M34 + c13*M13M24 + c14*M14M23``.
_STRUCTURE_METRIC_PAIRS = {
    "VVVV1": (0, -1, 1),
    "VVVV2": (-1, 0, 1),
    "VVVV3": (-1, 1, 0),
}


def metric_pair_coefficients(structures):
    """``(c12, c13, c14)`` of a ``{VVVV1/2/3: coeff}`` dict.

    The inverse of :func:`structures_from_metric_pairs` (up to the
    1-parameter redundancy of the over-complete catalog).
    """
    c12 = c13 = c14 = sp.S.Zero
    for name, coeff in structures.items():
        d12, d13, d14 = _STRUCTURE_METRIC_PAIRS[name]
        c12 += d12 * coeff
        c13 += d13 * coeff
        c14 += d14 * coeff
    return sp.expand(c12), sp.expand(c13), sp.expand(c14)


def structures_from_metric_pairs(c12, c13, c14, simplifier=sp.simplify):
    """``{VVVV1/2/3: coeff}`` for ``c12*M12M34 + c13*M13M24 + c14*M14M23``.

    Raises:
        ValueError: if ``c12 + c13 + c14 != 0``, i.e. the vertex has a piece
            outside the span of the three catalog structures.
    """
    trace = simplifier(c12 + c13 + c14)
    if trace != 0:
        raise ValueError(
            f"vertex is outside the span of VVVV1/2/3: c12+c13+c14 = {trace} "
            f"(the three structures are linearly dependent and traceless)")
    raw = {
        "VVVV1": simplifier((c14 - c13) / 3),
        "VVVV2": simplifier((c14 - c12) / 3),
        "VVVV3": simplifier((c13 - c12) / 3),
    }
    return {name: coeff for name, coeff in raw.items() if coeff != 0}


def permute_vvvv(structures, perm, simplifier=sp.simplify):
    """Structures after relabelling the external legs by ``perm``.

    Args:
        structures: ``{VVVV1/2/3: coeff}`` for the current leg order.
        perm: a 4-tuple; leg ``n`` of the result is leg ``perm[n]`` of the
            input (0-based).

    A permutation mixes the catalog structures (swapping legs 2 and 3 sends
    ``VVVV1 <-> VVVV2`` and flips ``VVVV3``), so callers that need a partner
    ordering -- the hermiticity check, or matching MadGraph's leg order --
    must go through here rather than reusing the coefficients verbatim.
    """
    c12, c13, c14 = metric_pair_coefficients(structures)
    # metric pairs indexed by the partition of {0,1,2,3} into two pairs
    pairs = {frozenset([frozenset([0, 1]), frozenset([2, 3])]): c12,
             frozenset([frozenset([0, 2]), frozenset([1, 3])]): c13,
             frozenset([frozenset([0, 3]), frozenset([1, 2])]): c14}
    out = {}
    for partition, coeff in pairs.items():
        moved = frozenset(frozenset(perm[n] for n in pair)
                          for pair in partition)
        out[moved] = out.get(moved, sp.S.Zero) + coeff
    key12 = frozenset([frozenset([0, 1]), frozenset([2, 3])])
    key13 = frozenset([frozenset([0, 2]), frozenset([1, 3])])
    key14 = frozenset([frozenset([0, 3]), frozenset([1, 2])])
    return structures_from_metric_pairs(
        out.get(key12, sp.S.Zero), out.get(key13, sp.S.Zero),
        out.get(key14, sp.S.Zero), simplifier=simplifier)


def adjoint_vvv(group):
    """Colour-stripped VVV coupling of an UNBROKEN non-abelian group.

    The cubic twin of :func:`adjoint_vvvv`.  An unbroken group's 3-boson
    vertex is ONE UFO particle repeated three times, with the adjoint index
    carried by :data:`ADJOINT_VVV_COLOR`, so the coupling that pairs with it
    must be colour-*stripped*.

    ``cubic_couplings`` works in the weak basis, where its entry for a
    component triple is ``-g * f^{abc}`` — it already contains the colour
    factor.  Feeding those component-specific values to the writer alongside
    a colour tensor multiplies by colour twice.  Divided out, the coupling is
    ``-g`` on every triple, independent of N (verified for SU(2), SU(3) and
    SU(4) against every non-zero structure constant) — matching MadGraph's
    stock ``sm`` ``GC_10 = -G`` for ``[g,g,g]``.

    That bug hid for a long time because the only exported triple was
    ``(G_1,G_2,G_3)``, and ``f^{123} = 1`` makes the colour factor unity, so
    the emitted number was accidentally right.

    **No ``feynman_rule`` flag, unlike** :func:`adjoint_vvvv`.  The cubic
    carries one derivative, so the ``i`` from ``d_mu -> i p_mu`` cancels the
    Feynman-rule ``i`` and ``cubic_couplings``' output *is already* the vertex
    coefficient — real for a real basis, which is why ``ggg = -g_s`` has no
    ``i`` while ``gggg = i g_s^2`` does.  (Confirmed by differentiating
    ``-1/4 F F`` directly, including the Feynman ``i``: it reproduces
    ``cubic_couplings`` exactly.)  The quartic has no derivative, so its ``i``
    survives and has to be added.

    Returns:
        the single VVV1 coupling, to be emitted with
        :data:`ADJOINT_VVV_COLOR`.
    """
    if group.abelian:
        raise ValueError(f"{group!r} is abelian — no self-coupling")
    return -group.g


def adjoint_vvvv(group, feynman_rule=True):
    """Colour-stripped VVVV coupling of an UNBROKEN non-abelian group.

    An unbroken group's 4-boson vertex is ONE UFO particle repeated four
    times, with the adjoint index carried by the colour tensors
    :data:`ADJOINT_VVVV_COLORS`.  The coupling that pairs with them must
    therefore be colour-*stripped*.

    ``quartic_couplings`` works in the weak basis, where its entry for a
    component quadruple is ``-g^2/4 * sum_e f_{ije} f_{kle}`` — it already
    contains the colour contraction.  Feeding those component-specific values
    to the writer alongside a colour tensor multiplies by colour twice, which
    is what ``tests/test_ufo_qcd.py`` did before this helper existed.  Divide
    it out and the result is ``i g^2`` on every structure, independent of N —
    matching MadGraph's stock ``sm`` ``GC_12 = i*G**2`` for ``[g,g,g,g]``
    (pinned in ``tests/test_yangmills.py::TestMadGraphOracle``).

    Returns:
        ``{VVVV1/2/3: coupling}``, to be emitted with
        :data:`ADJOINT_VVVV_COLORS`.
    """
    if group.abelian:
        raise ValueError(f"{group!r} is abelian — no self-coupling")
    coeff = _K * sp.Rational(-1, 4) * group.g ** 2
    if feynman_rule:
        coeff *= sp.I
    return {name: coeff for name in ADJOINT_VVVV_COLORS}


def assemble_vvvv(quartic_raw, quadruple, feynman_rule=False):
    """Build ``{lorentz_name: coeff}`` (VVVV1/2/3) for one boson quadruple.

    Args:
        quartic_raw: the raw dict returned by
            :func:`~feynlag.vertices.yangmills.quartic_couplings`.
        quadruple: ``(i, j, k, l)`` — one representative ordering of the 4
            physical boson symbols.  The result is ordering-covariant: a
            different ordering of the same multiset gives the structures of
            that ordering (see :func:`permute_vvvv`), not a different vertex.
        feynman_rule: multiply by the Feynman-rule ``i``, giving the UFO
            coupling.  Default ``False`` returns the raw Lagrangian-level
            coefficient.

    Returns:
        dict with only the nonzero VVVV1/2/3 entries.
    """
    i, j, k, l = quadruple
    factor = _K * sp.I if feynman_rule else _K
    raw = {
        "VVVV1": factor * quartic_raw.get((i, j, k, l), 0),
        "VVVV2": factor * quartic_raw.get((i, k, j, l), 0),
        "VVVV3": factor * quartic_raw.get((i, l, j, k), 0),
    }
    return {name: coeff for name, coeff in raw.items() if coeff != 0}

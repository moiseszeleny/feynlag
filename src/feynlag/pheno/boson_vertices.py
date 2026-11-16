"""Bosonic three-point vertices as Lorentz tensors (Tier 4, stages 4b–4c).

A boson emitted by the fermion line of a two-fermion + two-boson process can
meet the two external bosons at a bosonic three-point vertex — the γ/Z
s channel of ``e⁺e⁻→W⁺W⁻``, the Z of ``e⁺e⁻→Zh``.  feynlag's vertex objects
carry a *scalar* coupling and name their Lorentz structure only by catalog
type; this module writes the structures out, in the loose-tensor format of
:func:`~feynlag.pheno.contraction.contract_trace`
(``[(coefficient, tensors)]``).

Every leg is described from the vertex's point of view: its Lorentz index
name (``None`` for a scalar) and the momentum flowing **into** the vertex
along it, as a combo ``((c, head), …)``.

``VVV``.  :meth:`~feynlag.lagrangian.Model.gauge_vertices` returns
``c = −g F_{ijk}`` (:func:`~feynlag.vertices.yangmills.cubic_couplings`) for
one canonical ordering of the field legs.  Differentiating
``−¼F^a_{μν}F^{aμν}`` with ``D_μ = ∂_μ − i g T^a A^a_μ`` and incoming plane
waves ``e^{−ip·x}`` gives the vertex

``i δ³L = g F_{ijk} [g^{μν}(p₁−p₂)^ρ + g^{νρ}(p₂−p₃)^μ + g^{ρμ}(p₃−p₁)^ν]``,

i.e. :data:`VVV_KAPPA` ``= −1`` times that coupling.  The sign is derived
(``tests/test_ww.py`` redoes the differentiation and pins it for ``AWW`` and
``ZWW``), and the high-energy cancellation in ``e⁺e⁻→W⁺W⁻`` is the physical
proof.  It is **not** the field→particle relabelling sign of
:mod:`feynlag.export.ufo.legs`, which is a statement about UFO leg names:
here a leg is attached by the field symbol it carries, with its own incoming
momentum, so no relabelling happens.

``VSS``.  The extractor's coupling carries ``p(φ)`` momentum tags, written
with ``∂_μφ → +i p(φ) φ``; :func:`vss_tensor` resolves each to
:data:`VSS_KAPPA` ``= −1`` times the momentum flowing into the vertex on
field leg ``φ`` — the same plane-wave sign as above (``tests/test_vss.py``
evaluates the kinetic term on plane waves directly, and the Ward identity of
``e⁻γ→χφ⁻`` is the physical proof).
"""

import sympy as sp

from ..operators import momentum

__all__ = ["VSS_KAPPA", "VVV_KAPPA", "sss_tensor", "vss_tensor",
           "vvs_tensor", "vvv_tensor"]

#: vertex = ``VVV_KAPPA × Vertex.coupling × [g^{μν}(p₁−p₂)^ρ + cyclic]``
VVV_KAPPA = -sp.S.One

#: a ``VSS`` coupling's tag ``p(φ)`` stands for ``VSS_KAPPA ×`` the momentum
#: flowing into the vertex with field leg ``φ``
VSS_KAPPA = -sp.S.One


def _difference(p, q):
    """The combo ``p − q``."""
    return tuple(p) + tuple((-c, head) for c, head in q)


def vvv_tensor(coupling, legs):
    """The triple-vector vertex.

    Args:
        coupling: the ``VVV`` :class:`~feynlag.vertices.vertex.Vertex`
            coupling.
        legs: three ``(index, incoming-momentum combo)`` pairs, **in the
            order of the vertex's** ``particles``.
    """
    (i1, p1), (i2, p2), (i3, p3) = legs
    c = VVV_KAPPA * sp.sympify(coupling)
    return [
        (c, (("g", i1, i2), ("v", _difference(p1, p2), i3))),
        (c, (("g", i2, i3), ("v", _difference(p2, p3), i1))),
        (c, (("g", i3, i1), ("v", _difference(p3, p1), i2))),
    ]


def vvs_tensor(coupling, index_a, index_b):
    """``coupling × g^{μν}`` on the two vector legs (the scalar carries no
    index and no momentum dependence)."""
    return [(sp.sympify(coupling), (("g", index_a, index_b),))]


def sss_tensor(coupling):
    """A scalar contact vertex: the coupling alone."""
    return [(sp.sympify(coupling), ())]


def vss_tensor(coupling, index, momenta):
    """The vector–scalar–scalar vertex, a derivative coupling.

    The extractor's ``VSS`` coupling is linear in the momentum tags of its
    two scalar legs, ``Σ_φ c_φ·p(φ)`` (each ``c_φ`` carrying the
    Feynman-rule ``i``), written with feynlag's ``∂_μφ → +i p(φ) φ``.  A
    physical particle entering the vertex on field leg ``φ`` with momentum
    ``k`` is a plane wave ``e^{−ik·x}``, so ``∂_μ → −ik_μ`` and each tag
    resolves to ``p(φ) → VSS_KAPPA · k^μ = −k^μ``, on the vector leg's index.

    Args:
        coupling: the ``VSS`` :class:`~feynlag.vertices.vertex.Vertex`
            coupling, with its ``p(φ)`` tags.
        index: the vector leg's index name.
        momenta: ``{scalar field symbol: incoming-momentum combo}`` for the
            two scalar legs.

    Raises:
        NotImplementedError: both scalar legs carry the same field symbol
            (their tags cannot be told apart), or the coupling is not
            exactly linear in the two legs' tags.
    """
    coupling = sp.expand(sp.sympify(coupling))
    if len(momenta) != 2:
        raise NotImplementedError(
            "vss_tensor: the two scalar legs carry the same field symbol, "
            "so their momentum tags p(φ) cannot be assigned to legs")
    pieces, rest = [], coupling
    for symbol, combo in momenta.items():
        tag = momentum(symbol)
        coeff = coupling.coeff(tag, 1)
        rest -= coeff * tag
        if coeff != 0:
            pieces.append((VSS_KAPPA * coeff, (("v", tuple(combo), index),)))
    rest = sp.expand(rest)
    if rest != 0 or any(piece[0].has(momentum) for piece in pieces):
        raise NotImplementedError(
            f"vss_tensor: the coupling {coupling} is not linear in the "
            f"momentum tags of its scalar legs {list(momenta)}")
    return pieces

"""Bosonic three-point vertices as Lorentz tensors (Tier 4, stage 4b).

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
"""

import sympy as sp

__all__ = ["VVV_KAPPA", "sss_tensor", "vvs_tensor", "vvv_tensor"]

#: vertex = ``VVV_KAPPA × Vertex.coupling × [g^{μν}(p₁−p₂)^ρ + cyclic]``
VVV_KAPPA = -sp.S.One


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

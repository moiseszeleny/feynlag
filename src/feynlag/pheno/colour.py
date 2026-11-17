"""Colour factors of 2→2 diagrams (Tier 5, stage 5a).

Colour is **factorised, not enumerated**.  A diagram with coloured external
legs is a colour tensor ``K_d`` over the external colour indices times the
Lorentz amplitude ``L_d`` the rest of :mod:`feynlag.pheno` computes with a
colour-stripped coupling, so the colour-summed

``Σ_colours |Σ_d K_d L_d|² = Σ_{d,d'} C_{dd'} L_d L̄_{d'}``,
``C_{dd'} = Σ_colours K_d K̄_{d'}``

needs only the **colour matrix** ``C``.  The tensors are built from the gauge
group's own generators (:meth:`~feynlag.groups.gauge.SUN.generators`), so no
Casimir is typed in: ``C_F``, the ``−2/3`` of the s×t interference and the
vanishing of a colour-singlet × colour-octet s channel all come out of the
index sums, for any SU(N) and any irrep.

A four-fermion diagram has two fermion lines, each a ``(bar, field)`` pair of
external positions.  A line's colour structure is ``T^a[x, y]`` (``x`` the
bar-slot leg's index, ``y`` the field-slot leg's) when it emits the gauge
boson of the colour group, ``δ[x, y]`` when it emits a colour singlet.
"""

import itertools

import sympy as sp

__all__ = ["colour_matrix", "four_fermion_colour_tensor",
           "line_colour_tensor"]


def four_fermion_colour_tensor(lines, dims, generators=None):
    """``K_d`` of a two-line diagram, as ``{(c₀, c₁, c₂, c₃): value}``.

    Args:
        lines: ``((bar_a, field_a), (bar_b, field_b))`` — external positions
            ``0..3`` of the two fermion lines.
        dims: the colour dimension of each of the four externals (``1`` for
            a colour singlet).
        generators: ``None`` for a colour-singlet mediator; for the colour
            group's gauge boson, a pair ``(T_a, T_b)`` of generator lists
            (one matrix per adjoint index) in the reps of line a and line b.

    Raises:
        ValueError: a colour-singlet mediator on a line whose two ends have
            different colour dimensions (a quark–lepton line), or generators
            whose size does not match the line's ends.
    """
    (bar_a, field_a), (bar_b, field_b) = lines
    for bar, field in lines:
        if dims[bar] != dims[field]:
            raise ValueError(
                f"colour: the fermion line between externals {bar} and "
                f"{field} joins colour dimensions {dims[bar]} and "
                f"{dims[field]}; a coloured mediator (leptoquark-like) is "
                f"not supported")
    if generators is not None:
        for (bar, _), T in zip(lines, generators):
            if T[0].shape != (dims[bar], dims[bar]):
                raise ValueError(
                    f"colour: generators of shape {T[0].shape} on a line of "
                    f"colour dimension {dims[bar]}")
    tensor = {}
    for colours in itertools.product(*(range(d) for d in dims)):
        xa, ya = colours[bar_a], colours[field_a]
        xb, yb = colours[bar_b], colours[field_b]
        if generators is None:
            value = sp.Integer(int(xa == ya and xb == yb))
        else:
            T_a, T_b = generators
            value = sum(Ta[xa, ya] * Tb[xb, yb] for Ta, Tb in zip(T_a, T_b))
        if value != 0:
            tensor[colours] = sp.nsimplify(value)
    return tensor


def line_colour_tensor(kind, dims, order=None, generators=None,
                       structure=None):
    """``K_d`` of a one-line diagram with two external bosons, as
    ``{(x, y, a₁, a₂): value}`` — bar-slot colour, field-slot colour and the
    colour index of each external boson (in the order of ``dims[2:]``).

    Args:
        kind: ``'fermion'`` (both bosons on the line), ``'gluon'`` (the line
            emits the colour group's gauge boson, which meets the two
            external bosons at the ``ggg`` vertex) or ``'singlet'`` (a
            colour-singlet mediator).
        dims: ``(d_bar, d_field, d_boson1, d_boson2)``; a colour singlet has
            dimension ``1``.
        order: for ``'fermion'``, ``(first, second)`` — which boson (``0`` or
            ``1``) sits at the vertex next to the bar slot and which next to
            the field slot; the tensor is ``(M_first · M_second)[x, y]`` with
            ``M = T^a`` for a coloured boson and ``1`` for a singlet.
        generators: the line's generator list ``T^a`` (needed whenever a
            coloured boson couples to it).
        structure: for ``'gluon'``, the structure constants as
            ``{(a, b, c): f^{abc}}``.  The tensor is
            ``Σ_c T^c[x, y] f^{c a₁ a₂}`` — mediator index first, then the
            external bosons in order, the leg order the Lorentz tensor of
            the three-boson vertex is built with.
    """
    d_bar, d_field, d1, d2 = dims
    if d_bar != d_field:
        raise ValueError(
            f"colour: the fermion line joins colour dimensions {d_bar} and "
            f"{d_field}; a coloured mediator or flavour-changing coloured "
            f"vertex of that kind is not supported")
    identity = sp.eye(d_bar)

    def matrix(boson, index):
        if (d1, d2)[boson] == 1:
            return identity
        return generators[index]

    tensor = {}
    for a1 in range(d1):
        for a2 in range(d2):
            adjoint = (a1, a2)
            if kind == "fermion":
                first, second = order
                block = matrix(first, adjoint[first]) * matrix(second, adjoint[second])
            elif kind == "singlet":
                if d1 != 1 or d2 != 1:
                    raise ValueError(
                        "colour: a colour-singlet mediator cannot end on "
                        "coloured external bosons")
                block = identity
            elif kind == "gluon":
                block = sp.zeros(d_bar, d_bar)
                for c, Tc in enumerate(generators):
                    f = structure.get((c, a1, a2), 0)
                    if f != 0:
                        block += f * Tc
            else:
                raise ValueError(f"colour: unknown line diagram kind {kind!r}")
            for x in range(d_bar):
                for y in range(d_field):
                    if block[x, y] != 0:
                        tensor[(x, y, a1, a2)] = sp.nsimplify(block[x, y])
    return tensor


def colour_matrix(tensors):
    """``C[d][d'] = Σ_colours K_d · conj(K_{d'})`` for a list of tensors."""
    n = len(tensors)
    matrix = [[sp.S.Zero] * n for _ in range(n)]
    for i, j in itertools.product(range(n), repeat=2):
        small, large = sorted((tensors[i], tensors[j]), key=len)
        common = [key for key in small if key in large]
        matrix[i][j] = sp.simplify(sum(
            tensors[i][key] * sp.conjugate(tensors[j][key]) for key in common))
    return matrix

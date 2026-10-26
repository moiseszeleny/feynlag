"""Weak → physical basis for gauge bosons, derived from a Model's rotations.

The Yang-Mills self-couplings are built group-theoretically from structure
constants (``vertices/yangmills.py``), never from a ``-1/4 F F`` Lagrangian
term, so they bypass ``Model.physical_lagrangian`` entirely.  To get them in
the *physical* basis, ``cubic_couplings``/``quartic_couplings`` take a matrix
``U`` with ``A^a = sum_i U[a,i] V_i`` and rotate the tensor instead of the
Lagrangian.

That ``U`` used to be hand-typed at every call site (three copies of the same
electroweak matrix), which made a physical-basis quartic something a user had
to derive themselves and kept the Yang-Mills track out of ``Model``.
:func:`adjoint_rotation` derives it from the ``Rotation`` objects already
registered on the model, applying them **sequentially in registration order**
exactly as :meth:`Model.physical_lagrangian` does — so the two tracks cannot
drift apart, and chained rotations (a Weinberg rotation feeding a Z-Z' one)
compose for free.
"""

import itertools

import sympy as sp

from .export.ufo.vvvv import assemble_vvvv
from .vertices.vertex import Vertex
from .vertices.yangmills import cubic_couplings, quartic_couplings

__all__ = ["physical_vector_basis", "adjoint_rotation",
           "gauge_self_couplings"]


def _rotate_symbol(model, sym):
    """Image of a weak-basis symbol after every registered rotation."""
    expr = sym
    for rot in model.rotations:
        expr = expr.xreplace(rot.substitution())
    return sp.expand(expr)


def physical_vector_basis(model, groups=None):
    """Ordered physical vector symbols reachable from the gauge groups'
    adjoint components after all registered rotations.

    Deduplicated and sorted by ``sp.default_sort_key`` — which, for the
    electroweak case, happens to give ``(A, Wm, Wp, Z)``, matching
    MadGraph's own leg ordering.
    """
    groups = list(model.gauge_groups if groups is None else groups)
    spins = model.spin_map()
    seen = []
    for group in groups:
        if group.abelian:
            continue
        for comp in group.bosons().components:
            for sym in _rotate_symbol(model, comp).free_symbols:
                if sym in seen:
                    continue
                # a rotation's coefficients bring in couplings/angles, so
                # keep ONLY symbols the model knows to be spin-1 fields
                if spins.get(sym) == 1:
                    seen.append(sym)
    return sorted(seen, key=sp.default_sort_key)


def adjoint_rotation(model, group, basis=None, simplifier=sp.expand):
    """``(U, basis)`` with ``A^a = sum_i U[a,i] V_i`` for one gauge group.

    ``U`` has shape ``(group.n_generators, len(basis))``.  Components no
    rotation touches give a unit row, so an unbroken group (gluons) yields
    the identity and a partially-rotated one is handled per component.

    Raises:
        ValueError: a component's image contains a vector symbol outside
            ``basis`` — e.g. an intermediate field from a chained rotation
            when ``basis`` was passed in incomplete.
    """
    if group.abelian:
        raise ValueError(f"{group!r} is abelian — it has no adjoint to rotate")
    if basis is None:
        basis = physical_vector_basis(model, [group])
    basis = list(basis)
    comps = list(group.bosons().components)
    U = sp.zeros(len(comps), len(basis))
    for a, comp in enumerate(comps):
        image = simplifier(_rotate_symbol(model, comp))
        residual = image
        for i, sym in enumerate(basis):
            coeff = image.coeff(sym)
            U[a, i] = coeff
            residual = residual - coeff * sym
        residual = sp.simplify(sp.expand(residual))
        if residual != 0:
            raise ValueError(
                f"{comp} rotates to {image}, which is not spanned by the "
                f"physical basis {basis} (leftover {residual}) — pass the "
                f"full basis, including any intermediate field of a chained "
                f"rotation")
    return U, basis


def _reject_unrotated(model, group):
    """Refuse a group whose adjoint components no rotation touches.

    For such a group the "physical" basis IS its weak-basis adjoint
    components, so ``cubic_couplings``/``quartic_couplings`` return one value
    per component carrying the colour factor (``-g f^{abc}``,
    ``-g^2/4 sum_e f f``).  Emitting those as per-component vertices is the
    trap ``export/ufo/writer.py`` and ``docs/manual/export.md`` warn about: an
    unbroken group is ONE UFO particle repeated, with the adjoint index in a
    colour tensor, so the coupling must be colour-STRIPPED.

    Returning them from here would hand a caller ``Vertex`` objects that look
    export-ready and are not — and the ``(G_1,G_2,G_3)`` triple would even
    look correct, because ``f^{123} = 1``.  Refuse instead, and name the two
    helpers that do it properly.
    """
    for comp in group.bosons().components:
        if _rotate_symbol(model, comp) != comp:
            return                      # something mixed it: physical basis
    raise NotImplementedError(
        f"no registered rotation touches {group.name}'s adjoint components, "
        f"so its self-couplings are weak-basis values carrying the colour "
        f"factor. An unbroken group exports as ONE particle repeated with a "
        f"colour tensor — use export.ufo.vvvv.adjoint_vvv / adjoint_vvvv "
        f"(with ADJOINT_VVV_COLOR / ADJOINT_VVVV_COLORS), or call "
        f"cubic_couplings / quartic_couplings directly for the weak-basis "
        f"tensor (internal verification only)")


def gauge_self_couplings(model, groups=None, basis=None,
                         simplifier=sp.simplify, include=("VVV", "VVVV")):
    """VVV and VVVV :class:`Vertex` objects for the gauge self-couplings.

    These are derived group-theoretically and therefore do **not** duplicate
    :meth:`Model.vertices` output — the ``-1/4 F F`` term is never written
    into the Lagrangian, so the extractor never sees these vertices.

    One canonical representative ordering per leg multiset (sorted by
    ``sp.default_sort_key``, which for the electroweak case reproduces
    MadGraph's own leg ordering); a VVVV vertex carries its three
    per-structure couplings in ``meta['structures']`` (see
    :meth:`Vertex.from_structures`), since a single scalar cannot express
    them.  The couplings carry the Feynman-rule ``i`` and are pinned against
    MadGraph's stock ``sm`` in ``tests/test_yangmills.py::TestMadGraphOracle``.

    Couplings are in **feynlag's own convention**, like every other
    :class:`~feynlag.vertices.vertex.Vertex`.  The field->particle leg sign a
    UFO needs (which flips an electroweak cubic but not ``ggg``) is applied at
    export, by the writer — the only layer that knows the particle/antiparticle
    pairing.  See :mod:`feynlag.export.ufo.legs`.

    Args:
        include: which vertex types to build.

    Returns:
        list of :class:`~feynlag.vertices.vertex.Vertex`.
    """
    unknown = set(include) - {"VVV", "VVVV"}
    if unknown:
        raise ValueError(f"unknown vertex types {sorted(unknown)}")
    groups = [g for g in (model.gauge_groups if groups is None else groups)
              if not g.abelian]
    if basis is None:
        basis = physical_vector_basis(model, groups)
    basis = list(basis)

    cubic_total, quartic_total = {}, {}
    for group in groups:
        _reject_unrotated(model, group)
        U, _ = adjoint_rotation(model, group, basis)
        if "VVV" in include:
            for key, val in cubic_couplings(group, physical=basis,
                                            U=U).items():
                cubic_total[key] = cubic_total.get(key, sp.S.Zero) + val
        if "VVVV" in include:
            for key, val in quartic_couplings(group, physical=basis,
                                              U=U).items():
                quartic_total[key] = quartic_total.get(key, sp.S.Zero) + val

    vertices = []
    for triple in itertools.combinations_with_replacement(basis, 3):
        ordering = tuple(sorted(triple, key=sp.default_sort_key))
        raw = cubic_total.get(ordering, sp.S.Zero)
        if raw == 0:
            continue
        coupling = simplifier(raw)
        if coupling != 0:
            vertices.append(Vertex(ordering, coupling, "VVV"))
    for quad in itertools.combinations_with_replacement(basis, 4):
        ordering = tuple(sorted(quad, key=sp.default_sort_key))
        structures = assemble_vvvv(quartic_total, ordering, feynman_rule=True)
        structures = {name: simplifier(c) for name, c in structures.items()}
        structures = {name: c for name, c in structures.items() if c != 0}
        if structures:
            vertices.append(Vertex.from_structures(ordering, structures))
    return vertices

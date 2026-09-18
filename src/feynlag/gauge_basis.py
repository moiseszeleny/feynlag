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
from .vertices.yangmills import quartic_couplings

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


def gauge_self_couplings(model, groups=None, basis=None,
                         simplifier=sp.simplify, include=("VVVV",)):
    """VVVV :class:`Vertex` objects for the gauge self-couplings.

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

    Args:
        include: which vertex types to build.  ``"VVV"`` is **not**
            supported: the UFO sign convention for the cubic is unresolved.
            ``cubic_couplings``'s raw output is exported *unflipped* for the
            gluon (``ggg = -g_s``, matching MG's ``GC_10``) but *flipped* for
            the electroweak vertices (validated by the ``e+e-→W+W-``
            round-trip at 19.50 pb, and MG's ``[a,W-,W+] = +i e`` against our
            ``-i e`` at the same ordering).  Until that asymmetry is derived
            rather than observed, this function refuses to emit a VVV vertex
            instead of guessing a sign — build it with ``cubic_couplings``
            and apply the convention at the call site, as
            ``scripts/export_sm_ufo.py`` does.

    Returns:
        list of :class:`~feynlag.vertices.vertex.Vertex`.
    """
    unknown = set(include) - {"VVVV"}
    if unknown:
        raise NotImplementedError(
            f"gauge_self_couplings cannot build {sorted(unknown)}: the UFO "
            f"sign convention for the cubic gauge coupling is unresolved "
            f"(unflipped for gluons, flipped for the electroweak vertices) "
            f"— see this function's docstring")
    groups = [g for g in (model.gauge_groups if groups is None else groups)
              if not g.abelian]
    if basis is None:
        basis = physical_vector_basis(model, groups)
    basis = list(basis)

    quartic_total = {}
    for group in groups:
        U, _ = adjoint_rotation(model, group, basis)
        for key, val in quartic_couplings(group, physical=basis, U=U).items():
            quartic_total[key] = quartic_total.get(key, sp.S.Zero) + val

    vertices = []
    for quad in itertools.combinations_with_replacement(basis, 4):
        ordering = tuple(sorted(quad, key=sp.default_sort_key))
        structures = assemble_vvvv(quartic_total, ordering, feynman_rule=True)
        structures = {name: simplifier(c) for name, c in structures.items()}
        structures = {name: c for name, c in structures.items() if c != 0}
        if structures:
            vertices.append(Vertex.from_structures(ordering, structures))
    return vertices

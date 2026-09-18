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

from .export.ufo.vvvv import (assemble_vvvv,
                              metric_pair_coefficients,
                              permute_vvvv)
from .vertices.vertex import Vertex
from .vertices.yangmills import cubic_couplings, quartic_couplings

__all__ = ["physical_vector_basis", "adjoint_rotation",
           "gauge_self_couplings", "ufo_leg_sign"]


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


def ufo_leg_sign(legs, conjugates=None):
    """Sign relating feynlag's field-symbol leg order to UFO's particle legs.

    feynlag's symbols label **fields**; a UFO leg labels a **particle**.  The
    field ``W+`` annihilates a W+ but *creates* a W−, so the leg carrying the
    symbol ``Wp`` is UFO's ``W-`` leg.  Emitting the legs under their naive
    names therefore transposes each conjugate pair, and a Lorentz structure
    antisymmetric under that transposition picks up its signature.

    This is the whole content of the "cubic sign convention" puzzle:

    - a VVV with one conjugate pair (``A W+ W-``, ``W+ W- Z``) gets **−1**,
      which is exactly the flip ``scripts/export_sm_ufo.py`` used to apply by
      hand and the ``e+e-→W+W-`` round-trip validated;
    - a VVV with no conjugate pair (``ggg``) gets **+1**, which is why the
      gluon never needed one.

    Derived in this session against MadGraph's stock ``sm``: with this sign
    and no other adjustment, feynlag's ``[a,W-,W+]``, ``[W-,W+,Z]``,
    ``[g,g,g]``, ``[a,G-,G+]`` and ``[Z,G-,G+]`` all reproduce ``GC_4``,
    ``GC_53``, ``GC_10``, ``GC_3`` and ``GC_61``.

    Args:
        legs: the field symbols, in the order they will be emitted.
        conjugates: ``{field: antifield}`` for the non-self-conjugate fields
            (both directions), as ``check_hermiticity_pairing`` takes.  Fields
            absent from it are self-conjugate.  ``None`` means every field is
            self-conjugate, i.e. sign ``+1``.

    Returns:
        ``+1`` or ``−1`` for a structure that is totally antisymmetric in the
        exchanged legs (VVV); callers of non-antisymmetric structures must
        check invariance instead (see :func:`gauge_self_couplings`).

    Raises:
        ValueError: a conjugate leg's partner is not also among the legs, so
            the relabelling is not a permutation of this vertex's legs.
    """
    conjugates = conjugates or {}
    legs = list(legs)
    target = [conjugates.get(L, L) for L in legs]
    remaining = list(range(len(legs)))
    perm = []
    for t in target:
        for i in remaining:
            if legs[i] == t:
                perm.append(i)
                remaining.remove(i)
                break
        else:
            raise ValueError(
                f"leg {t} (the antiparticle of a leg of {tuple(legs)}) is not "
                f"itself a leg, so the field->particle relabelling is not a "
                f"permutation of this vertex")
    parity = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                parity = -parity
    return parity


def _assert_relabelling_invariant(ordering, structures, conjugates):
    """A VVVV's structures must be INVARIANT under the field->particle
    relabelling, not merely pick up a sign.

    Unlike VVV1 the quartic structures are not totally antisymmetric, so
    there is no signature to apply — either the relabelling is a symmetry of
    this vertex's structures (it is, for all four electroweak quartics: the
    coefficients it would exchange are equal) or the vertex cannot be emitted
    under naive leg labels at all.  Checked rather than assumed.
    """
    conjugates = conjugates or {}
    if not any(L in conjugates for L in ordering):
        return
    target = [conjugates.get(L, L) for L in ordering]
    remaining = list(range(len(ordering)))
    perm = []
    for t in target:
        for i in remaining:
            if ordering[i] == t:
                perm.append(i)
                remaining.remove(i)
                break
        else:
            raise ValueError(
                f"{ordering}: the field->particle relabelling is not a "
                f"permutation of this vertex's legs")
    moved = permute_vvvv(structures, tuple(perm))
    if metric_pair_coefficients(moved) != metric_pair_coefficients(structures):
        raise NotImplementedError(
            f"quartic {ordering} is not invariant under the field->particle "
            f"relabelling {perm}; emitting it under naive leg labels would "
            f"be wrong and no signature can fix a non-antisymmetric structure")


def gauge_self_couplings(model, groups=None, basis=None,
                         simplifier=sp.simplify, include=("VVV", "VVVV"),
                         conjugates=None):
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

    The VVV couplings are UFO-ready: they carry the field->particle leg sign
    from :func:`ufo_leg_sign`, which is what makes an electroweak cubic come
    out flipped relative to ``cubic_couplings``'s raw tensor while the gluon
    does not.  That asymmetry used to be applied by hand in
    ``scripts/export_sm_ufo.py`` and described as an unresolved convention;
    it is neither — see :func:`ufo_leg_sign`.

    Args:
        include: which vertex types to build.
        conjugates: ``{field: antifield}`` for the non-self-conjugate physical
            bosons (e.g. ``{Wp: Wm, Wm: Wp}``), needed for the VVV leg sign.
            Omitted, every boson is treated as self-conjugate — correct for an
            unbroken group (gluons), wrong for W±, so passing it is required
            whenever a charged vector is in ``basis``.

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
        # VVV1 is totally antisymmetric, so the field->particle relabelling
        # contributes its permutation signature (see ufo_leg_sign).
        coupling = simplifier(ufo_leg_sign(ordering, conjugates) * raw)
        if coupling != 0:
            vertices.append(Vertex(ordering, coupling, "VVV"))
    for quad in itertools.combinations_with_replacement(basis, 4):
        ordering = tuple(sorted(quad, key=sp.default_sort_key))
        structures = assemble_vvvv(quartic_total, ordering, feynman_rule=True)
        structures = {name: simplifier(c) for name, c in structures.items()}
        structures = {name: c for name, c in structures.items() if c != 0}
        if structures:
            _assert_relabelling_invariant(ordering, structures, conjugates)
            vertices.append(Vertex.from_structures(ordering, structures))
    return vertices

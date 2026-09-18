"""The field -> particle leg relabelling, and what each Lorentz structure
costs under it.

feynlag's symbols label **fields**; a UFO leg labels a **particle**.  The field
``W+`` annihilates a W+ but *creates* a W-, so the leg carrying the symbol
``Wp`` is UFO's ``W-`` leg.  Emitting legs under their naive names therefore
transposes each conjugate pair, and a Lorentz structure antisymmetric under
that transposition picks up its signature.

This is the whole content of what was long recorded as an unresolved "cubic
sign convention":

- a VVV with one conjugate pair (``A W+ W-``, ``W+ W- Z``) gets **-1**, which
  is the flip ``scripts/export_sm_ufo.py`` used to apply by hand and the
  ``e+e- -> W+W-`` round-trip validated;
- a VVV with no conjugate pair (``ggg``) gets **+1**, which is why the gluon
  never needed one;
- a VSS whose two scalars are a conjugate pair (``A G+ G-``) also gets **-1**
  -- which nothing applied until this module existed, so every exported
  Feynman-gauge VSS was wrong by a sign;
- VVS/VVSS/SSS/SSSS get **+1**, which is why the Higgs couplings matched
  MadGraph all along and must keep doing so.

Derived against MadGraph's shipped model files: with this rule and no other
adjustment, feynlag reproduces ``GC_4`` ``[a,W-,W+]``, ``GC_53`` ``[W-,W+,Z]``,
``GC_10`` ``[g,g,g]``, ``GC_3`` ``[a,G-,G+]`` and ``GC_61`` ``[Z,G-,G+]``.

It lives in the export package, not next to the physics, because it is purely
a UFO convention: a :class:`~feynlag.vertices.vertex.Vertex`'s coupling is
always in feynlag's own convention, and the writer is the only layer that
knows the particle/antiparticle pairing (``UFOParticle.antisymbol``).
"""

__all__ = ["ufo_leg_sign", "structure_leg_sign", "SYMMETRIC_STRUCTURES"]

#: Lorentz structures invariant under a transposition of any two legs that can
#: form a conjugate pair -- ``Metric(1,2)`` is symmetric in the two vectors and
#: the scalars carry no index, and ``"1"`` has no index at all.
SYMMETRIC_STRUCTURES = frozenset({"VVS1", "VVSS1", "SSS1", "SSSS1"})


def ufo_leg_sign(legs, conjugates=None):
    """Permutation parity of the field -> particle relabelling of ``legs``.

    For a **totally antisymmetric** structure (VVV1) this is the sign
    directly; other structures go through :func:`structure_leg_sign`.

    Args:
        legs: the field symbols, in the order they will be emitted.
        conjugates: ``{field: antifield}`` for the non-self-conjugate fields
            (both directions).  Fields absent from it are self-conjugate;
            ``None`` means everything is, i.e. sign ``+1``.

    Raises:
        ValueError: a conjugate leg's partner is not also among the legs, so
            the relabelling is not a permutation of this vertex's legs.
    """
    conjugates = conjugates or {}
    legs = list(legs)
    target = [conjugates.get(leg, leg) for leg in legs]
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


def structure_leg_sign(structure, legs, conjugates=None):
    """Sign the emitted coupling picks up for one Lorentz structure.

    Args:
        structure: UFO structure name (``'VVV1'``, ``'VSS1'``, ...), as
            ``export.ufo.lorentz_map`` names them.
        legs: the field symbols in the order they will be emitted.
        conjugates: as :func:`ufo_leg_sign`.

    Raises:
        ValueError: the structure has no recorded behaviour.  Refusing beats
            silently assuming ``+1`` for a structure nobody has checked --
            that assumption is exactly what left VSS wrong.
    """
    if structure in SYMMETRIC_STRUCTURES:
        return 1
    if structure == "VVV1":
        # totally antisymmetric
        return ufo_leg_sign(legs, conjugates)
    if structure == "VSS1":
        # P(1,2) - P(1,3): antisymmetric in legs 2 and 3 only, and a vector
        # cannot pair with a scalar, so the only possible pair is those two.
        conjugates = conjugates or {}
        _, a, b = legs
        return -1 if conjugates.get(a) == b and a != b else 1
    raise ValueError(
        f"no leg-relabelling rule recorded for Lorentz structure "
        f"{structure!r}; add it to {__name__} rather than assuming +1 "
        f"(VVVV is handled by its own invariance check, and the fermion "
        f"structures have their own explicit bar/field leg convention)")

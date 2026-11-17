"""Model-derived calculator setup, shared by the decay and scattering
calculators' ``from_model`` constructors.

Everything a calculator needs that the :class:`~feynlag.lagrangian.Model`
already knows is read from it here: the unitary-gauge boson list, the
conjugate maps, the boson masses (derived from the physical Lagrangian by
:func:`~feynlag.vacuum.physical.physical_boson_masses`) and the broken
non-abelian groups.  A keyword the caller passes always wins.
"""

import sympy as sp

from ..parameters import ParameterPoint

__all__ = ["model_setup", "resolve_point"]


def resolve_point(model, point):
    """The ``parameters=`` value of a calculator: ``point`` (a
    :class:`ParameterPoint`, a ``{name: value}`` dict of external overrides,
    or ``None``) on top of ``model.parameters``."""
    if point is None:
        return model.parameters
    if isinstance(point, ParameterPoint):
        return point
    return ParameterPoint(model.parameters, point)


def model_setup(model, boson_fields=None, masses=None):
    """``(boson_fields, cmap, conjugates, masses)`` derived from ``model``.

    Args:
        boson_fields: override of the unitary-gauge boson list
            (default ``model.physical.unitary_bosons``).
        masses: ``{symbol: mass}`` entries that replace the derived ones
            (a typed number, a mass symbol, a fermion leg).

    Raises:
        ValueError: the model has no physical basis yet.
    """
    reg = model.physical
    if boson_fields is None:
        if not reg.bosons:
            raise ValueError(
                f"model {model.name!r} has no physical basis: register its "
                f"rotations first (models.to_physical_basis, or "
                f"Model.diagonalize on each mass block), or pass "
                f"boson_fields= and masses= explicitly")
        boson_fields = reg.unitary_bosons
    boson_fields = list(boson_fields)
    given = dict(masses or {})
    wanted = [b for b in boson_fields if b not in given]
    derived = {}
    if wanted:
        from ..vacuum.physical import physical_boson_masses
        for b, m2 in physical_boson_masses(model, wanted).items():
            derived[b] = sp.sqrt(m2)
    derived.update(given)
    return boson_fields, dict(reg.cmap), dict(reg.conjugates), derived

"""Global (ungauged) continuous symmetries: a U(1) with per-flavour charges.

A global U(1) has no gauge boson and enters no covariant derivative, so it is
neither a :class:`~feynlag.groups.base.GaugeGroup` (a coupling-less ``U1`` in
``reps`` would still be gauged by ``Dmu``/``fermion_gauge_current``) nor a
discrete group.  Unlike a :class:`~feynlag.groups.discrete.ZN`, whose
``assign`` gives a whole multiplet one charge, a fermion here may carry one
charge **per flavour** — the horizontal symmetries of Froggatt–Nielsen
models, where ``Q_L`` of the three generations carries e.g. ``(3, 2, 0)``.

The transformation is ``φ → e^{iqθ}φ`` (``φ* → e^{−iqθ}φ*``) for a scalar
and ``ψ_k → e^{iq_kθ}ψ_k``, ``ψ̄_k → e^{−iq_kθ}ψ̄_k`` for flavour ``k`` of a
fermion.  :func:`feynlag.invariance.check_global_invariance` tests it.
"""

import sympy as sp

from .base import SymmetryGroup

__all__ = ["GlobalU1"]


def _real_charge(q):
    q = sp.sympify(q)
    if q.is_real is not True:
        raise ValueError(f"U(1) charge {q} must be real (declare a symbolic "
                         "charge with real=True)")
    return q


class GlobalU1(SymmetryGroup):
    """A global U(1) acting by phases, with per-flavour fermion charges.

    Example (Froggatt–Nielsen)::

        FN = GlobalU1("U1_FN")
        FN.assign((3, 2, 0), QL)       # one charge per generation
        FN.assign((3, 1, 0), uR)
        FN.assign(1, phi)              # the flavon

    Fields not assigned are neutral.  Every gauge component of a field shares
    its charge; Dirac-adjoint (bar) legs and conjugated scalars carry minus it.
    """

    def __init__(self, name):
        super().__init__(name)
        #: ``{component Symbol: charge}`` for scalars
        self.scalar_charges = {}
        #: ``{IndexedBase: (charge per flavour)}`` for fermion legs (bars negated)
        self.fermion_charges = {}
        #: the declared (field, charge) pairs, in assignment order
        self.assignments = []

    def assign(self, charge, *fields):
        """Give ``fields`` the U(1) charge ``charge``.

        Args:
            charge: a real number or real symbol; for a fermion also a tuple
                with one charge per flavour (length ``nflavors``).
            fields: :class:`~feynlag.fields.Field`\\ s or bare component
                Symbols.

        Returns ``self`` so assignments chain.
        """
        from ..fields import Fermion

        for field in fields:
            if isinstance(field, Fermion):
                self._assign_fermion(charge, field)
            else:
                self._assign_scalar(charge, field)
            self.assignments.append((field, charge))
        return self

    def _assign_fermion(self, charge, field):
        if isinstance(charge, (tuple, list)):
            qs = tuple(_real_charge(q) for q in charge)
            if len(qs) != field.nflavors:
                raise ValueError(f"{field.name} has {field.nflavors} "
                                 f"flavour(s), got {len(qs)} charges")
        else:
            qs = (_real_charge(charge),) * field.nflavors
        for comp, bar in zip(field.components, field.bar_components):
            if comp in self.fermion_charges:
                raise ValueError(f"{field.name} already has a "
                                 f"{self.name} charge")
            self.fermion_charges[comp] = qs
            self.fermion_charges[bar] = tuple(-q for q in qs)

    def _assign_scalar(self, charge, field):
        if isinstance(charge, (tuple, list)):
            raise ValueError("per-flavour charges apply to fermions only; "
                             f"give {field} a single charge")
        q = _real_charge(charge)
        comps = (list(field.components) if hasattr(field, "components")
                 else [field])
        for comp in comps:
            if not isinstance(comp, sp.Symbol):
                raise TypeError(f"expected a Field or Symbol, got {comp!r}")
            if comp in self.scalar_charges:
                raise ValueError(f"{comp} already has a {self.name} charge")
            if comp.is_real and q != 0:
                raise ValueError(f"the real component {comp} cannot carry a "
                                 f"non-zero {self.name} charge")
            self.scalar_charges[comp] = q

    def components(self):
        """The charged scalar component symbols."""
        return list(self.scalar_charges)

    def leg_charge(self, leg):
        """The charge of a fermion leg ``Indexed(base, flavour)``.

        An integer flavour index reads its own charge.  A symbolic index is
        allowed only when every flavour shares the charge — otherwise the
        phase of ``ψ_i`` is undefined, and a summed term must be written out
        flavour by flavour.
        """
        qs = self.fermion_charges.get(leg.base)
        if qs is None:
            return sp.S.Zero
        k = leg.indices[0]
        if k.is_Integer:
            return qs[int(k)]
        if len(set(qs)) == 1:
            return qs[0]
        raise ValueError(
            f"{leg} has a symbolic flavour index but {leg.base} carries "
            f"flavour-dependent {self.name} charges {qs}; write the term "
            "with explicit integer flavour indices")

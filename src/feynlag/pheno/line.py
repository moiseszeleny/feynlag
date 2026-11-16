"""One open fermion line with external bosons: ``f f̄ → B B``, ``f B → f B``.

Tier 4 of ``docs/manual/scattering_roadmap.md``.  The four-fermion engine in
:mod:`~feynlag.pheno.diagrams` has exactly one vertex on each fermion line and
no external boson.  A process with **two external fermions and two external
bosons** is a different shape: every diagram is the *same* single fermion
line, dressed either with two vertices and a fermion propagator between them
(``e⁺e⁻→γγ``, Compton, the ν exchange of ``e⁺e⁻→W⁺W⁻``) or with one vertex
whose boson then splits at a bosonic three-point vertex (the γ/Z s channel of
``e⁺e⁻→W⁺W⁻``).

An amplitude is therefore kept as a list of :class:`LineTerm`\\ s,

``M = Σ_terms coeff × ū_bar [γ-string] u_field × (loose tensors)``,

with the external vector indices left open, and :meth:`LineAmplitude.squared`
evaluates

``Σ|M|² = Σ_{a,b} c_a c̄_b Tr[S_bar Γ_a S_field Γ̄_b] × ∏_V P_V(μ_V, μ'_V)``

through the scalar route of :mod:`~feynlag.pheno.contraction` (no SymPy
tensors: a squared fermion-exchange diagram slashes one momentum twice, which
``gamma_trace`` mis-evaluates).

Signs.  With a single fermion line there is **no** fermion-permutation sign
between diagrams.  The relative phases are those of the Feynman rules
themselves: the vertex couplings (which carry the rule's ``i``), ``i`` for a
fermion or scalar propagator and ``−i`` for a vector one.

The γ₅ part.  Every Lorentz index ends up contracted with a metric or a
momentum (unpolarized sums), so the γ₅ part of each trace is a pseudoscalar
built from the process's own momenta.  A 2→2 process has three independent
ones, which cannot fill an ε tensor;
:func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes` proves it on
the actual ``dot`` table before the γ₅ coefficients are dropped.

Polarization sums.  A massive vector uses ``−g_{μμ'} + k_μk_{μ'}/m²``.  A
massless one uses ``−g_{μμ'}``, which is only right if the amplitude obeys
the Ward identity ``k_μ M^μ = 0``; :meth:`LineAmplitude.squared` *checks*
that (it replaces the leg's sum by ``k_μk_{μ'}`` and requires zero) and
refuses otherwise — the case of two external non-abelian massless vectors,
which needs ghosts or physical polarizations (Tier 5).
"""

import itertools
from dataclasses import dataclass

import sympy as sp

from .contraction import contract_trace
from .epsilon import assert_epsilon_single_vanishes

__all__ = ["LineAmplitude", "LineTerm", "VectorLeg", "fermion_exchange_term"]

_dummy_counter = itertools.count()


def fresh_name(tag="d"):
    """A never-before-used index name for the scalar route."""
    return f"{tag}_{next(_dummy_counter)}"


@dataclass(frozen=True)
class LineTerm:
    """One term of a fermion-line amplitude.

    Attributes:
        coeff: scalar prefactor — propagator phases and ``1/denominators``
            (the vertex couplings sit in the ``'G'`` factors).
        string: the Dirac factors between the bar-slot and the field-slot
            spinor, in matrix order, in the format of
            :func:`~feynlag.pheno.contraction._expand_chiral_string`
            (``('G', index-name | None, g_L, g_R)`` and
            ``('Q', combo, mass)``).
        tensors: loose ``('g', a, b)`` / ``('v', combo, a)`` tensors on the
            index names (a bosonic three-point vertex, a propagator
            numerator).
    """

    coeff: sp.Expr
    string: tuple
    tensors: tuple = ()

    def index_names(self):
        names = []
        for factor in self.string:
            if factor[0] == "G" and factor[1] is not None \
                    and not isinstance(factor[1], tuple):
                names.append(factor[1])
        for tensor in self.tensors:
            names.extend(tensor[1:] if tensor[0] == "g" else tensor[2:])
        return names

    def renamed(self, mapping):
        """This term with its index names sent through ``mapping``."""
        string = tuple(
            ("G", mapping[f[1]], f[2], f[3])
            if f[0] == "G" and f[1] is not None and not isinstance(f[1], tuple)
            else f
            for f in self.string)
        tensors = tuple(
            ("g", mapping[t[1]], mapping[t[2]]) if t[0] == "g"
            else ("v", t[1], mapping[t[2]])
            for t in self.tensors)
        return LineTerm(self.coeff, string, tensors)

    def bar(self, external):
        """The term of ``M̄``: ``Γ̄ = γ⁰Γ†γ⁰`` with the string reversed.

        A γ-carrying factor keeps its chirality labels and conjugates its
        couplings (``γ^μP_L`` is self-conjugate); a scalar factor conjugates
        **and exchanges** them (``P̄_L = P_R``) — the rule of
        :meth:`~feynlag.pheno.diagrams.ChainVertex.bar`.  A propagator
        numerator ``q̸ + m`` is self-conjugate.  The external index names in
        ``external`` are primed, every other (dummy) name is replaced by a
        fresh one.
        """
        mapping = {}
        for name in self.index_names():
            if name not in mapping:
                mapping[name] = (primed(name) if name in external
                                 else fresh_name("b"))
        string = []
        for factor in reversed(self.string):
            if factor[0] != "G":
                string.append(factor)
            elif factor[1] is None:
                string.append(("G", None, sp.conjugate(factor[3]),
                               sp.conjugate(factor[2])))
            else:
                string.append(("G", factor[1], sp.conjugate(factor[2]),
                               sp.conjugate(factor[3])))
        return LineTerm(sp.conjugate(self.coeff), tuple(string),
                        self.tensors).renamed(mapping)

    def is_chiral(self):
        return any(f[0] == "G" and sp.simplify(f[2] - f[3]) != 0
                   for f in self.string)


def primed(name):
    return f"{name}'"


@dataclass(frozen=True)
class VectorLeg:
    """An external spin-1 boson: its open index name, momentum head and mass."""

    index: str
    momentum: object
    mass: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "mass", sp.sympify(self.mass))

    def polarization_sum(self):
        """``Σ ε_μ ε*_{μ'}`` as ``[(coeff, tensors)]`` on ``index`` and its
        primed partner: ``−g`` and, for a massive vector, ``+k k/m²``."""
        a, b = self.index, primed(self.index)
        pieces = [(-sp.S.One, (("g", a, b),))]
        if self.mass != 0:
            pieces.append((1 / self.mass**2, self.longitudinal()))
        return pieces

    def longitudinal(self):
        """The tensors of ``k_μ k_{μ'}``."""
        k = ((sp.S.One, self.momentum),)
        return (("v", k, self.index), ("v", k, primed(self.index)))


def fermion_exchange_term(vertex_bar, vertex_field, q, q2, mass, width=0):
    """``Γ₁ · i(q̸ + m)/(q² − m² + imΓ) · Γ₂`` as a :class:`LineTerm`.

    Args:
        vertex_bar, vertex_field: the ``('G', …)`` factors next to the
            bar-slot and the field-slot spinor.
        q: the propagator momentum **along the fermion arrow** (from the
            field slot towards the bar slot), as ``((c, head), …)``.
        q2: its invariant ``q²``.
        mass, width: of the exchanged fermion.
    """
    from .propagator import fermion_propagator_phase, propagator_denominator
    mass = sp.sympify(mass)
    coeff = fermion_propagator_phase() / propagator_denominator(q2, mass, width)
    return LineTerm(coeff, (vertex_bar, ("Q", tuple(q), mass), vertex_field))


def _is_zero(expr, reference, symbols):
    """Is ``expr`` identically zero?  Exact if it simplifies to ``0``;
    otherwise (float couplings leave round-off) compared numerically with
    ``reference`` at two generic points of ``symbols``."""
    if expr == 0:
        return True
    simplified = sp.simplify(sp.together(expr))
    if simplified == 0:
        return True
    if not simplified.has(sp.Float):
        return False
    for point in ((sp.Rational(731, 100), sp.Rational(-213, 100)),
                  (sp.Rational(1907, 100), sp.Rational(-477, 100))):
        values = dict(zip(symbols, point))
        try:
            num = complex(expr.subs(values))
            ref = complex(reference.subs(values))
        except TypeError:
            return False            # other free symbols: not a float residue
        if abs(num) > 1e-9 * max(abs(ref), 1e-300):
            return False
    return True


@dataclass(frozen=True)
class LineAmplitude:
    """A sum of diagrams on one open fermion line.

    Args:
        out: the bar-slot :class:`~feynlag.pheno.diagrams.Leg` (outgoing
            particle / incoming antiparticle).
        inn: the field-slot leg (incoming particle / outgoing antiparticle).
        diagrams: one tuple of :class:`LineTerm` per diagram.
        vectors: the external :class:`VectorLeg`\\ s (external scalars carry
            no index and need no entry).
    """

    out: object
    inn: object
    diagrams: tuple
    vectors: tuple = ()

    def terms(self):
        return [term for diagram in self.diagrams for term in diagram]

    def _summed(self, kin, polarization):
        """``Σ_{a,b}`` with the given ``{index: [(coeff, tensors)]}`` sums."""
        external = {leg.index for leg in self.vectors}
        terms = self.terms()
        barred = [term.bar(external) for term in terms]
        choices = list(itertools.product(
            *(polarization[leg.index] for leg in self.vectors)))
        memo = {}
        total = sp.S.Zero
        for a in terms:
            for b in barred:
                factors = ((("S", self.out),) + a.string
                           + (("S", self.inn),) + b.string)
                piece = sp.S.Zero
                for choice in choices:
                    weight = sp.S.One
                    tensors = a.tensors + b.tensors
                    for c, extra in choice:
                        weight *= c
                        tensors += extra
                    piece += weight * contract_trace(factors, tensors,
                                                     kin.dot, memo)
                total += a.coeff * b.coeff * piece
        return total

    def squared(self, kin, ward_check=True):
        """The spin/polarization-**summed** ``|M|²`` (no averaging — see
        :func:`~feynlag.pheno.scattering.average_factor`).

        Args:
            kin: a :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
            ward_check: verify ``k_μ M^μ = 0`` for every massless external
                vector before trusting its ``−g`` polarization sum.

        Raises:
            NotImplementedError: a massless external vector whose Ward
                identity fails, or a γ₅ term that does not provably vanish.
        """
        if any(term.is_chiral() for term in self.terms()):
            assert_epsilon_single_vanishes(
                [kin.k1, kin.k2, kin.k3, kin.k4], kin.dot)
        sums = {leg.index: leg.polarization_sum() for leg in self.vectors}
        total = self._summed(kin, sums)
        if ward_check:
            for leg in self.vectors:
                if leg.mass != 0:
                    continue
                probe = dict(sums)
                probe[leg.index] = [(sp.S.One, leg.longitudinal())]
                residue = self._summed(kin, probe)
                if not _is_zero(residue, total, (kin.s, kin.t)):
                    raise NotImplementedError(
                        f"LineAmplitude.squared: the Ward identity "
                        f"k·M = 0 fails for the massless external vector "
                        f"with momentum {leg.momentum.name}, so its "
                        f"polarization sum is not −g. Two external massless "
                        f"non-abelian vectors need ghosts or physical "
                        f"polarizations (Tier 5 of the scattering roadmap); "
                        f"otherwise a diagram is missing.")
        return sp.expand(total)

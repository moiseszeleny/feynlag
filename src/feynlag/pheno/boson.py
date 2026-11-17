"""An amplitude with no fermion line: four external bosons (``gg→gg``).

Tier 5, stage 5c of ``docs/manual/scattering_roadmap.md``.  The four-fermion
engine (:mod:`~feynlag.pheno.diagrams`) and the line engine
(:mod:`~feynlag.pheno.line`) both hang their Lorentz structure on a Dirac
trace.  A pure-boson amplitude has none: every term is a scalar coefficient
times loose tensors — metrics and momentum vectors — on the external vector
indices and on internal (propagator) dummies,

``M = Σ_terms coeff × ∏ tensors``,

and :meth:`BosonAmplitude.squared` evaluates

``Σ|M|² = Σ_{d,d'} C_{dd'} Σ_{a∈d, b∈d'} c_a c̄_b (T_a · ∏_V P_V · T_b)``

with :func:`~feynlag.pheno.contraction.contract_tensors`, the trace-free
sibling of ``contract_trace``.

How the sum is organised.  Done term by term this is slow: ``gg→gg`` has 33
terms and four three-piece polarization sums, i.e. ``33² × 3⁴`` contractions
(over ten minutes).  Two exact reductions bring it to seconds:

1. each diagram is rewritten as a few **canonical monomials** — metrics and
   single momenta on the four external indices.  A vector on external index
   ``μ_V`` is taken modulo momentum conservation and modulo the momenta the
   leg's polarization sum annihilates (``k`` and the reference ``n`` for a
   physical sum).  Those are *computed* from the sum actually used
   (:func:`_null_heads`), so the Ward probes, which replace one sum by
   ``k k``, are reduced correctly too;
2. the contraction of two monomials through the sums factorises over the
   connected components of its index graph, and a component only sums over
   the pieces of the legs it contains.

Everything else is shared with the line engine on purpose: the external legs
are the same :class:`~feynlag.pheno.line.VectorLeg`\\ s (a massless
non-abelian vector must carry a ``reference`` for its physical polarization
sum), and the polarization sums are trusted only after the same computed Ward
check (:func:`~feynlag.pheno.line.check_ward_identities`).

Signs.  There is no fermion sign.  The relative phase between an exchange
diagram and the contact term is that of the Feynman rules: two cubic
couplings and ``−i`` for the vector propagator against the quartic coupling,
which carries its own ``i``.  A process built from cubic vertices in pairs is
blind to the overall sign of the cubic coupling; it is the *contact* term's
sign that the Ward identity tests.
"""

import itertools
from dataclasses import dataclass

import sympy as sp

from .contraction import _combo_dot, contract_tensors
from .line import check_ward_identities, fresh_name, primed

__all__ = ["BosonAmplitude", "BosonTerm"]


def _names(tensors):
    names = []
    for tensor in tensors:
        names.extend(tensor[1:] if tensor[0] == "g" else tensor[2:])
    return names


def _renamed(tensors, mapping):
    return tuple(
        ("g", mapping[t[1]], mapping[t[2]]) if t[0] == "g"
        else ("v", t[1], mapping[t[2]])
        for t in tensors)


def _cached_dot(dot, memo):
    """``dot`` with its values kept in ``memo`` (the kinematics rebuild
    their table on every call, and a ``Σ|M|²`` asks tens of thousands of
    times)."""
    table = memo.setdefault("dot", {})

    def cached(a, b):
        key = (a, b)
        if key not in table:
            table[key] = dot(a, b)
        return table[key]

    return cached


def _tidy(expr):
    """An exact rational function brought over one denominator, so that a
    vanishing Ward residue is literally ``0``."""
    if expr.has(sp.Float):
        return expr
    return sp.cancel(sp.together(expr))


def _null_heads(pieces, index, heads, dot):
    """The momentum heads a polarization sum annihilates:
    ``P^{μμ'} h_μ = 0`` and ``P^{μμ'} h_{μ'} = 0``, computed from the pieces.

    A physical sum ``−g + (kn + nk)/(k·n)`` kills ``k`` and ``n``; ``−g``
    kills nothing; the Ward probe ``k k`` kills ``k``.  A vector of the
    amplitude along such a head drops out of ``Σ|M|²``, which is what lets
    :func:`_canonical` shrink the amplitude.  Unrecognised pieces give the
    safe answer: no head.
    """
    a, b = index, primed(index)
    nulls = []
    for head in heads:
        probe = ((sp.S.One, head),)
        dead = True
        for side in (a, b):
            image = {}
            for coeff, tensors in pieces:
                kinds = [t[0] for t in tensors]
                if kinds == ["g"]:
                    image[head] = image.get(head, 0) + coeff
                elif kinds == ["v", "v"]:
                    here = [t for t in tensors if t[2] == side]
                    there = [t for t in tensors if t[2] != side]
                    if len(here) != 1 or len(there) != 1:
                        return []
                    weight = coeff * _combo_dot(here[0][1], probe, dot)
                    for c, h in there[0][1]:
                        image[h] = image.get(h, 0) + weight * c
                else:
                    return []
            if any(sp.simplify(value) != 0 for value in image.values()):
                dead = False
                break
        if dead:
            nulls.append(head)
    return nulls


def _externalise(tensors, external, dot):
    """Contract away every internal index of one term.

    Returns ``(scalar, metrics, vectors)``: metrics between external
    indices, and ``{external index: combo}``.
    """
    metrics = [[t[1], t[2]] for t in tensors if t[0] == "g"]
    vectors = [[t[1], t[2]] for t in tensors if t[0] == "v"]
    coeff = sp.S.One
    while True:
        for n, (a, b) in enumerate(metrics):
            if a not in external or b not in external or a == b:
                break
        else:
            break
        metrics.pop(n)
        if a == b:
            coeff *= 4
            continue
        if a not in external:
            a, b = b, a                       # ``b`` is the internal one
        for metric in metrics:
            metric[:] = [a if x == b else x for x in metric]
        for vector in vectors:
            if vector[1] == b:
                vector[1] = a
    by_index = {}
    for combo, a in vectors:
        by_index.setdefault(a, []).append(combo)
    out = {}
    for a, combos in by_index.items():
        if a in external:
            (out[a],) = combos
        else:
            first, second = combos
            coeff *= _combo_dot(first, second, dot)
    return coeff, [tuple(m) for m in metrics], out


#: momentum conservation ``k1 + k2 − k3 − k4 = 0``
_CONSERVATION = (1, 1, -1, -1)


def _reduced(combo, nulls, heads):
    """A vector on an external index as ``{head: coeff}``, modulo the heads
    its polarization sum annihilates and momentum conservation (the last
    surviving head is eliminated)."""
    acc = {}
    for c, head in combo:
        acc[head] = acc.get(head, 0) + c
    for head in nulls:
        acc.pop(head, None)
    rest = [head for head in heads if head not in nulls]
    if rest:
        last = rest[-1]
        c_last = acc.pop(last, 0)
        if c_last != 0:
            sign = _CONSERVATION[heads.index(last)]
            for head in rest[:-1]:
                acc[head] = acc.get(head, 0) - sp.Rational(
                    _CONSERVATION[heads.index(head)], sign) * c_last
    return {head: c for head, c in acc.items() if sp.expand(c) != 0}


def _canonical(diagram, external, nulls, heads, dot):
    """A diagram as ``{monomial: coefficient}``.

    A monomial is a sorted tuple of metrics and single-head vectors carrying
    each external index exactly once.  Internal indices are contracted, and
    every vector is reduced with :func:`_reduced` — exact inside
    ``Σ|M|²`` for the given polarization sums.
    """
    out = {}
    for term in diagram:
        coeff, metrics, vectors = _externalise(term.tensors, external, dot)
        indices = sorted(vectors)
        options = [_reduced(vectors[a], nulls.get(a, ()), heads)
                   for a in indices]
        for picks in itertools.product(*(opt.items() for opt in options)):
            weight = term.coeff * coeff
            tensors = [("g",) + tuple(sorted(m)) for m in metrics]
            for a, (head, c) in zip(indices, picks):
                weight *= c
                tensors.append(("v", ((sp.S.One, head),), a))
            key = tuple(sorted(tensors, key=str))
            out[key] = out.get(key, 0) + weight
    return {key: value for key, value in out.items() if value != 0}


@dataclass(frozen=True)
class BosonTerm:
    """One term of a pure-boson amplitude.

    Attributes:
        coeff: scalar prefactor — couplings, propagator phases and
            ``1/denominators``.
        tensors: loose ``('g', a, b)`` / ``('v', combo, a)`` tensors on the
            index names (vertex structures, propagator numerators).
    """

    coeff: sp.Expr
    tensors: tuple = ()

    def bar(self, external):
        """The term of ``M̄``: the coefficient conjugated (metrics and
        momenta are real), the external index names in ``external`` primed
        and every other (dummy) name replaced by a fresh one."""
        mapping = {}
        for name in _names(self.tensors):
            if name not in mapping:
                mapping[name] = (primed(name) if name in external
                                 else fresh_name("b"))
        return BosonTerm(sp.conjugate(self.coeff),
                         _renamed(self.tensors, mapping))


@dataclass(frozen=True)
class BosonAmplitude:
    """A sum of diagrams with bosons only.

    Args:
        diagrams: one tuple of :class:`BosonTerm` per diagram (per colour
            structure, when a colour matrix is used).
        vectors: the external :class:`~feynlag.pheno.line.VectorLeg`\\ s
            (external scalars carry no index and need no entry).
    """

    diagrams: tuple
    vectors: tuple = ()

    def terms(self):
        return [term for diagram in self.diagrams for term in diagram]

    def _summed(self, kin, polarization, colour=None, memo=None):
        """``Σ_{d,d'}`` with the given ``{index: [(coeff, tensors)]}`` sums,
        each pair of diagrams weighted by ``colour[d][d']``.

        Each diagram is first reduced to a few canonical monomials
        (:func:`_canonical`); the contraction of two monomials through the
        sums factorises over the connected components of its index graph
        (:meth:`_gram`), which are cached in ``memo``."""
        memo = {} if memo is None else memo
        dot = _cached_dot(kin.dot, memo)
        heads = (kin.k1, kin.k2, kin.k3, kin.k4)
        external = {leg.index for leg in self.vectors}
        nulls = {leg.index: _null_heads(polarization[leg.index], leg.index,
                                        heads, dot)
                 for leg in self.vectors}
        canon = [_canonical(diagram, external, nulls, heads, dot)
                 for diagram in self.diagrams]
        barred = [{_renamed(m, {x: primed(x) for x in external}):
                   sp.conjugate(c) for m, c in diagram.items()}
                  for diagram in canon]
        total = sp.S.Zero
        for i, diagram in enumerate(canon):
            for j, conjugate in enumerate(barred):
                factor = sp.S.One if colour is None else colour[i][j]
                if factor == 0:
                    continue
                pair = sp.S.Zero
                for m, c in diagram.items():
                    for mb, cb in conjugate.items():
                        pair += c * cb * self._gram(m, mb, polarization,
                                                    dot, memo)
                total += factor * pair
        return total

    def _gram(self, m, mb, polarization, dot, memo):
        """``m · ∏_V P_V · m̄``: the two monomials contracted through the
        polarization sums, as a product over the connected components of the
        index graph (a sum over the pieces of the legs in that component
        only, instead of over all legs at once)."""
        tensors = m + mb
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for tensor in tensors:
            names = _names((tensor,))
            for name in names[1:]:
                parent[find(name)] = find(names[0])
        for leg in self.vectors:
            parent[find(primed(leg.index))] = find(leg.index)
        groups = {}
        for tensor in tensors:
            groups.setdefault(find(_names((tensor,))[0]), []).append(tensor)
        value = sp.S.One
        for root, group in groups.items():
            legs = tuple(leg.index for leg in self.vectors
                         if find(leg.index) == root)
            sums = tuple(tuple(polarization[index]) for index in legs)
            key = (tuple(sorted(group, key=str)), sums)
            if key not in memo:
                part = sp.S.Zero
                for choice in itertools.product(*sums):
                    weight, extra = sp.S.One, ()
                    for c, piece in choice:
                        weight *= c
                        extra += piece
                    part += weight * contract_tensors(key[0] + extra, dot)
                memo[key] = sp.factor(sp.together(part)) \
                    if not part.has(sp.Float) else sp.expand(part)
            value *= memo[key]
            if value == 0:
                break
        return value

    def squared(self, kin, ward_check=True, colour=None):
        """The polarization-**summed** ``|M|²`` (no averaging — see
        :func:`~feynlag.pheno.scattering.average_factor`).

        Args:
            kin: a :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
            ward_check: verify ``k_μ M^μ = 0`` for every massless external
                vector before trusting its polarization sum
                (:func:`~feynlag.pheno.line.check_ward_identities`).
            colour: optional colour matrix ``C[d][d']``, one row per diagram
                (:func:`~feynlag.pheno.colour.colour_matrix`).

        Raises:
            NotImplementedError: a massless external vector whose Ward
                identity fails.
        """
        if colour is not None and len(colour) != len(self.diagrams):
            raise ValueError(
                "BosonAmplitude.squared: the colour matrix needs one row per "
                "diagram")
        sums = {leg.index: leg.polarization_sum(kin.dot)
                for leg in self.vectors}
        memo = {}
        total = self._summed(kin, sums, colour, memo)
        if ward_check:
            check_ward_identities(
                lambda probe: _tidy(self._summed(kin, probe, colour, memo)),
                self.vectors, sums, total, kin, "BosonAmplitude.squared")
        return sp.expand(total)

"""s/t/u topology enumeration for 2→2 four-fermion processes (Tier 3).

The 2→2 analogue of :meth:`~feynlag.pheno.calculator.DecayCalculator.channels`'s
"vertex contains the parent" search.  Given four external fermions and the
extracted three-leg vertices (:class:`~feynlag.pheno.vertices.DecayVertex`),
:func:`enumerate_diagrams` builds every tree diagram with two fermion lines
joined by one boson propagator, as :class:`~feynlag.pheno.diagrams.Diagram`
objects ready for :meth:`~feynlag.pheno.diagrams.Amplitude.squared`.

**Slots.**  Each external fermion occupies the ψ (field) or ψ̄ (bar) slot of
its line, by the standard external-line table (see
:class:`~feynlag.pheno.diagrams.Leg`): an incoming particle or an outgoing
antiparticle is a *field* slot (``u``/``v``), an outgoing particle or an
incoming antiparticle a *bar* slot (``ū``/``v̄``).  A line joining bar slot
``X`` to field slot ``Y`` needs a vertex ``ψ̄_X Γ ψ_Y`` — a ``DecayVertex``
whose bar leg is ``X``'s antiparticle symbol and whose field leg is ``Y``'s
particle symbol (the :class:`~feynlag.pheno.particles.DiracParticle`
``particle_map`` convention).

**Pairings.**  With two bar and two field slots there are exactly two ways to
pair them into lines; both are enumerated.  For e⁺e⁻→μ⁺μ⁻ only one pairing
has vertices (the s channel); for e⁺e⁻→e⁺e⁻ both do (Bhabha's s and t);
for e⁻e⁻→e⁻e⁻ both do (Møller's t and u).

**Propagator.**  A mediator joins a vertex carrying the boson field ``B`` on
one line to a vertex carrying ``B†`` on the other (``⟨B B†⟩`` is the only
non-zero contraction), so the caller passes the antiparticle map for charged
bosons, e.g. ``{Wp: Wm, Wm: Wp}`` — the same explicit-map convention as
:func:`~feynlag.charges.check_hermiticity_pairing`, since ``W±`` come from a
rotation rather than from ``conjugate_pair``.  A boson absent from the map is
taken as self-conjugate.

**Fermion sign.**  Each diagram's relative sign is the parity of the
permutation ``(bar_A, field_A, bar_B, field_B)`` of the external labels
``0..3`` (Wick's theorem: each bilinear ``ψ̄ψ`` is bosonic, so the order of
the two lines does not matter, and this parity is invariant under swapping
them).  It is stored in :attr:`~feynlag.pheno.diagrams.Diagram.coefficient`.
For Bhabha it gives the textbook relative −1 between the s and t channels,
which ``tests/test_interference.py`` pins against the closed form and against
an explicit-spinor oracle that shares no code with this module.
"""

import itertools
from dataclasses import dataclass

import sympy as sp
from sympy.combinatorics import Permutation

from .diagrams import (
    BosonPropagator, ChainVertex, Diagram, Leg, SpinorChain, _chain_indices,
)

__all__ = ["DiagramInfo", "ExternalFermion", "enumerate_diagrams"]

_diagram_counter = itertools.count()


@dataclass(frozen=True)
class ExternalFermion:
    """One external fermion of a 2→2 process.

    Args:
        particle: the :class:`~feynlag.pheno.particles.DiracParticle` species.
        anti: ``True`` for the antiparticle.
        incoming: ``True`` for an initial-state particle.
    """

    particle: object
    anti: bool
    incoming: bool

    @property
    def symbol(self):
        """The particle or antiparticle symbol."""
        return self.particle.antiparticle if self.anti else self.particle.particle

    @property
    def slot(self):
        """``'field'`` (ψ: incoming particle / outgoing antiparticle) or
        ``'bar'`` (ψ̄: outgoing particle / incoming antiparticle)."""
        return "field" if self.incoming != self.anti else "bar"


@dataclass(frozen=True)
class DiagramInfo:
    """One enumerated diagram and how it was built.

    Attributes:
        channel: ``'s'``, ``'t'`` or ``'u'``.
        mediator: the boson field symbol on line A's vertex.
        sign: the fermion-permutation sign (±1).
        vertices: the two :class:`~feynlag.pheno.vertices.DecayVertex` used.
        diagram: the :class:`~feynlag.pheno.diagrams.Diagram`.
    """

    channel: str
    mediator: sp.Expr
    sign: int
    vertices: tuple
    diagram: Diagram

    def __repr__(self):
        sign = "+" if self.sign > 0 else "-"
        return f"DiagramInfo({sign}{self.channel}-channel {self.mediator})"


_CHANNELS = {
    frozenset({0, 1}): "s", frozenset({2, 3}): "s",
    frozenset({0, 2}): "t", frozenset({1, 3}): "t",
    frozenset({0, 3}): "u", frozenset({1, 2}): "u",
}


def _channel_momentum(channel, kin):
    """Canonical propagator momentum (its overall sign never matters: a
    numerator is quadratic in ``q``)."""
    if channel == "s":
        return lambda i: kin.k1(i) + kin.k2(i)
    if channel == "t":
        return lambda i: kin.k1(i) - kin.k3(i)
    return lambda i: kin.k1(i) - kin.k4(i)


def _channel_q2(channel, kin):
    return {"s": kin.s, "t": kin.t, "u": kin.u}[channel]


def enumerate_diagrams(externals, vertices, kin, mass_of, width_of=None,
                       conjugates=None):
    """Every two-line, one-propagator tree diagram of a 2→2 fermion process.

    Args:
        externals: four :class:`ExternalFermion`, in the order of the
            kinematics' momenta ``k1..k4`` (two incoming, then two outgoing).
        vertices: :class:`~feynlag.pheno.vertices.DecayVertex` list (only the
            ``FFV``/``FFS`` ones are used).
        kin: a :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
        mass_of: callable ``boson -> mass`` (should raise ``KeyError`` for an
            undeclared boson rather than assume zero).
        width_of: optional callable ``boson -> width`` (default ``0``).
        conjugates: ``{boson: antiboson}`` for charged mediators.

    Returns:
        list of :class:`DiagramInfo`.

    Raises:
        ValueError: the externals do not form two bar and two field slots
            (fermion number is not conserved by any pairing).
    """
    conjugates = dict(conjugates or {})
    width_of = width_of or (lambda boson: sp.S.Zero)
    heads = (kin.k1, kin.k2, kin.k3, kin.k4)
    masses = (kin.m1, kin.m2, kin.m3, kin.m4)
    legs = [Leg(heads[j], masses[j], anti=ext.anti) for j, ext in enumerate(externals)]
    bars = [j for j, ext in enumerate(externals) if ext.slot == "bar"]
    fields = [j for j, ext in enumerate(externals) if ext.slot == "field"]
    if len(bars) != 2 or len(fields) != 2:
        raise ValueError(
            "enumerate_diagrams: a 2→2 four-fermion process needs two ψ̄ "
            "(bar) and two ψ (field) slots; got bar slots "
            f"{[str(externals[j].symbol) for j in bars]} and field slots "
            f"{[str(externals[j].symbol) for j in fields]}")

    def line_vertices(bar, field):
        want_bar = externals[bar].particle.antiparticle
        want_field = externals[field].particle.particle
        return [v for v in vertices
                if v.vertex_type in ("FFV", "FFS")
                and v.particles[0] == want_bar and v.particles[1] == want_field]

    out = []
    for field_order in (fields, fields[::-1]):
        (bar_a, field_a), (bar_b, field_b) = zip(bars, field_order)
        channel = _CHANNELS[frozenset({bar_a, field_a})]
        sign = Permutation([bar_a, field_a, bar_b, field_b]).signature()
        candidates_b = line_vertices(bar_b, field_b)
        for vertex_a in line_vertices(bar_a, field_a):
            boson = vertex_a.particles[2]
            partner = conjugates.get(boson, boson)
            for vertex_b in candidates_b:
                if vertex_b.particles[2] != partner:
                    continue
                if vertex_b.vertex_type != vertex_a.vertex_type:
                    continue
                out.append(_build(channel, sign, boson, vertex_a, vertex_b,
                                  (legs[bar_a], legs[field_a]),
                                  (legs[bar_b], legs[field_b]),
                                  kin, mass_of, width_of))
    return out


def _build(channel, sign, boson, vertex_a, vertex_b, line_a, line_b, kin,
           mass_of, width_of):
    spin = 1 if vertex_a.vertex_type == "FFV" else 0
    structure = "V" if spin == 1 else "S"
    tag = f"td{next(_diagram_counter)}"
    chains = []
    for name, vertex, (bar_leg, field_leg) in (("a", vertex_a, line_a),
                                              ("b", vertex_b, line_b)):
        if spin == 1:
            mu, mu_bar = _chain_indices(f"{tag}{name}")
            idx, bar_idx = (mu,), (mu_bar,)
        else:
            idx = bar_idx = ()
        chains.append(SpinorChain(
            out=bar_leg, inn=field_leg,
            vertices=(ChainVertex(structure, vertex.g_left, vertex.g_right),),
            indices=idx, bar_indices=bar_idx))
    mass = sp.sympify(mass_of(boson))
    momentum = _channel_momentum(channel, kin) if (spin == 1 and mass != 0) else None
    prop = BosonPropagator(spin=spin, q2=_channel_q2(channel, kin), mass=mass,
                           width=sp.sympify(width_of(boson)), momentum=momentum)
    diagram = Diagram(chains=tuple(chains), propagators=(prop,), coefficient=sign)
    return DiagramInfo(channel=channel, mediator=boson, sign=sign,
                       vertices=(vertex_a, vertex_b), diagram=diagram)

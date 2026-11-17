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
from .boson_vertices import sss_tensor, vss_tensor, vvs_tensor, vvv_tensor
from .line import (
    LineAmplitude, VectorLeg, boson_exchange_terms, fermion_exchange_term,
    fresh_name,
)

__all__ = ["DiagramInfo", "ExternalBoson", "ExternalFermion",
           "LineDiagramInfo", "enumerate_diagrams", "enumerate_line_diagrams",
           "line_amplitude"]

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

    @property
    def spin(self):
        return sp.Rational(1, 2)


@dataclass(frozen=True)
class ExternalBoson:
    """One external boson of a 2→2 process.

    Args:
        symbol: the boson's **particle** symbol — the field symbol that
            annihilates it (``Wp`` for a W⁺).
        spin: ``0`` or ``1``.
        mass: its mass.
        incoming: ``True`` for an initial-state particle.
    """

    symbol: sp.Expr
    spin: int
    mass: sp.Expr
    incoming: bool

    def __post_init__(self):
        object.__setattr__(self, "symbol", sp.sympify(self.symbol))
        object.__setattr__(self, "spin", sp.sympify(self.spin))
        object.__setattr__(self, "mass", sp.sympify(self.mass))
        if self.spin not in (0, 1):
            raise NotImplementedError(
                f"ExternalBoson: unsupported spin {self.spin}")


@dataclass(frozen=True)
class DiagramInfo:
    """One enumerated diagram and how it was built.

    Attributes:
        channel: ``'s'``, ``'t'`` or ``'u'``.
        mediator: the boson field symbol on line A's vertex.
        sign: the fermion-permutation sign (±1).
        vertices: the two :class:`~feynlag.pheno.vertices.DecayVertex` used.
        diagram: the :class:`~feynlag.pheno.diagrams.Diagram`.
        lines: ``((bar_a, field_a), (bar_b, field_b))`` — the external
            positions ``0..3`` each fermion line joins (what a colour
            tensor is built from).
    """

    channel: str
    mediator: sp.Expr
    sign: int
    vertices: tuple
    diagram: Diagram
    lines: tuple = ()

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
                                  kin, mass_of, width_of,
                                  lines=((bar_a, field_a), (bar_b, field_b))))
    return out


def _build(channel, sign, boson, vertex_a, vertex_b, line_a, line_b, kin,
           mass_of, width_of, lines=()):
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
                       vertices=(vertex_a, vertex_b), diagram=diagram,
                       lines=lines)


# --------------------------------------------------------------------------
# two external fermions + two external bosons: one open fermion line (Tier 4)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LineDiagramInfo:
    """One enumerated fermion-line diagram and how it was built.

    Attributes:
        channel: ``'s'``, ``'t'`` or ``'u'`` — the invariant carried by the
            internal propagator.
        mediator: the particle symbol of the exchanged fermion, or the
            boson field symbol on the line's vertex.
        kind: ``'fermion'`` (two vertices on the line, a fermion propagator)
            or ``'boson'`` (one vertex on the line, a boson propagator and a
            bosonic three-point vertex).
        vertices: the :class:`~feynlag.pheno.vertices.DecayVertex` used — in
            line order (bar-slot end first) for ``'fermion'``; the line
            vertex then the bosonic vertex for ``'boson'``.
        terms: the diagram's :class:`~feynlag.pheno.line.LineTerm`\\ s.
    """

    channel: str
    mediator: sp.Expr
    kind: str
    vertices: tuple
    terms: tuple

    def __repr__(self):
        return f"LineDiagramInfo({self.channel}-channel {self.mediator})"


def _external_index(j):
    """The open Lorentz-index name of external leg ``j`` (0-based)."""
    return f"x{j + 1}"


def _split_externals(externals):
    fermions = [j for j, ext in enumerate(externals)
                if isinstance(ext, ExternalFermion)]
    bosons = [j for j, ext in enumerate(externals)
              if isinstance(ext, ExternalBoson)]
    if len(externals) != 4 or len(fermions) != 2 or len(bosons) != 2:
        raise ValueError(
            "enumerate_line_diagrams: needs two external fermions and two "
            "external bosons")
    slots = {externals[j].slot: j for j in fermions}
    if set(slots) != {"bar", "field"}:
        raise ValueError(
            "enumerate_line_diagrams: the two external fermions must fill one "
            "ψ̄ (bar) and one ψ (field) slot — fermion number is not conserved "
            f"for {[str(externals[j].symbol) for j in fermions]}")
    return slots["bar"], slots["field"], bosons


def enumerate_line_diagrams(externals, vertices, kin, species,
                            conjugates=None, fermion_mass=None,
                            mass_of=None, width_of=None):
    """Every tree diagram of a two-fermion + two-boson 2→2 process.

    **Fermion exchange** — both bosons attach to the fermion line.  The line
    runs from the field slot ``Y`` to the bar slot ``X``:
    ``ψ̄_X Γ₁ S_F(q) Γ₂ ψ_Y`` with vertex 2 ``(F̄, Y, B)`` next to the field
    slot and vertex 1 ``(X̄, F, B')`` next to the bar slot, for both
    assignments of the two bosons and every declared intermediate species
    ``F``.

    External bosons follow the Wick rule the four-fermion enumerator uses for
    mediators: an incoming particle ``B`` is annihilated by the field ``B``,
    an outgoing one is created by ``B† = conjugates[B]`` — so that is the
    boson symbol its vertex must carry.  The propagator momentum is taken
    along the fermion arrow: ``q = p_in(Y) + p_in(B)`` with ``p_in = +k`` for
    an incoming and ``−k`` for an outgoing leg.

    **Boson exchange** — the line has one vertex ``(X̄, Y, M)``; the boson
    ``M`` propagates to a bosonic three-point vertex carrying ``M†`` and the
    two external bosons' fields (``VVV`` from
    :meth:`~feynlag.lagrangian.Model.gauge_vertices`, ``VVS``, ``VSS``,
    ``SSS``).
    Each leg enters that vertex with its own incoming momentum — ``p_in`` for
    an external boson, ``p_in(X) + p_in(Y)`` for the mediator — which is all
    that fixes the triple-gauge sign
    (:mod:`~feynlag.pheno.boson_vertices`).

    Args:
        externals: four :class:`ExternalFermion`/:class:`ExternalBoson`, in
            the order of the kinematics' momenta ``k1..k4``.
        vertices: :class:`~feynlag.pheno.vertices.DecayVertex` list.
        kin: a :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
        species: the declared :class:`~feynlag.pheno.particles.DiracParticle`\\ s
            (candidates for the exchanged fermion).
        conjugates: ``{boson: antiboson}`` for charged bosons.
        fermion_mass: optional callable ``DiracParticle -> mass`` (default:
            its ``mass``).
        mass_of, width_of: callables ``boson -> mass`` / ``-> width`` for a
            boson mediator (``mass_of`` is required once a boson-exchange
            diagram exists; ``width_of`` defaults to ``0``).

    Returns:
        list of :class:`LineDiagramInfo`.

    Raises:
        ValueError: the externals are not two fermions (one bar, one field
            slot) and two bosons, or a vertex that would complete a diagram
            has an intermediate fermion leg no declared species claims —
            dropping it silently would lose a diagram.
        NotImplementedError: a charged external vector with no ``VVV``
            vertex among ``vertices`` (its triple-gauge diagrams would be
            missing).
    """
    conjugates = dict(conjugates or {})
    fermion_mass = fermion_mass or (lambda dp: dp.mass)
    width_of = width_of or (lambda boson: sp.S.Zero)
    bar, field, bosons = _split_externals(externals)
    heads = (kin.k1, kin.k2, kin.k3, kin.k4)
    by_antiparticle = {dp.antiparticle: dp for dp in species}
    by_particle = {dp.particle: dp for dp in species}
    bar_symbol = externals[bar].particle.antiparticle
    field_symbol = externals[field].particle.particle

    def p_in(j):
        return (sp.S.One if externals[j].incoming else -sp.S.One, heads[j])

    def vertex_boson(j):
        symbol = externals[j].symbol
        return symbol if externals[j].incoming else conjugates.get(symbol, symbol)

    def carries(vertex, j):
        wanted = "FFV" if externals[j].spin == 1 else "FFS"
        return (vertex.vertex_type == wanted
                and vertex.particles[2] == vertex_boson(j))

    def factor(vertex, j):
        name = _external_index(j) if externals[j].spin == 1 else None
        return ("G", name, vertex.g_left, vertex.g_right)

    _require_gauge_vertices(externals, bosons, vertices, conjugates)

    out = []
    for first, second in (bosons, bosons[::-1]):
        q = (p_in(field), p_in(second))
        q2 = sp.expand(sum(ca * cb * kin.dot(ha, hb)
                           for ca, ha in q for cb, hb in q))
        channel = _CHANNELS[frozenset({field, second})]
        for vertex_2 in vertices:
            if not (carries(vertex_2, second)
                    and vertex_2.particles[1] == field_symbol):
                continue
            exchanged = by_antiparticle.get(vertex_2.particles[0])
            if exchanged is None:
                raise ValueError(
                    f"enumerate_line_diagrams: the vertex {vertex_2!r} joins "
                    f"{field_symbol} to the fermion leg "
                    f"{vertex_2.particles[0]!r}, which no declared "
                    f"DiracParticle claims; declare that species (use "
                    f"right=None for a left-handed-only one) so its "
                    f"exchange diagram is not lost")
            for vertex_1 in vertices:
                if not (carries(vertex_1, first)
                        and vertex_1.particles[0] == bar_symbol
                        and vertex_1.particles[1] == exchanged.particle):
                    continue
                term = fermion_exchange_term(
                    factor(vertex_1, first), factor(vertex_2, second), q, q2,
                    fermion_mass(exchanged))
                out.append(LineDiagramInfo(
                    channel=channel, mediator=exchanged.particle,
                    kind="fermion", vertices=(vertex_1, vertex_2),
                    terms=(term,)))
        # the mirror-image check: a vertex next to the bar slot whose other
        # fermion leg is undeclared would also complete a diagram
        for vertex_1 in vertices:
            if (carries(vertex_1, first)
                    and vertex_1.particles[0] == bar_symbol
                    and vertex_1.particles[1] not in by_particle):
                raise ValueError(
                    f"enumerate_line_diagrams: the vertex {vertex_1!r} joins "
                    f"{bar_symbol} to the fermion leg "
                    f"{vertex_1.particles[1]!r}, which no declared "
                    f"DiracParticle claims; declare that species (use "
                    f"right=None for a left-handed-only one)")

    # ---- boson exchange: line vertex -> propagator -> bosonic 3-point vertex
    q = (p_in(bar), p_in(field))
    q2 = sp.expand(sum(ca * cb * kin.dot(ha, hb)
                       for ca, ha in q for cb, hb in q))
    channel = _CHANNELS[frozenset({bar, field})]
    wanted = [vertex_boson(j) for j in bosons]
    for boson_vertex in vertices:
        if boson_vertex.vertex_type not in ("VVV", "VVS", "VSS", "SSS"):
            continue
        rest = list(boson_vertex.particles)
        try:
            for symbol in wanted:
                rest.remove(symbol)
        except ValueError:
            continue
        (at_vertex,) = rest
        line_boson = conjugates.get(at_vertex, at_vertex)
        for line_vertex in vertices:
            if not (line_vertex.vertex_type in ("FFV", "FFS")
                    and line_vertex.particles[0] == bar_symbol
                    and line_vertex.particles[1] == field_symbol
                    and line_vertex.particles[2] == line_boson):
                continue
            spin = 1 if line_vertex.vertex_type == "FFV" else 0
            n_vectors = spin + sum(externals[j].spin == 1 for j in bosons)
            if boson_vertex.vertex_type != {3: "VVV", 2: "VVS", 1: "VSS",
                                            0: "SSS"}[n_vectors]:
                continue
            if mass_of is None:
                raise ValueError(
                    "enumerate_line_diagrams: a boson-exchange diagram "
                    "exists; pass `mass_of` for its mediator")
            line_index = fresh_name("r") if spin == 1 else None
            vertex_index = fresh_name("s") if spin == 1 else None
            # legs of the bosonic vertex: (field symbol, index, momentum in)
            legs = [(at_vertex, vertex_index, q)]
            for j in bosons:
                legs.append((vertex_boson(j),
                             _external_index(j) if externals[j].spin == 1 else None,
                             (p_in(j),)))
            pieces = _boson_vertex_pieces(boson_vertex, legs)
            terms = boson_exchange_terms(
                ("G", line_index, line_vertex.g_left, line_vertex.g_right),
                spin, q, q2, mass_of(line_boson), width_of(line_boson),
                line_index, vertex_index, pieces)
            out.append(LineDiagramInfo(
                channel=channel, mediator=line_boson, kind="boson",
                vertices=(line_vertex, boson_vertex), terms=terms))
    return out


def _boson_vertex_pieces(vertex, legs):
    """The bosonic vertex as loose tensors, its legs ``(symbol, index,
    momentum)`` matched to ``vertex.particles`` by field symbol."""
    if vertex.vertex_type == "SSS":
        return sss_tensor(vertex.coupling)
    if vertex.vertex_type == "VVS":
        index_a, index_b = [index for _, index, _ in legs if index is not None]
        return vvs_tensor(vertex.coupling, index_a, index_b)
    if vertex.vertex_type == "VSS":
        (index,) = [index for _, index, _ in legs if index is not None]
        momenta = {symbol: combo for symbol, idx, combo in legs if idx is None}
        return vss_tensor(vertex.coupling, index, momenta)
    remaining = list(legs)
    ordered = []
    for symbol in vertex.particles:
        match = next(leg for leg in remaining if leg[0] == symbol)
        remaining.remove(match)
        ordered.append((match[1], match[2]))
    return vvv_tensor(vertex.coupling, ordered)


def _require_gauge_vertices(externals, bosons, vertices, conjugates):
    """Raise for a charged external vector when no ``VVV`` vertex was given.

    A charged vector always has a triple-gauge coupling to the photon, and
    the gauge self-couplings are not part of the extractor's output (they
    come from :meth:`~feynlag.lagrangian.Model.gauge_vertices`).  Without
    them the enumeration would return the fermion-exchange diagrams alone —
    for ``e⁺e⁻→W⁺W⁻`` the ν exchange without the γ/Z s channel that cancels
    its high-energy growth: a wrong cross section, not a partial one.
    """
    if any(vertex.vertex_type == "VVV" for vertex in vertices):
        return
    for j in bosons:
        symbol = externals[j].symbol
        if externals[j].spin == 1 and conjugates.get(symbol, symbol) != symbol:
            raise NotImplementedError(
                f"enumerate_line_diagrams: the charged external vector "
                f"{symbol} couples through a triple-gauge vertex, but no VVV "
                f"vertex was supplied; pass `gauge_groups=[...]` to "
                f"ScatteringCalculator (or add Model.gauge_vertices() "
                f"output to `vertices`)")


def line_amplitude(externals, infos, kin):
    """The :class:`~feynlag.pheno.line.LineAmplitude` of enumerated diagrams."""
    bar, field, bosons = _split_externals(externals)
    heads = (kin.k1, kin.k2, kin.k3, kin.k4)
    masses = (kin.m1, kin.m2, kin.m3, kin.m4)

    def leg(j):
        return Leg(heads[j], masses[j], anti=externals[j].anti)

    vectors = tuple(VectorLeg(_external_index(j), heads[j], masses[j])
                    for j in bosons if externals[j].spin == 1)
    return LineAmplitude(out=leg(bar), inn=leg(field),
                         diagrams=tuple(info.terms for info in infos),
                         vectors=vectors)

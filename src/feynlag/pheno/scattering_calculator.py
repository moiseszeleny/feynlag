"""2→2 cross sections straight from a feynlag model (Tier 3).

:class:`ScatteringCalculator` is the scattering counterpart of
:class:`~feynlag.pheno.calculator.DecayCalculator`: it extracts the three-leg
vertices once (:func:`~feynlag.pheno.vertices.collect_decay_vertices`, with
the ``P_L``/``P_R`` split and the :class:`~feynlag.pheno.particles.DiracParticle`
merge already done), and :meth:`ScatteringCalculator.process` enumerates
every tree diagram of a four-fermion process
(:func:`~feynlag.pheno.topology.enumerate_diagrams`) and returns a
:class:`ScatteringProcess` whose ``|M|²`` sums them **with interference**
(:meth:`~feynlag.pheno.diagrams.Amplitude.squared`).

Scope, enforced rather than documented only:

- external fermions are declared ``DiracParticle``\\ s; a process is either
  four fermions, or **two fermions and two bosons** (Tier 4): fermion
  exchange (``e⁺e⁻→γγ``, ``e⁺e⁻→ZZ``, Compton) and boson exchange through a
  ``VVV``/``VVS``/``VSS``/``SSS`` vertex (``e⁺e⁻→W⁺W⁻``, ``e⁺e⁻→Zh``,
  ``e⁺e⁻→φ⁺φ⁻``).  The
  triple-gauge vertices are only collected when ``gauge_groups`` is given;
  a charged external vector without them raises rather than returning the
  fermion-exchange diagrams alone;
- **coloured fermions** in four-fermion processes (Tier 5, stage 5a): pass
  ``colour_group`` and ``gluon``, and declare a quark as a
  :class:`~feynlag.pheno.particles.DiracParticle` on **one** colour component
  with ``color=N_c``.  The gluon becomes one mediator (``calc.gluon``) with a
  colour-stripped coupling, and the colour factors come from
  :mod:`~feynlag.pheno.colour`.  External gluons raise (stage 5b);
- mediators are the declared bosons in **unitary gauge**: leave Goldstones out
  of ``boson_fields`` (a massive vector's propagator already carries their
  physics as ``q_μq_ν/m²``).
"""

from dataclasses import replace as _replace

import sympy as sp

from ..operators import momentum
from .kinematics import TwoToTwoKinematics
from .particles import ExternalState, expand_particles
from .scattering import (
    cross_section, differential_cross_section, forward_backward_asymmetry,
)
from .colour import colour_matrix, four_fermion_colour_tensor
from .diagrams import Amplitude
from .topology import (
    ExternalBoson, ExternalFermion, enumerate_diagrams,
    enumerate_line_diagrams, line_amplitude,
)
from .vertices import DecayVertex, collect_decay_vertices

__all__ = ["ScatteringCalculator", "ScatteringProcess"]


class ScatteringCalculator:
    """Tree-level 2→2 four-fermion cross sections for a built model.

    Args:
        model: a :class:`~feynlag.lagrangian.Model` in its physical basis.
        particles: :class:`~feynlag.pheno.particles.DiracParticle` list — the
            external fermions (and any fermion you want resolvable).
        masses: ``{boson: mass}`` for every mediator.  An undeclared mediator
            raises ``KeyError`` instead of silently becoming massless.
        widths: ``{boson: Γ}`` (Breit–Wigner widths); absent means ``0``.
        boson_fields: the physical boson symbols to extract vertices over.
        fermion_sectors: Lagrangian sectors with fermion bilinears.
        conjugate_map: passed through to the bosonic extractor.
        conjugates: ``{boson: antiboson}`` for charged mediators, e.g.
            ``{Wp: Wm, Wm: Wp}``; unlisted bosons are self-conjugate.
        parameters: optional :class:`~feynlag.parameters.ParameterSet` for
            numeric evaluation (``process(..., numeric=True)``).
        gauge_groups: the broken non-abelian groups whose triple-gauge
            (``VVV``) vertices to collect through
            :meth:`~feynlag.lagrangian.Model.gauge_vertices`, e.g.
            ``[SU2L]`` — needed for ``e⁺e⁻→W⁺W⁻``.  An unbroken group
            (colour) must not be listed.
        colour_group: the unbroken colour group (e.g. ``SU3c``), needed for
            coloured external fermions.
        gluon: its :class:`~feynlag.fields.GaugeBoson` field.  The
            calculator extracts over its components itself and exposes one
            mediator symbol, :attr:`gluon`; do not list the components in
            ``boson_fields``.
    """

    def __init__(self, model, particles, masses, widths=None, boson_fields=(),
                 fermion_sectors=("gauge",), conjugate_map=None,
                 conjugates=None, parameters=None, gauge_groups=None,
                 colour_group=None, gluon=None):
        self.model = model
        self.particles = list(particles)
        self.masses = dict(masses)
        self.widths = dict(widths or {})
        self.boson_fields = list(boson_fields)
        self.fermion_sectors = tuple(fermion_sectors)
        self.conjugate_map = conjugate_map
        self.conjugates = dict(conjugates or {})
        self.parameters = parameters
        self.gauge_groups = list(gauge_groups or [])
        if (colour_group is None) != (gluon is None):
            raise ValueError(
                "ScatteringCalculator: pass `colour_group` and `gluon` "
                "together")
        self.colour_group = colour_group
        self._gluon_components = (list(gluon.components) if gluon is not None
                                  else [])
        #: the single gluon mediator symbol (``None`` without a colour group)
        self.gluon = sp.Symbol(gluon.name) if gluon is not None else None
        self.particle_map, _, _ = expand_particles(self.particles)
        self._vertices = None

    def vertices(self):
        """All three-leg vertices, extracted once and cached: the
        extractor's (:func:`~feynlag.pheno.vertices.collect_decay_vertices`)
        plus, with ``gauge_groups``, the triple-gauge ones."""
        if self._vertices is None:
            vertices = collect_decay_vertices(
                self.model, self.boson_fields + self._gluon_components,
                fermion_sectors=self.fermion_sectors,
                conjugate_map=self.conjugate_map,
                particle_map=self.particle_map)
            vertices = self._at_zero_fields(vertices)
            if self.colour_group is not None:
                vertices = self._colour_stripped(vertices)
            if self.gauge_groups:
                wanted = set(self.boson_fields)
                for vertex in self.model.gauge_vertices(
                        groups=self.gauge_groups, include=("VVV",)):
                    if set(vertex.particles) <= wanted:
                        vertices.append(DecayVertex(
                            particles=tuple(vertex.particles),
                            vertex_type="VVV", coupling=vertex.coupling,
                            meta={"source": "gauge"}))
            self._vertices = vertices
        return self._vertices

    def _at_zero_fields(self, vertices):
        """Bosonic vertices evaluated at zero for every field that is not an
        extraction leg.

        The extractor treats a field missing from ``boson_fields`` as a
        parameter, so with the Goldstones left out (unitary gauge) a
        four-leg term ``A h W⁺ G⁻`` shows up as a three-leg "vertex"
        ``A h W⁺`` with coupling ``∝ G⁻``.  A vertex is a derivative **at
        zero fields**: setting the stray field symbols to zero removes those
        and leaves the genuine three-point couplings untouched.
        """
        fields = set(self.model.spin_map(conjugate_map=self.conjugate_map))
        out = []
        for vertex in vertices:
            if vertex.vertex_type.startswith("FF"):
                out.append(vertex)
                continue
            # a VSS coupling's momentum tags p(φ) name their legs: they are
            # not field dependence, so hide them from the substitution
            coupling = sp.sympify(vertex.coupling)
            tags = {tag: sp.Dummy() for tag in coupling.atoms(momentum)}
            hidden = coupling.xreplace(tags)
            stray = [sym for sym in hidden.free_symbols if sym in fields]
            if stray:
                hidden = hidden.subs({sym: 0 for sym in stray})
                if hidden == 0:
                    continue
                vertex = _replace(vertex, coupling=hidden.xreplace(
                    {dummy: tag for tag, dummy in tags.items()}))
            out.append(vertex)
        return out

    def colour_rep(self, particle):
        """The colour-group rep label of a declared species (``None`` for a
        colour singlet), read from the model field owning its left leg."""
        if self.colour_group is None:
            return None
        for field in self.model.fields:
            if any(particle.left == comp
                   for comp in getattr(field, "components", ())):
                return field.reps.get(self.colour_group)
        return None

    def _generators(self, particle):
        """The colour generators of a coloured species, checked against its
        declared ``color``."""
        rep = self.colour_rep(particle)
        if rep is None or self.colour_group.rep_dim(rep) != particle.color:
            raise ValueError(
                f"ScatteringCalculator: {particle.particle} is declared with "
                f"color={particle.color}, but its field's "
                f"{self.colour_group.name} rep is {rep!r}")
        return self.colour_group.generators(rep)

    def _colour_stripped(self, vertices):
        """Replace the colour-component gluon vertices by one vertex per
        coloured species on the single mediator :attr:`gluon`.

        A species is declared on one colour component ``r``, so the
        extractor reports ``(q̄, q, G_a)`` with ``g·T^a[r, r]`` for the
        diagonal generators only.  Dividing by ``T^a[r, r]`` must give the
        same ``g`` for each of them — that fixes ``r`` and is a check that
        the vertex really has the ``g T^a`` colour structure.
        """
        components = set(self._gluon_components)
        out = [v for v in vertices
               if not (v.vertex_type.startswith("FF")
                       and v.particles[2] in components)]
        for dp in self.particles:
            if dp.color == 1:
                continue
            T = self._generators(dp)
            found = {}
            for v in vertices:
                if (v.vertex_type == "FFV" and v.particles[2] in components
                        and v.particles[0] == dp.antiparticle
                        and v.particles[1] == dp.particle):
                    found[self._gluon_components.index(v.particles[2])] = v
            if not found:
                continue
            stripped = None
            for r in range(dp.color):
                expected = {a for a, Ta in enumerate(T)
                            if Ta.is_diagonal() and Ta[r, r] != 0}
                if set(found) != expected:
                    continue
                values = {(sp.simplify(found[a].g_left / T[a][r, r]),
                           sp.simplify(found[a].g_right / T[a][r, r]))
                          for a in expected}
                if len(values) == 1:
                    (stripped,) = values
                    break
            if stripped is None:
                raise ValueError(
                    f"ScatteringCalculator: the {self.colour_group.name} "
                    f"couplings of {dp.particle} are not g·T^a for one "
                    f"colour component; declare the DiracParticle on a "
                    f"single colour component of its field(s)")
            out.append(DecayVertex(
                particles=(dp.antiparticle, dp.particle, self.gluon),
                vertex_type="FFV", g_left=stripped[0], g_right=stripped[1],
                meta={"source": "colour", "bar": dp.antiparticle,
                      "field": dp.particle, "boson": self.gluon}))
        return out

    def _colour_matrix(self, externals, diagrams):
        """The colour matrix of a four-fermion process, or ``None`` when
        every external is a colour singlet."""
        dims = [ext.particle.color for ext in externals]
        if all(d == 1 for d in dims):
            return None
        tensors = []
        for info in diagrams:
            generators = None
            if info.mediator == self.gluon:
                generators = tuple(
                    self._generators(externals[bar].particle)
                    for bar, _ in info.lines)
            tensors.append(four_fermion_colour_tensor(
                info.lines, dims, generators))
        return colour_matrix(tensors)

    def mass_of(self, boson):
        if self.gluon is not None and boson == self.gluon:
            return sp.S.Zero
        if boson in self.masses:
            return sp.sympify(self.masses[boson])
        partner = self.conjugates.get(boson)
        if partner in self.masses:          # W⁻ has the mass declared for W⁺
            return sp.sympify(self.masses[partner])
        raise KeyError(
            f"no mass declared for the mediator {boson!r}; add it to the "
            f"calculator's `masses` mapping (0 for a photon)")

    def width_of(self, boson):
        return sp.sympify(self.widths.get(boson, 0))

    def values(self, extra=None):
        """``{symbol: number}`` from ``parameters`` plus ``extra``."""
        values = {}
        if self.parameters is not None:
            values.update(self.parameters.numeric())
        if extra:
            values.update({sp.sympify(k): v for k, v in extra.items()})
        return values

    def _external(self, symbol, incoming):
        symbol = sp.sympify(symbol)
        for dp in self.particles:
            if symbol in (dp.particle, dp.antiparticle):
                if dp.color != 1 and self.colour_group is None:
                    raise NotImplementedError(
                        f"ScatteringCalculator: {symbol} is coloured "
                        f"(N_c={dp.color}); pass `colour_group` and `gluon` "
                        f"to the calculator")
                return ExternalFermion(dp, anti=(symbol == dp.antiparticle),
                                       incoming=incoming)
        if self.gluon is not None and (symbol == self.gluon
                                       or symbol in self._gluon_components):
            raise NotImplementedError(
                f"ScatteringCalculator: an external gluon ({symbol}) needs "
                f"physical polarization sums and the ggg vertex — stage 5b "
                f"of Tier 5 of the scattering roadmap")
        if symbol in self.boson_fields or symbol in self.conjugates:
            spin = self.model.spin_map(conjugate_map=self.conjugate_map).get(symbol)
            if spin is None:
                spin = self.model.spin_map(
                    conjugate_map=self.conjugate_map).get(
                        self.conjugates.get(symbol, symbol))
            if spin not in (0, 1):
                raise KeyError(
                    f"cannot determine the spin of the external boson "
                    f"{symbol!r} from the model's spin map (got {spin!r})")
            return ExternalBoson(symbol, spin, self.mass_of(symbol), incoming)
        raise KeyError(
            f"{symbol!r} is neither a declared DiracParticle (or its "
            f"antiparticle) nor one of the calculator's `boson_fields`")

    def process(self, incoming, outgoing, numeric=False, extra=None):
        """The 2→2 process ``incoming → outgoing``.

        Args:
            incoming, outgoing: 2-tuples of particle/antiparticle symbols
                (momenta ``k1, k2`` and ``k3, k4`` in that order; the CM
                angle ``θ`` is between ``k1`` and ``k3``).
            numeric: substitute the parameter point (``parameters`` +
                ``extra``) into every coupling, mass and width *before* the
                Dirac algebra — much faster than carrying symbolic
                ``√(g²+g'²)``-type couplings, at the price of a number-valued
                ``|M|²(s,t)``.
            extra: additional ``{symbol: value}`` for ``numeric``.

        Returns:
            :class:`ScatteringProcess`.
        """
        externals = ([self._external(p, True) for p in incoming]
                     + [self._external(p, False) for p in outgoing])
        values = self.values(extra) if numeric else {}

        def num(expr):
            expr = sp.sympify(expr)
            return expr.subs(values) if values else expr

        kin = TwoToTwoKinematics(*(num(_mass(ext)) for ext in externals))
        vertices = self.vertices()
        if values:
            vertices = [_substituted(v, values) for v in vertices]
        n_bosons = sum(isinstance(ext, ExternalBoson) for ext in externals)
        amplitude = None
        colour = None
        if n_bosons and any(isinstance(ext, ExternalFermion)
                            and ext.particle.color != 1 for ext in externals):
            raise NotImplementedError(
                "ScatteringCalculator: coloured fermions with external "
                "bosons are stage 5b of Tier 5 of the scattering roadmap")
        if n_bosons == 0:
            diagrams = enumerate_diagrams(
                externals, vertices, kin,
                mass_of=lambda b: num(self.mass_of(b)),
                width_of=lambda b: num(self.width_of(b)),
                conjugates=self.conjugates)
            colour = self._colour_matrix(externals, diagrams)
        elif n_bosons == 2:
            diagrams = enumerate_line_diagrams(
                externals, vertices, kin, self.particles,
                conjugates=self.conjugates,
                fermion_mass=lambda dp: num(dp.mass),
                mass_of=lambda b: num(self.mass_of(b)),
                width_of=lambda b: num(self.width_of(b)))
            if diagrams:
                amplitude = line_amplitude(externals, diagrams, kin)
        else:
            raise NotImplementedError(
                f"ScatteringCalculator: a process with {n_bosons} external "
                f"boson(s) and {4 - n_bosons} external fermion(s) is not "
                f"implemented (four fermions, or two fermions and two "
                f"bosons)")
        if not diagrams:
            raise ValueError(
                f"no tree diagram connects {tuple(map(str, incoming))} -> "
                f"{tuple(map(str, outgoing))} through the extracted vertices "
                f"(check `boson_fields`, `fermion_sectors` and `particles`)")
        return ScatteringProcess(externals, kin, diagrams,
                                 amplitude=amplitude, colour=colour)


def _colour_dim(external):
    if isinstance(external, ExternalFermion):
        return external.particle.color
    return 1


def _mass(external):
    """The mass of an external fermion (its species') or boson."""
    if isinstance(external, ExternalFermion):
        return external.particle.mass
    return external.mass


def _substituted(vertex, values):
    return _replace(vertex,
                   g_left=sp.sympify(vertex.g_left).subs(values),
                   g_right=sp.sympify(vertex.g_right).subs(values),
                   coupling=sp.sympify(vertex.coupling).subs(values))


class ScatteringProcess:
    """One 2→2 process: its diagrams, ``Σ|M|²`` and cross sections.

    Attributes:
        externals: the four :class:`~feynlag.pheno.topology.ExternalFermion`.
        kinematics: the :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`.
        diagrams: the :class:`~feynlag.pheno.topology.DiagramInfo` list.
        initial_states: the two :class:`~feynlag.pheno.particles.ExternalState`
            averaged over.
        symmetry_factor: ``1/2`` for identical final-state particles.
        colour: the colour matrix ``C[d][d']`` of a process with coloured
            externals (:mod:`~feynlag.pheno.colour`), else ``None``.
    """

    def __init__(self, externals, kinematics, diagrams, amplitude=None,
                 colour=None):
        self.externals = list(externals)
        self.kinematics = kinematics
        self.diagrams = list(diagrams)
        self._amplitude = amplitude
        self.colour = colour
        self.initial_states = tuple(
            ExternalState(ext.symbol, kinematics_mass, ext.spin,
                          color=_colour_dim(ext))
            for ext, kinematics_mass in zip(self.externals[:2],
                                            (kinematics.m1, kinematics.m2)))
        identical = self.externals[2].symbol == self.externals[3].symbol
        self.symmetry_factor = sp.Rational(1, 2) if identical else sp.S.One
        self._squared = None

    def __repr__(self):
        names = [str(e.symbol) for e in self.externals]
        return (f"ScatteringProcess({names[0]} {names[1]} -> {names[2]} "
                f"{names[3]}: {self.diagrams})")

    def amplitude(self):
        """The four-fermion :class:`~feynlag.pheno.diagrams.Amplitude`, or
        the :class:`~feynlag.pheno.line.LineAmplitude` of a process with two
        external bosons."""
        if self._amplitude is not None:
            return self._amplitude
        return Amplitude(tuple(d.diagram for d in self.diagrams))

    def squared(self):
        """``Σ|M|²`` (spin- and colour-summed, not averaged), as a function
        of ``kinematics.s``/``.t``; cached."""
        if self._squared is None:
            if self.colour is not None:
                self._squared = self.amplitude().squared(
                    self.kinematics, colour=self.colour)
            else:
                self._squared = self.amplitude().squared(self.kinematics)
        return self._squared

    def differential_cross_section(self, variable="cos"):
        """``dσ/dcosθ`` (default) or ``dσ/dt``, averaged over the initial
        spins with the identical-particle factor applied."""
        return differential_cross_section(
            self.squared(), self.kinematics, self.initial_states,
            variable=variable, symmetry_factor=self.symmetry_factor)

    def cross_section(self, cos_range=None):
        """``σ``, optionally inside an angular acceptance ``cosθ ∈ cos_range``
        (see :func:`~feynlag.pheno.scattering.cross_section`)."""
        return cross_section(self.squared(), self.kinematics,
                             self.initial_states,
                             symmetry_factor=self.symmetry_factor,
                             cos_range=cos_range)

    def forward_backward_asymmetry(self):
        return forward_backward_asymmetry(self.squared(), self.kinematics)

    def numeric_cross_section(self, s, cos_range=(-1, 1), backend="auto"):
        """``σ`` at a numeric ``s`` by 1-D quadrature over ``cosθ``
        (:func:`~feynlag.pheno.integrate.quad_1d`), in the inverse-square
        units of ``s``.

        The fast route once every coupling is a number (``process(...,
        numeric=True)``): SymPy's symbolic ``t``-integral of a ``|M|²`` with
        several complex Breit–Wigner denominators in different channels (a
        t-channel Z next to an s-channel Z, as in Bhabha) is slow, and
        pointless when only a number is wanted.  A t-channel photon needs a
        ``cos_range`` away from ``cosθ = 1``.
        """
        from .integrate import quad_1d
        cos = sp.Symbol("_sigma_cos", real=True)
        kin = self.kinematics
        density = (self.differential_cross_section(variable="cos")
                   .subs(kin.t, kin.t_of_cos(cos)).subs(kin.s, s))
        f = sp.lambdify(cos, density, "numpy")
        lo, hi = (float(c) for c in cos_range)
        return quad_1d(lambda c: complex(f(c)).real, lo, hi, backend=backend)

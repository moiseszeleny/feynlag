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

- external states are Dirac fermions declared as ``DiracParticle``\\ s;
  external bosons (``e⁺e⁻→W⁺W⁻``) need derivative couplings — Tier 4;
- **colour-singlet** external fermions only — a coloured one raises, since
  the colour-flow bookkeeping (``N_c`` per closed line, ``T^a`` traces for
  gluon exchange) is Tier 5's;
- mediators are the declared bosons in **unitary gauge**: leave Goldstones out
  of ``boson_fields`` (a massive vector's propagator already carries their
  physics as ``q_μq_ν/m²``).
"""

import sympy as sp

from .kinematics import TwoToTwoKinematics
from .particles import ExternalState, expand_particles
from .scattering import (
    cross_section, differential_cross_section, forward_backward_asymmetry,
)
from .diagrams import Amplitude
from .topology import ExternalFermion, enumerate_diagrams
from .vertices import collect_decay_vertices

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
    """

    def __init__(self, model, particles, masses, widths=None, boson_fields=(),
                 fermion_sectors=("gauge",), conjugate_map=None,
                 conjugates=None, parameters=None):
        self.model = model
        self.particles = list(particles)
        self.masses = dict(masses)
        self.widths = dict(widths or {})
        self.boson_fields = list(boson_fields)
        self.fermion_sectors = tuple(fermion_sectors)
        self.conjugate_map = conjugate_map
        self.conjugates = dict(conjugates or {})
        self.parameters = parameters
        self.particle_map, _, _ = expand_particles(self.particles)
        self._vertices = None

    def vertices(self):
        """All three-leg vertices, extracted once and cached."""
        if self._vertices is None:
            self._vertices = collect_decay_vertices(
                self.model, self.boson_fields,
                fermion_sectors=self.fermion_sectors,
                conjugate_map=self.conjugate_map,
                particle_map=self.particle_map)
        return self._vertices

    def mass_of(self, boson):
        if boson in self.masses:
            return sp.sympify(self.masses[boson])
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
                if dp.color != 1:
                    raise NotImplementedError(
                        f"ScatteringCalculator: {symbol} is coloured "
                        f"(N_c={dp.color}); coloured external states need the "
                        f"colour-flow layer of Tier 5 of the scattering roadmap")
                return ExternalFermion(dp, anti=(symbol == dp.antiparticle),
                                       incoming=incoming)
        raise KeyError(
            f"{symbol!r} is not a declared DiracParticle (or its "
            f"antiparticle); external bosons are Tier 4 of the scattering "
            f"roadmap")

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

        kin = TwoToTwoKinematics(*(num(ext.particle.mass) for ext in externals))
        vertices = self.vertices()
        if values:
            vertices = [_substituted(v, values) for v in vertices]
        diagrams = enumerate_diagrams(
            externals, vertices, kin,
            mass_of=lambda b: num(self.mass_of(b)),
            width_of=lambda b: num(self.width_of(b)),
            conjugates=self.conjugates)
        if not diagrams:
            raise ValueError(
                f"no tree diagram connects {tuple(map(str, incoming))} -> "
                f"{tuple(map(str, outgoing))} through the extracted vertices "
                f"(check `boson_fields`, `fermion_sectors` and `particles`)")
        return ScatteringProcess(externals, kin, diagrams)


def _substituted(vertex, values):
    from dataclasses import replace
    return replace(vertex,
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
    """

    def __init__(self, externals, kinematics, diagrams):
        self.externals = list(externals)
        self.kinematics = kinematics
        self.diagrams = list(diagrams)
        self.initial_states = tuple(
            ExternalState(ext.symbol, kinematics_mass, sp.Rational(1, 2))
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
        return Amplitude(tuple(d.diagram for d in self.diagrams))

    def squared(self):
        """``Σ|M|²`` (spin-summed, not averaged), as a function of
        ``kinematics.s``/``.t``; cached."""
        if self._squared is None:
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

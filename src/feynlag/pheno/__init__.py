"""Phenomenology: decay widths, branching ratios and 2→2 cross sections.

Turns the Feynman rules feynlag extracts into observables at tree level:
1→2 (and off-shell 1→3) partial widths, total widths and branching ratios,
and 2→2 cross sections with every diagram and its interference.  The path
is covariant: Dirac traces, ε algebra and polarization sums reduced to
on-shell invariants, checked against explicit-matrix oracles in the tests.

This namespace is the user API::

    from feynlag.pheno import DecayCalculator, DiracParticle, ScatteringCalculator

    decays = DecayCalculator.from_model(model, particles=[e, nu])
    print(decays.decay_table("Z", unit="MeV"))

    calc = ScatteringCalculator.from_model(model, [e, nu], widths={Z: 2.4952},
                                           aliases={"e-": e.particle,
                                                    "e+": e.antiparticle})
    ww = calc.process("e- e+ > w+ w-")
    print(ww.summary(sqrt_s=200, unit="pb"))

- calculators: :class:`DecayCalculator`, :class:`ScatteringCalculator` and
  what they return (:class:`DecayChannel`, :class:`ScatteringProcess`);
- particles: :class:`DiracParticle`, :class:`ExternalState`;
- results and names: :class:`ProcessSummary`, :class:`DecayTable`,
  :class:`DecayRow`, :class:`ParticleNames`, :func:`diagram_row`;
- units: :func:`cross_section_in`, :func:`energy_in`, :func:`eta_acceptance`
  and the conversion tables;
- closed-form widths: the loop-induced ``h→gg/γγ/Zγ`` and the off-shell
  ``S→VV*``.

The building blocks (vertex collection, kinematics, traces, diagrams,
colour tensors, ``|M|²`` → observables) are in :mod:`feynlag.pheno.engine`.
Importing one of those from here still works but raises a
``DeprecationWarning``.
"""

import importlib
import warnings

from .calculator import DecayCalculator, DecayChannel
from .loop import higgs_gammagamma_width, higgs_gg_width, higgs_zgamma_width
from .names import ParticleNames
from .offshell import scalar_offshell_vv_width
from .particles import DiracParticle, ExternalState
from .scattering_calculator import ScatteringCalculator, ScatteringProcess
from .summary import DecayRow, DecayTable, ProcessSummary, diagram_row
from .units import (
    CROSS_SECTION_UNITS, ENERGY_UNITS, HBARC2_GEV2_MB, cross_section_in,
    energy_in, eta_acceptance,
)

__all__ = [
    # calculators
    "DecayCalculator", "DecayChannel", "ScatteringCalculator",
    "ScatteringProcess",
    # particles
    "DiracParticle", "ExternalState",
    # results and names
    "DecayRow", "DecayTable", "ParticleNames", "ProcessSummary", "diagram_row",
    # units
    "CROSS_SECTION_UNITS", "ENERGY_UNITS", "HBARC2_GEV2_MB",
    "cross_section_in", "energy_in", "eta_acceptance",
    # closed-form widths
    "higgs_gammagamma_width", "higgs_gg_width", "higgs_zgamma_width",
    "scalar_offshell_vv_width",
]


def __getattr__(name):
    """An engine name imported from here: still served, with a warning."""
    # not `from . import engine`: that looks the name up on this package
    # first, i.e. calls __getattr__ again
    engine = importlib.import_module(f"{__name__}.engine")
    if name in engine.__all__:
        warnings.warn(
            f"feynlag.pheno.{name} moved to feynlag.pheno.engine; import it "
            f"from there (from feynlag.pheno.engine import {name})",
            DeprecationWarning, stacklevel=2)
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    import pkgutil
    submodules = [m.name for m in pkgutil.iter_modules(__path__)]
    return sorted(set(__all__) | set(submodules))

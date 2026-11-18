"""The engine under :mod:`feynlag.pheno`: the building blocks the
calculators are made of.

:mod:`feynlag.pheno` itself exports what a user calls without knowing the
engine (the calculators, ``DiracParticle``, result tables, units and the
closed-form widths).  This module re-exports everything else: the vertex
collection, kinematics, the Dirac/ε/tensor algebra, the 1→2 and 1→3 squared
amplitudes, the 2→2 diagram and amplitude objects, colour tensors and the
``|M|²`` → observable functions.  It is what the tests, the independent
oracles and low-level work import::

    from feynlag.pheno.engine import SpinorChain, TwoToTwoKinematics

Every name here is also importable from the submodule that defines it
(``feynlag.pheno.diagrams.SpinorChain`` is the same object).  Importing one
of these names from :mod:`feynlag.pheno` directly still works but is
deprecated.
"""
# --- vertices and particles
from .vertices import (
    at_zero_fields, classify_gamma, collect_decay_vertices, DecayVertex,
)
from .particles import (
    expand_particles,
)


# --- kinematics and phase space
from .kinematics import (
    is_allowed, kallen, ThreeBodyKinematics, two_body_momentum,
    two_body_phase_space, TwoBodyKinematics, TwoToTwoKinematics,
)
from .integrate import (
    dalitz_integral, have_scipy,
)


# --- Dirac, Lorentz and tensor algebra
from .lorentz import (
    contract_to_dots, dirac_trace, reduce_projectors,
)
from .epsilon import (
    assert_epsilon_single_vanishes, epsilon_pair_tensor,
    gamma5_trace_coefficient,
)
from .contraction import (
    contract_tensors, contract_trace,
)


# --- 1→2 and 1→3 squared amplitudes, propagators, loop form factors
from .amplitudes import (
    amplitude_squared, ffs_squared, ffv_squared, polarization_sum,
    spin_sum, sss_squared, vvs_squared,
)
from .calculator import (
    partial_width,
)
from .propagator import (
    breit_wigner, fermion_propagator_phase, propagator_denominator,
    scalar_propagator, vector_propagator, vector_propagator_numerator,
)
from .offshell import (
    offshell_scalar_vv_width, scalar_vv_s12_integral, scalar_vv_squared,
)
from .loop import (
    A_half, A_one,
)


# --- 2→2 diagrams and amplitudes
from .diagrams import (
    Amplitude, BosonPropagator, ChainVertex, Diagram, Leg, SpinorChain,
)
from .line import (
    boson_exchange_terms, fermion_exchange_term, LineAmplitude, LineTerm,
    VectorLeg,
)
from .boson import (
    BosonAmplitude, BosonTerm,
)
from .boson_vertices import (
    sss_tensor, VSS_KAPPA, vss_tensor, vvs_tensor, VVV_KAPPA, vvv_tensor,
    vvvv_tensor,
)
from .topology import (
    boson_amplitude, BosonDiagramInfo, DiagramInfo,
    enumerate_boson_diagrams, enumerate_diagrams, enumerate_line_diagrams,
    ExternalBoson, ExternalFermion, line_amplitude, LineDiagramInfo,
)


# --- colour
from .colour import (
    colour_matrix, four_boson_colour_tensor, four_fermion_colour_tensor,
    line_colour_tensor,
)


# --- |M|² → observables
from .scattering import (
    average_factor, cross_section, differential_cross_section,
    ffs_s_channel_squared, ffv_s_channel_squared,
    forward_backward_asymmetry,
)

__all__ = [
    "A_half",
    "A_one",
    "Amplitude",
    "amplitude_squared",
    "assert_epsilon_single_vanishes",
    "at_zero_fields",
    "average_factor",
    "boson_amplitude",
    "boson_exchange_terms",
    "BosonAmplitude",
    "BosonDiagramInfo",
    "BosonPropagator",
    "BosonTerm",
    "breit_wigner",
    "ChainVertex",
    "classify_gamma",
    "collect_decay_vertices",
    "colour_matrix",
    "contract_tensors",
    "contract_to_dots",
    "contract_trace",
    "cross_section",
    "dalitz_integral",
    "DecayVertex",
    "Diagram",
    "DiagramInfo",
    "differential_cross_section",
    "dirac_trace",
    "enumerate_boson_diagrams",
    "enumerate_diagrams",
    "enumerate_line_diagrams",
    "epsilon_pair_tensor",
    "expand_particles",
    "ExternalBoson",
    "ExternalFermion",
    "fermion_exchange_term",
    "fermion_propagator_phase",
    "ffs_s_channel_squared",
    "ffs_squared",
    "ffv_s_channel_squared",
    "ffv_squared",
    "forward_backward_asymmetry",
    "four_boson_colour_tensor",
    "four_fermion_colour_tensor",
    "gamma5_trace_coefficient",
    "have_scipy",
    "is_allowed",
    "kallen",
    "Leg",
    "line_amplitude",
    "line_colour_tensor",
    "LineAmplitude",
    "LineDiagramInfo",
    "LineTerm",
    "offshell_scalar_vv_width",
    "partial_width",
    "polarization_sum",
    "propagator_denominator",
    "reduce_projectors",
    "scalar_propagator",
    "scalar_vv_s12_integral",
    "scalar_vv_squared",
    "spin_sum",
    "SpinorChain",
    "sss_squared",
    "sss_tensor",
    "ThreeBodyKinematics",
    "two_body_momentum",
    "two_body_phase_space",
    "TwoBodyKinematics",
    "TwoToTwoKinematics",
    "vector_propagator",
    "vector_propagator_numerator",
    "VectorLeg",
    "VSS_KAPPA",
    "vss_tensor",
    "vvs_squared",
    "vvs_tensor",
    "VVV_KAPPA",
    "vvv_tensor",
    "vvvv_tensor",
]

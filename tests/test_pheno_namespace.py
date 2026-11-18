"""The ``feynlag.pheno`` namespace: user API on top, building blocks in
``feynlag.pheno.engine``.

What is pinned:

- the top-level ``__all__`` is the curated user API, ``engine.__all__`` the
  rest, disjoint, and together they cover every name exported before the
  split (plus ``at_zero_fields``), so nothing was dropped;
- each name resolves to the object its defining submodule holds;
- an engine name imported from the top level still works but raises a
  ``DeprecationWarning`` naming ``feynlag.pheno.engine``; user names,
  submodules and ``import *`` raise none;
- ``dir(feynlag.pheno)`` shows the user API, not the engine.
"""

import importlib
import subprocess
import sys
import warnings

import pytest

import feynlag.pheno as pheno
from feynlag.pheno import engine

#: the curated user API
USER_API = {
    "CROSS_SECTION_UNITS", "DecayCalculator", "DecayChannel", "DecayRow",
    "DecayTable", "DiracParticle", "ENERGY_UNITS", "ExternalState",
    "HBARC2_GEV2_MB", "ParticleNames", "ProcessSummary",
    "ScatteringCalculator", "ScatteringProcess", "cross_section_in",
    "diagram_row", "energy_in", "eta_acceptance", "higgs_gammagamma_width",
    "higgs_gg_width", "higgs_zgamma_width", "scalar_offshell_vv_width"
}

#: ``feynlag.pheno.__all__`` before the split (feynlag 0.4.x and PR #59)
PRE_SPLIT = {
    "A_half", "A_one", "Amplitude", "BosonAmplitude", "BosonDiagramInfo",
    "BosonPropagator", "BosonTerm", "CROSS_SECTION_UNITS", "ChainVertex",
    "DecayCalculator", "DecayChannel", "DecayRow", "DecayTable",
    "DecayVertex", "Diagram", "DiagramInfo", "DiracParticle",
    "ENERGY_UNITS", "ExternalBoson", "ExternalFermion", "ExternalState",
    "HBARC2_GEV2_MB", "Leg", "LineAmplitude", "LineDiagramInfo",
    "LineTerm", "ParticleNames", "ProcessSummary", "ScatteringCalculator",
    "ScatteringProcess", "SpinorChain", "ThreeBodyKinematics",
    "TwoBodyKinematics", "TwoToTwoKinematics", "VSS_KAPPA", "VVV_KAPPA",
    "VectorLeg", "amplitude_squared", "assert_epsilon_single_vanishes",
    "average_factor", "boson_amplitude", "boson_exchange_terms",
    "breit_wigner", "classify_gamma", "collect_decay_vertices",
    "colour_matrix", "contract_tensors", "contract_to_dots",
    "contract_trace", "cross_section", "cross_section_in",
    "dalitz_integral", "diagram_row", "differential_cross_section",
    "dirac_trace", "energy_in", "enumerate_boson_diagrams",
    "enumerate_diagrams", "enumerate_line_diagrams", "epsilon_pair_tensor",
    "eta_acceptance", "expand_particles", "fermion_exchange_term",
    "fermion_propagator_phase", "ffs_s_channel_squared", "ffs_squared",
    "ffv_s_channel_squared", "ffv_squared", "forward_backward_asymmetry",
    "four_boson_colour_tensor", "four_fermion_colour_tensor",
    "gamma5_trace_coefficient", "have_scipy", "higgs_gammagamma_width",
    "higgs_gg_width", "higgs_zgamma_width", "is_allowed", "kallen",
    "line_amplitude", "line_colour_tensor", "offshell_scalar_vv_width",
    "partial_width", "polarization_sum", "propagator_denominator",
    "reduce_projectors", "scalar_offshell_vv_width", "scalar_propagator",
    "scalar_vv_s12_integral", "scalar_vv_squared", "spin_sum",
    "sss_squared", "sss_tensor", "two_body_momentum",
    "two_body_phase_space", "vector_propagator",
    "vector_propagator_numerator", "vss_tensor", "vvs_squared",
    "vvs_tensor", "vvv_tensor", "vvvv_tensor"
}


def test_user_api_is_the_curated_list():
    assert set(pheno.__all__) == USER_API
    assert len(pheno.__all__) == len(USER_API)


def test_split_is_disjoint_and_complete():
    assert not set(pheno.__all__) & set(engine.__all__)
    assert set(pheno.__all__) | set(engine.__all__) == \
        PRE_SPLIT | {"at_zero_fields"}


@pytest.mark.parametrize("module", [pheno, engine])
def test_every_exported_name_resolves(module):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in module.__all__:
            assert getattr(module, name) is not None, name


def test_engine_names_are_the_submodule_objects():
    from feynlag.pheno import diagrams, vertices
    assert engine.SpinorChain is diagrams.SpinorChain
    assert engine.at_zero_fields is vertices.at_zero_fields
    for name in engine.__all__:
        obj = getattr(engine, name)
        home = getattr(obj, "__module__", None)
        if isinstance(home, str) and home.startswith("feynlag.pheno."):
            assert getattr(importlib.import_module(home), name) is obj, name


def test_engine_name_from_the_top_level_warns():
    with pytest.warns(DeprecationWarning,
                      match=r"feynlag\.pheno\.SpinorChain moved to "
                            r"feynlag\.pheno\.engine"):
        from feynlag.pheno import SpinorChain
    assert SpinorChain is engine.SpinorChain
    with pytest.warns(DeprecationWarning, match="feynlag.pheno.engine"):
        assert pheno.TwoToTwoKinematics is engine.TwoToTwoKinematics


def test_user_names_and_submodules_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from feynlag.pheno import ScatteringCalculator   # noqa: F401
        from feynlag.pheno import boson_vertices          # noqa: F401
        import feynlag.pheno.lorentz                      # noqa: F401
        assert pheno.DecayCalculator is not None


def test_star_import_is_the_user_api_and_silent():
    """In a fresh interpreter, so no earlier import has bound anything."""
    code = (
        "import warnings; warnings.simplefilter('error')\n"
        "ns = {}\n"
        "exec('from feynlag.pheno import *', ns)\n"
        "import feynlag.pheno as p\n"
        "got = {k for k in ns if not k.startswith('__')}\n"
        "assert got == set(p.__all__), got ^ set(p.__all__)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_name_still_raises():
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        pheno.not_a_name
    with pytest.raises(ImportError):
        from feynlag.pheno import not_a_name               # noqa: F401


def test_dir_shows_the_user_api_not_the_engine():
    listing = set(dir(pheno))
    assert USER_API <= listing
    assert "engine" in listing                       # the submodule
    assert not (set(engine.__all__) - USER_API) & listing

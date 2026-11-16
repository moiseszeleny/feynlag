"""Model.validate() umbrella: aggregation, skip logic, failure propagation.

The individual checks (invariance, anomalies, UFO round-trip) have their own
dedicated test files; here we pin only the umbrella behaviour — that
``validate`` runs the applicable checks, skips the inapplicable ones, and
reports ``ok`` iff every check that ran passed.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, Model, Scalar, SU2, SU3, U1, WeylFermion,
    ValidationReport,
)


def _groups():
    gs = ExternalParameter("g_s", 1.2, positive=True)
    gw = ExternalParameter("gw", 0.65, positive=True)
    g1 = ExternalParameter("g1", 0.36, positive=True)
    return SU3("SU3c", coupling=gs), SU2("SU2L", coupling=gw), \
        U1("U1Y", coupling=g1)


def _sm_fermions(SU3c, SU2L, U1Y):
    R = sp.Rational

    def reps(su3, su2, y):
        d = {U1Y: y}
        if su3 != 1:
            d[SU3c] = su3
        if su2 != 1:
            d[SU2L] = su2
        return d

    return [
        WeylFermion("QL", reps=reps(3, 2, R(1, 6)), chirality="L", nflavors=3),
        WeylFermion("uR", reps=reps(3, 1, R(2, 3)), chirality="R", nflavors=3),
        WeylFermion("dR", reps=reps(3, 1, R(-1, 3)), chirality="R", nflavors=3),
        WeylFermion("LL", reps=reps(1, 2, R(-1, 2)), chirality="L", nflavors=3),
        WeylFermion("eR", reps=reps(1, 1, -1), chirality="R", nflavors=3),
    ]


def test_validate_aggregates_passing_checks():
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate()
    assert isinstance(report, ValidationReport)
    assert report.ok, report.summary()
    assert "invariance" in report.checks
    assert report.checks["anomalies"] is not None      # ran, did not skip
    assert report.checks["anomalies"].ok


def test_validate_flags_anomaly_failure():
    """Drop eR: hypercharge anomalies no longer cancel -> umbrella fails."""
    SU3c, SU2L, U1Y = _groups()
    fields = _sm_fermions(SU3c, SU2L, U1Y)[:-1]        # no eR
    model = Model("SM_noER", gauge_groups=[SU3c, SU2L, U1Y], fields=fields)
    report = model.validate()
    assert not report.ok
    assert not report.checks["anomalies"].ok
    with pytest.raises(ValueError, match="anomalies"):
        report.raise_on_failure()


def test_validate_skips_anomalies_without_fermions():
    SU3c, SU2L, U1Y = _groups()
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    model = Model("scalars", gauge_groups=[SU3c, SU2L, U1Y], fields=[H])
    report = model.validate()
    assert report.checks["anomalies"] is None          # skipped
    assert report.ok
    assert "skipped" in report.summary()


def test_validate_can_disable_checks():
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate(invariance=False, anomalies=True)
    assert "invariance" not in report.checks
    assert "anomalies" in report.checks


def test_validate_omits_charge_checks_without_charges():
    """Backward compatible: no charge/hermiticity entries unless charges given."""
    SU3c, SU2L, U1Y = _groups()
    model = Model("SM", gauge_groups=[SU3c, SU2L, U1Y],
                  fields=_sm_fermions(SU3c, SU2L, U1Y))
    report = model.validate()
    assert "charge_conservation" not in report.checks
    assert "hermiticity_pairing" not in report.checks


# ---------------------------------------------- parameters check + summary

def _higgs_model(declare_lam=True):
    from feynlag import InternalParameter, Lagrangian, inner

    _, SU2L, U1Y = _groups()
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    lam = ExternalParameter("lam", 0.13)
    mu2 = InternalParameter("mu2", unit_dim=2)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.H0: v})
    HdH = inner(H, H)
    L = Lagrangian().add(-(-mu2 * HdH + lam * HdH**2), sector="potential")
    params = [SU2L.coupling, U1Y.coupling, v, mu2] + ([lam] if declare_lam else [])
    return Model("H", gauge_groups=[SU2L, U1Y], fields=[H], parameters=params,
                 lagrangian=L), lam


def test_check_parameters_flags_an_undeclared_symbol():
    model, lam = _higgs_model(declare_lam=True)
    assert model.check_parameters().ok
    assert model.undeclared_symbols() == []

    model, lam = _higgs_model(declare_lam=False)
    report = model.check_parameters()
    assert not report.ok and report.undeclared == [lam.s]
    with pytest.raises(ValueError, match="lam"):
        report.raise_on_failure()
    # opt-in: the default validate() is unchanged by an undeclared symbol
    assert "parameters" not in model.validate().checks
    assert model.validate().ok
    strict = model.validate(parameters=True)
    assert not strict.ok and not strict.checks["parameters"].ok


def test_check_parameters_ignores_fields_indices_and_lists_matrices():
    from feynlag import Lagrangian, fermion_gauge_current, yukawa

    _, SU2L, U1Y = _groups()
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    H = Scalar("H", reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
               component_names=["Gp", "H0"])
    H.expand_vev({H.H0: v})
    LL = WeylFermion("LL", reps={SU2L: 2, U1Y: sp.Rational(-1, 2)},
                     chirality="L", nflavors=3)
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3)
    W, B = SU2L.bosons("W"), U1Y.bosons("B")
    Ye = sp.IndexedBase("Ye")
    i = sp.Symbol("i", integer=True)
    L = Lagrangian().add(yukawa(Ye, LL, H, eR), sector="yukawa")
    L.add(fermion_gauge_current(LL, i) + fermion_gauge_current(eR, i),
          sector="yukawa")
    model = Model("lep", gauge_groups=[SU2L, U1Y], fields=[H, W, B, LL, eR],
                  parameters=[SU2L.coupling, U1Y.coupling, v], lagrangian=L)
    report = model.check_parameters()
    # fermion legs, gauge bosons, flavour and Lorentz indices are not flagged
    assert report.ok, report
    assert report.flavour_matrices == [Ye]

    # an undeclared VEV symbol is caught even though it is not in L itself
    S = Scalar("S", reps={}, real=True)
    S.expand_vev({S.components[0]: sp.Symbol("vS", positive=True)})
    model2 = Model("s", fields=[S])
    assert [str(s) for s in model2.undeclared_symbols()] == ["vS"]


def test_summary_tables():
    from feynlag import ZN

    model, _ = _higgs_model()
    SU2L, U1Y = model.gauge_groups
    LL = WeylFermion("LL", reps={SU2L: 2, U1Y: sp.Rational(-1, 2)},
                     chirality="L", nflavors=3)
    W = SU2L.bosons("W")
    Z2 = ZN("Z2", 2)
    Z2.assign(1, model.fields[0])
    model.fields += [W, LL]
    model.discrete_groups.append(Z2)

    summary = model.summary()
    assert summary.field_header == ["field", "type", "SU2L", "U1Y", "Z2",
                                    "flavours", "components", "VEV"]
    rows = {r[0]: r for r in summary.field_rows}
    assert rows["H"] == ["H", "scalar", "2", "1/2", "1", "", "Gp, H0",
                         "⟨H0⟩ = v/√2"]
    assert rows["W"][1:5] == ["vector", "adj", "0", ""]
    assert rows["LL"][1:6] == ["fermion L", "2", "-1/2", "", "3"]
    params = {r[0]: r for r in summary.param_rows}
    assert params["v"] == ["v", "external", "246.0", "1"]
    assert params["mu2"][1] == "internal"

    text = str(summary)
    assert text.startswith("Model: H") and "Parameters" in text
    md = summary._repr_markdown_()
    assert "| field | type | SU2L | U1Y | Z2 |" in md

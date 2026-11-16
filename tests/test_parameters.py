"""Parameter classes and dependency-resolution tests."""

import sympy as sp
import pytest

from feynlag import ExternalParameter, InternalParameter, ParameterSet


def test_external_parameter_basics():
    v = ExternalParameter("v", 246.0, positive=True, unit_dim=1)
    assert v.s.is_positive
    assert v.value == 246.0
    assert v.unit_dim == 1


def test_parameter_arithmetic():
    v = ExternalParameter("v", 246.0, positive=True)
    lam = ExternalParameter("lam", 0.13)
    expr = lam * v**2 / 2
    assert expr == lam.s * v.s**2 / 2


def test_internal_defined_later():
    lam = ExternalParameter("lam", 0.13)
    v = ExternalParameter("v", 246.0, positive=True)
    mu2 = InternalParameter("mu2", unit_dim=2)
    mu2.define(lam.s * v.s**2)
    assert mu2.expr == lam.s * v.s**2


def test_dependency_order_and_resolve():
    v = ExternalParameter("v", 246.0, positive=True)
    mh = ExternalParameter("mh", 125.0, positive=True)
    lam = InternalParameter("lam", None)
    mu2 = InternalParameter("mu2", None)
    # mu2 depends on lam; lam depends only on externals
    lam.define(mh.s**2 / (2 * v.s**2))
    mu2.define(lam.s * v.s**2)

    ps = ParameterSet(v, mh, lam, mu2)
    order = ps.dependency_order()
    assert order.index(ps["lam"]) < order.index(ps["mu2"])

    resolved = ps.resolve()
    assert sp.simplify(resolved[mu2.s] - mh.s**2 / 2) == 0

    numeric = ps.numeric()
    assert numeric[mu2.s] == pytest.approx(125.0**2 / 2)


def test_cycle_detection():
    a = InternalParameter("a")
    b = InternalParameter("b")
    a.define(b.s + 1)
    b.define(a.s + 1)
    ps = ParameterSet(a, b)
    with pytest.raises(ValueError, match="cycle"):
        ps.dependency_order()


def test_undefined_internal_raises():
    a = InternalParameter("a")
    ps = ParameterSet(a)
    with pytest.raises(ValueError, match="no.*defining"):
        ps.dependency_order()


def test_unknown_dependency_raises():
    a = InternalParameter("a", sp.Symbol("nowhere"))
    ps = ParameterSet(a)
    with pytest.raises(ValueError, match="not in"):
        ps.dependency_order()


def test_duplicate_name_raises():
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSet(ExternalParameter("x", 1.0), ExternalParameter("x", 2.0))


def test_parameter_set_factories_declare_once():
    """``P.external`` / ``P.internal`` create, register and return."""
    P = ParameterSet()
    v = P.external("v", 246.0, positive=True, unit_dim=1)
    lam = P.external("lam", 0.13)
    mu2 = P.internal("mu2", lam * v**2, unit_dim=2)
    assert isinstance(v, ExternalParameter) and isinstance(mu2, InternalParameter)
    assert [p.name for p in P] == ["v", "lam", "mu2"]
    assert P["v"] is v and P[mu2.s] is mu2
    assert P.resolve()[mu2.s] == lam.s * v.s**2
    with pytest.raises(ValueError, match="duplicate"):
        P.external("v", 1.0)


def test_parameters_work_in_expressions_without_dot_s():
    """A Parameter is usable wherever its symbol is (no ``.s`` needed)."""
    from feynlag import Bilinear, diracPR

    v = ExternalParameter("v", 246.0, positive=True)
    y = ExternalParameter("y", 0.1, real=False)
    psi, psibar = sp.IndexedBase("psi"), sp.IndexedBase("psibar")
    i = sp.Symbol("i", integer=True)
    B = Bilinear(psibar[i], diracPR, psi[i])
    assert y * B == y.s * B and B * y == y.s * B
    assert -v == -v.s and 1 / v == 1 / v.s and 2 ** v == 2 ** v.s
    assert sp.sqrt(v) == sp.sqrt(v.s)
    assert sp.conjugate(y) == sp.conjugate(y.s)
    assert sp.conjugate(v) == v.s                      # positive ⇒ real
    assert sp.Matrix([[v, y]]) == sp.Matrix([[v.s, y.s]])

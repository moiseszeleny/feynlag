"""Model parameters: external (experiment-fixed) vs internal (derived).

- :class:`ExternalParameter`: fixed by experiment (``v``, ``m_h``, ``g``…),
  carries a numeric value for benchmarks / UFO param cards.
- :class:`InternalParameter`: derived from other parameters via a SymPy
  expression (tadpole solutions, mixing angles, inverted quartics).  The
  expression may be assigned *later* by the pipeline.
- :class:`ParameterSet`: dependency DAG over internal parameters with
  topological ordering, full symbolic resolution and numeric evaluation —
  exactly what a UFO ``parameters.py`` needs.
"""

import sympy as sp

from .texsymbol import tex_symbol

__all__ = ["Parameter", "ExternalParameter", "InternalParameter",
           "ParameterPoint", "ParameterSet"]


class Parameter:
    """Wraps a SymPy Symbol with physics metadata.

    Args:
        name: symbol name.
        tex: LaTeX string (defaults to ``name``).  When given, the symbol
            carries it (:class:`~feynlag.texsymbol.TexSymbol`), so
            ``sympy.latex`` of any expression renders it.
        real: SymPy assumption (default True — most Lagrangian parameters).
        positive: SymPy assumption (use for VEVs, masses; enables ``sqrt``
            simplifications per CONVENTIONS.md).
        unit_dim: mass dimension, for dimensional checks.
    """

    def __init__(self, name, tex=None, real=True, positive=False, unit_dim=0):
        self.name = name
        self.tex = tex if tex is not None else name
        assumptions = {"positive": True} if positive else {"real": True} if real else {}
        self.symbol = tex_symbol(name, tex, **assumptions)
        self.unit_dim = unit_dim

    @property
    def s(self):
        """The underlying SymPy symbol (shorthand for expressions)."""
        return self.symbol

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"

    def _repr_latex_(self):
        tex = self.tex if self.tex != self.name else sp.latex(self.symbol)
        return f"$\\displaystyle {tex}$"

    # Allow parameters to be used directly in SymPy arithmetic.
    def _sympy_(self):
        return self.symbol

    def __add__(self, other):
        return self.symbol + sp.sympify(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.symbol - sp.sympify(other)

    def __rsub__(self, other):
        return sp.sympify(other) - self.symbol

    def __mul__(self, other):
        return self.symbol * sp.sympify(other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.symbol / sp.sympify(other)

    def __rtruediv__(self, other):
        return sp.sympify(other) / self.symbol

    def __pow__(self, other):
        return self.symbol ** sp.sympify(other)

    def __rpow__(self, other):
        return sp.sympify(other) ** self.symbol

    def __neg__(self):
        return -self.symbol


class ExternalParameter(Parameter):
    """Parameter fixed by experiment; input of the model.

    ``value`` may be a float or a SymPy expression in *other external*
    parameters' numeric values (evaluated at benchmark time).
    """

    nature = "external"

    def __init__(self, name, value=None, tex=None, real=True, positive=False,
                 unit_dim=0):
        super().__init__(name, tex=tex, real=real, positive=positive,
                         unit_dim=unit_dim)
        self.value = value


class InternalParameter(Parameter):
    """Parameter derived from others via ``expr`` (assignable later).

    Examples: ``μ₁₁²`` from a tadpole condition, ``λ₁`` inverted from
    ``m_h`` and ``v``, a mixing angle from a ``tan 2θ`` condition.
    """

    nature = "internal"

    def __init__(self, name, expr=None, tex=None, real=True, positive=False,
                 unit_dim=0):
        super().__init__(name, tex=tex, real=real, positive=positive,
                         unit_dim=unit_dim)
        self.expr = sp.sympify(expr) if expr is not None else None

    def define(self, expr):
        """Assign (or reassign) the defining expression."""
        self.expr = sp.sympify(expr)
        return self


class ParameterSet:
    """Collection of parameters with dependency resolution.

    Internal parameters may depend on other internal parameters; the
    dependency graph must be acyclic.  :meth:`dependency_order` returns
    internals in evaluation order (what UFO's ``parameters.py`` requires).
    """

    def __init__(self, *params):
        self._by_name = {}
        self._by_symbol = {}
        self.add(*params)

    def add(self, *params):
        for p in params:
            if p.name in self._by_name and self._by_name[p.name] is not p:
                raise ValueError(f"duplicate parameter name: {p.name}")
            self._by_name[p.name] = p
            self._by_symbol[p.symbol] = p
        return self

    def external(self, name, value=None, **kwargs):
        """Create an :class:`ExternalParameter`, register it and return it.

        Declares a parameter exactly once: ``ye = P.external("ye", 0.01)``
        instead of building it and listing it again in
        ``Model(parameters=[...])`` (pass the set itself, ``parameters=P``).
        """
        param = ExternalParameter(name, value, **kwargs)
        self.add(param)
        return param

    def internal(self, name, expr=None, **kwargs):
        """Create an :class:`InternalParameter`, register it and return it."""
        param = InternalParameter(name, expr, **kwargs)
        self.add(param)
        return param

    def __getitem__(self, key):
        if isinstance(key, sp.Symbol):
            return self._by_symbol[key]
        return self._by_name[key]

    def __contains__(self, key):
        return key in self._by_name or key in self._by_symbol

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self):
        return len(self._by_name)

    def _repr_latex_(self):
        rows = []
        for p in self:
            tex = p.tex if p.tex != p.name else sp.latex(p.symbol)
            value = p.value if isinstance(p, ExternalParameter) else p.expr
            value_tex = sp.latex(sp.sympify(value)) if value is not None else "-"
            rows.append(f"{p.name} & {tex} & {p.nature} & {value_tex} \\\\\n\\hline\n")
        table = (r"\begin{array}{|c|c|c|c|}" + "\n\\hline\n"
                 + r"\textbf{name} & \textbf{symbol} & \textbf{nature} & "
                 r"\textbf{value/expr} \\" + "\n\\hline\n"
                 + "".join(rows) + r"\end{array}")
        return f"$\\displaystyle {table}$"

    @property
    def externals(self):
        return [p for p in self if isinstance(p, ExternalParameter)]

    @property
    def internals(self):
        return [p for p in self if isinstance(p, InternalParameter)]

    def dependency_order(self):
        """Topologically sort internal parameters by their dependencies.

        Returns the internals in an order where every parameter appears
        after all internal parameters occurring in its ``expr``.

        Raises:
            ValueError: on an undefined internal (``expr is None``), a
                dependency cycle, or a dependency on an unknown symbol.
        """
        internals = self.internals
        internal_symbols = {p.symbol for p in internals}
        external_symbols = {p.symbol for p in self.externals}

        deps = {}
        for p in internals:
            if p.expr is None:
                raise ValueError(f"internal parameter {p.name} has no "
                                 f"defining expression")
            free = p.expr.free_symbols
            unknown = free - internal_symbols - external_symbols
            if unknown:
                raise ValueError(
                    f"internal parameter {p.name} depends on symbols not in "
                    f"the ParameterSet: {sorted(unknown, key=str)}")
            deps[p.symbol] = free & internal_symbols

        ordered, resolved = [], set()
        remaining = dict(deps)
        while remaining:
            ready = [s for s, d in remaining.items() if d <= resolved]
            if not ready:
                cycle = sorted(remaining, key=str)
                raise ValueError(f"dependency cycle among internal "
                                 f"parameters: {cycle}")
            for s in sorted(ready, key=str):
                ordered.append(self._by_symbol[s])
                resolved.add(s)
                del remaining[s]
        return ordered

    def resolve(self):
        """Map every internal symbol to an expression in externals only."""
        resolution = {}
        for p in self.dependency_order():
            resolution[p.symbol] = p.expr.subs(resolution)
        return resolution

    def _external_overrides(self, overrides):
        """``{symbol: value}`` from ``{name | Symbol | Parameter: value}``."""
        out = {}
        for key, value in (overrides or {}).items():
            if isinstance(key, Parameter):
                key = key.symbol
            if key not in self:
                raise KeyError(
                    f"override of {key!r}: no such parameter in the "
                    f"ParameterSet (known: {sorted(self._by_name)})")
            param = self[key]
            if not isinstance(param, ExternalParameter):
                raise ValueError(
                    f"override of {param.name}: it is an internal (derived) "
                    f"parameter, fixed by its defining expression; override "
                    f"the external parameters it depends on instead")
            out[param.symbol] = value
        return out

    def numeric(self, overrides=None):
        """Numeric value of every parameter (externals must have values).

        Args:
            overrides: optional ``{name | Symbol | Parameter: value}`` for
                **external** parameters, used instead of their stored
                ``value``; the internals are re-derived from them.  Nothing
                is mutated, so one ``ParameterSet`` serves many points (see
                :class:`ParameterPoint`).

        Returns:
            dict ``{symbol: float/complex}``.

        Raises:
            KeyError: an override names no parameter of the set.
            ValueError: an override targets an internal parameter, or an
                external parameter has no value.
        """
        over = self._external_overrides(overrides)
        values = {}
        for p in self.externals:
            value = over.get(p.symbol, p.value)
            if value is None:
                raise ValueError(f"external parameter {p.name} has no value")
            values[p.symbol] = sp.sympify(value)
        # substitute progressively so external expressions of externals work
        for s in list(values):
            values[s] = values[s].subs(values)
        for p in self.dependency_order():
            values[p.symbol] = p.expr.subs(values)
        return {s: complex(v) if v.is_complex and not v.is_real else float(v)
                for s, v in values.items()}


class ParameterPoint:
    """One numeric point of a model: a :class:`ParameterSet`, overrides of
    its external parameters, and values for symbols outside the set.

    ::

        point = ParameterPoint(model.parameters, {"gw": 0.65, "v": 246.0},
                               extra={m_b: 4.18})
        point.values()          # {symbol: number}, internals re-derived
        point["gw"]             # by name, Symbol or Parameter
        point.replace(v=250.0)  # a new point; this one is unchanged

    The point is immutable and never touches the parameters' stored values,
    so a scan is ``[point.replace(v=x) for x in xs]``.  ``numeric()`` is an
    alias of :meth:`values`, which makes a point usable wherever a
    ``ParameterSet`` is passed as ``parameters=``.

    Args:
        parameters: the :class:`ParameterSet` (e.g. ``model.parameters``).
        overrides: ``{name | Symbol | Parameter: value}`` for externals.
        extra: ``{symbol: value}`` for symbols that are not parameters of
            the set (a fermion-mass symbol, a width).
    """

    def __init__(self, parameters, overrides=None, extra=None):
        if isinstance(parameters, ParameterPoint):
            overrides = {**parameters._overrides, **(
                parameters.parameters._external_overrides(overrides))}
            extra = {**parameters._extra, **{
                sp.sympify(k): v for k, v in (extra or {}).items()}}
            parameters = parameters.parameters
        self.parameters = parameters
        # normalised to symbols and validated here, so a bad name fails at
        # construction and not at the first evaluation
        self._overrides = dict(parameters._external_overrides(overrides))
        self._extra = {sp.sympify(k): v for k, v in (extra or {}).items()}
        self._values = None

    @property
    def overrides(self):
        """``{symbol: value}`` of the overridden external parameters."""
        return dict(self._overrides)

    @property
    def extra(self):
        """``{symbol: value}`` of the non-parameter symbols."""
        return dict(self._extra)

    def values(self):
        """``{symbol: number}``: every parameter, plus ``extra``."""
        if self._values is None:
            values = self.parameters.numeric(self._overrides)
            values.update(self._extra)
            self._values = values
        return dict(self._values)

    numeric = values

    def replace(self, overrides=None, extra=None, **named):
        """A new point with further overrides (``named``: by parameter
        name).  This point is left unchanged."""
        return ParameterPoint(self, {**(overrides or {}), **named}, extra)

    def __getitem__(self, key):
        values = self.values()
        if isinstance(key, Parameter):
            key = key.symbol
        if isinstance(key, str):
            if key in self.parameters:
                return values[self.parameters[key].symbol]
            for sym in self._extra:
                if str(sym) == key:
                    return values[sym]
            raise KeyError(key)
        return values[sp.sympify(key)]

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def _rows(self):
        values = self.values()
        rows = []
        for p in self.parameters:
            if isinstance(p, ExternalParameter):
                kind = ("override" if p.symbol in self._overrides
                        else "external")
            else:
                kind = "derived"
            rows.append((p.name, values[p.symbol], kind))
        rows += [(str(s), v, "extra") for s, v in self._extra.items()]
        return rows

    def __repr__(self):
        body = ", ".join(f"{n}={v:.6g}" for n, v, _ in self._rows())
        return f"ParameterPoint({body})"

    def _repr_markdown_(self):
        lines = ["| parameter | value | kind |", "|---|---|---|"]
        lines += [f"| `{n}` | {v:.6g} | {k} |" for n, v, k in self._rows()]
        return "\n".join(lines)

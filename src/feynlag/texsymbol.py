"""Symbols that carry their own LaTeX name.

A field component or parameter declared with ``tex=`` / ``component_tex=``
is built as a :class:`TexSymbol`, so plain ``sympy.latex(expr)`` renders it
in physics notation (``H^0``, ``\\nu_L``) with no caller-side name map.
``str()``/``.name`` stay the raw name, so the UFO export, ``lambdify`` and
the Python code printers are unaffected.

The tex string is part of the symbol's identity: ``TexSymbol('H0', 'H^0')``
is *not* equal to ``Symbol('H0')``.  Without a tex the constructors build
plain ``Symbol``\\ s exactly as before (see :func:`tex_symbol`).
"""

import sympy as sp

__all__ = ["TexSymbol", "tex_symbol"]


class TexSymbol(sp.Symbol):
    """A ``Symbol`` whose LaTeX rendering is the given ``tex`` string."""

    def __new__(cls, name, tex, **assumptions):
        # Symbol.__new__ caches on (name, assumptions) only, which would hand
        # back an instance carrying another call's tex — build uncached.
        cls._sanitize(assumptions, cls)
        obj = sp.Symbol.__xnew__(cls, name, **assumptions)
        obj._tex = str(tex)
        return obj

    def __getnewargs_ex__(self):
        return ((self.name, self._tex), self._assumptions_orig)

    def _hashable_content(self):
        return super()._hashable_content() + (self._tex,)

    def _latex(self, printer):
        # an explicit ``symbol_names`` entry still wins, as for a plain Symbol
        return printer._settings.get("symbol_names", {}).get(self, self._tex)

    def _sympyrepr(self, printer):
        args = [repr(self.name), repr(self._tex)]
        args += [f"{k}={v}" for k, v in sorted(self._assumptions_orig.items())]
        return f"TexSymbol({', '.join(args)})"


def tex_symbol(name, tex=None, **assumptions):
    """A :class:`TexSymbol` when ``tex`` is given, else a plain ``Symbol``."""
    if tex is None:
        return sp.Symbol(name, **assumptions)
    return TexSymbol(name, tex, **assumptions)

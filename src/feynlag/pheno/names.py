"""Process names: resolve ``"e- e+ > w+ w-"`` to the model's symbols.

Two kinds of name are accepted:

- the **symbol name** of every external a calculator can resolve (``e``,
  ``ebar``, ``Wp``, ``Wm``, ``A``), matched exactly;
- **aliases**, ``{name: symbol}``, matched case-insensitively (MadGraph writes
  ``w+``, ``a``, ``z``).  ``to_physical_basis`` registers the SM bosons'
  MadGraph names on ``model.physical.aliases``; fermion names (``e-``/``e+``)
  are the user's to choose, since which member of a pair is "the particle"
  is a convention.

An exact symbol name always wins, and an alias that names a *different*
symbol than an existing symbol name raises, so a name means one thing.
"""

import difflib
import re

import sympy as sp

__all__ = ["ParticleNames"]

_ARROW = re.compile(r"\s*(?:->|→|>)\s*")


class ParticleNames:
    """The name table of a calculator.

    Args:
        symbols: the resolvable external symbols.
        aliases: ``{name: symbol}``; a symbol not in ``symbols`` raises.
    """

    def __init__(self, symbols, aliases=None):
        self.symbols = {}
        for sym in symbols:
            sym = sp.sympify(sym)
            self.symbols.setdefault(str(sym), sym)
        known = set(self.symbols.values())
        self.aliases = {}
        for name, sym in (aliases or {}).items():
            sym = sp.sympify(sym)
            if sym not in known:
                raise ValueError(
                    f"alias {name!r} names {sym!r}, which is not an external "
                    f"this calculator can resolve")
            if name in self.symbols and self.symbols[name] != sym:
                raise ValueError(
                    f"alias {name!r} -> {sym!r} collides with the symbol "
                    f"name {name!r} ({self.symbols[name]!r})")
            key = name.lower()
            if key in self.aliases and self.aliases[key] != sym:
                raise ValueError(
                    f"alias {name!r} is ambiguous: it names both "
                    f"{self.aliases[key]!r} and {sym!r} (aliases are "
                    f"case-insensitive)")
            self.aliases[key] = sym

    def as_dict(self):
        """``{name: symbol}``: symbol names, then aliases."""
        out = dict(self.symbols)
        for name, sym in self.aliases.items():
            out.setdefault(name, sym)
        return out

    def resolve(self, token):
        """The symbol named ``token`` (a symbol passes through)."""
        if not isinstance(token, str):
            return sp.sympify(token)
        if token in self.symbols:
            return self.symbols[token]
        if token.lower() in self.aliases:
            return self.aliases[token.lower()]
        names = list(self.as_dict())
        close = difflib.get_close_matches(token, names, n=3, cutoff=0.5)
        close += [n for n in names if n.lower() == token.lower()
                  and n not in close]
        hint = f"; did you mean {', '.join(map(repr, close))}?" if close else ""
        raise KeyError(f"unknown particle name {token!r}{hint} "
                       f"(the calculator's particle_names() lists them)")

    def parse_process(self, text):
        """``"a b > c d"`` → ``((a, b), (c, d))`` as symbols (2→2 only)."""
        sides = _ARROW.split(text.strip())
        if len(sides) != 2:
            raise ValueError(f"process {text!r}: write it as "
                             f"'in1 in2 > out1 out2'")
        incoming, outgoing = (side.split() for side in sides)
        if len(incoming) != 2 or len(outgoing) != 2:
            raise ValueError(
                f"process {text!r} has {len(incoming)} incoming and "
                f"{len(outgoing)} outgoing particles; only 2→2 is supported")
        return (tuple(self.resolve(t) for t in incoming),
                tuple(self.resolve(t) for t in outgoing))

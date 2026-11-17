"""Readable results: diagram rows, :class:`ProcessSummary`, :class:`DecayTable`.

Each prints as a fixed-width text table (``print(...)``) and renders as a
markdown table in a notebook (``_repr_markdown_``).  They only *read* what the
calculators computed; no number is produced here that the calculator's own
``numeric_*`` methods would not give.
"""

from dataclasses import dataclass, field

import sympy as sp

__all__ = ["DecayTable", "ProcessSummary", "diagram_row"]


def diagram_row(diagram):
    """``(channel, mediator, kind, note)`` for any enumerated diagram.

    - a four-fermion diagram (:class:`~feynlag.pheno.topology.DiagramInfo`):
      a boson exchange, the note is the fermion-permutation sign;
    - a fermion-line diagram (:class:`~feynlag.pheno.topology.LineDiagramInfo`):
      a fermion or a boson exchange;
    - a four-boson colour structure
      (:class:`~feynlag.pheno.topology.BosonDiagramInfo`): a boson exchange
      plus the contact structure sharing its colour tensor.
    """
    channel, mediator = diagram.channel, str(diagram.mediator)
    if hasattr(diagram, "sign"):
        return (channel, mediator, "boson", f"fermion sign {diagram.sign:+d}")
    if hasattr(diagram, "kind"):
        return (channel, mediator, diagram.kind, "")
    if hasattr(diagram, "structure"):
        return (channel, mediator, "boson", f"+ {diagram.structure} contact")
    raise TypeError(f"not an enumerated diagram: {diagram!r}")


def _fmt(x):
    return f"{x:.6g}"


def _text_table(header, rows):
    rows = [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    def join(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    width = sum(widths) + 2 * (len(widths) - 1)
    return [join(header), "-" * width] + [join(r) for r in rows]


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


_DIAGRAM_HEADER = ("channel", "mediator", "exchange", "note")


@dataclass(frozen=True)
class ProcessSummary:
    """What :meth:`~feynlag.pheno.scattering_calculator.ScatteringProcess.summary`
    returns.

    Attributes:
        label: the process, ``"e ebar → Wp Wm"``.
        diagrams: :func:`diagram_row` per diagram.
        initial_dof: the initial-state spin × colour states averaged over.
        symmetry_factor: ``1/2`` for identical final-state particles.
        sqrt_s: the CM energy in GeV, or ``None`` (nothing evaluated).
        sigma: the cross section in ``unit`` (``None`` without ``sqrt_s``).
        unit: the cross-section unit.
        cos_range: the ``cos θ`` acceptance ``sigma`` was integrated over.
        afb: the forward–backward asymmetry (``None`` without ``sqrt_s`` or
            for identical final-state particles).
    """

    label: str
    diagrams: tuple
    initial_dof: int
    symmetry_factor: object
    sqrt_s: float = None
    sigma: float = None
    unit: str = None
    cos_range: tuple = None
    afb: float = None

    def _facts(self):
        facts = [f"initial-state average 1/{self.initial_dof}"]
        if self.symmetry_factor != 1:
            facts.append(f"symmetry factor {self.symmetry_factor}")
        if self.sqrt_s is not None:
            facts.insert(0, f"sqrt(s) = {_fmt(self.sqrt_s)} GeV")
            cut = ("full angle" if tuple(self.cos_range) == (-1, 1)
                   else f"{_fmt(self.cos_range[0])} < cos(theta) < "
                        f"{_fmt(self.cos_range[1])}")
            facts.append(f"sigma = {_fmt(self.sigma)} {self.unit} ({cut})")
            if self.afb is not None:
                facts.append(f"A_FB = {_fmt(self.afb)}")
        return facts

    def __str__(self):
        n = len(self.diagrams)
        out = [f"{self.label}   ({n} diagram{'s' * (n != 1)})"]
        out += ["  " + line for line in _text_table(_DIAGRAM_HEADER,
                                                     self.diagrams)]
        out += ["  " + fact for fact in self._facts()]
        return "\n".join(out)

    def _repr_markdown_(self):
        out = [f"**{self.label}**", ""]
        out += _md_table(_DIAGRAM_HEADER, self.diagrams)
        out += [""] + [f"- {fact}" for fact in self._facts()]
        return "\n".join(out)


@dataclass(frozen=True)
class DecayRow:
    """One channel of a :class:`DecayTable`."""

    children: tuple
    width: float
    branching_ratio: float


@dataclass(frozen=True)
class DecayTable:
    """What :meth:`~feynlag.pheno.calculator.DecayCalculator.decay_table`
    returns: the open two-body channels of one parent at one parameter point.

    Attributes:
        parent: the decaying symbol.
        rows: :class:`DecayRow` per channel, largest width first; closed
            channels (below threshold at this point) carry ``0``.
        total: the total two-body width, in ``unit``.
        unit: the width unit.
        other: ``(width, BR)`` of the channels folded away by ``min_br``, or
            ``None``.
    """

    parent: sp.Expr
    rows: tuple
    total: float
    unit: str
    other: tuple = None
    _names: object = field(default=None, repr=False, compare=False)

    @property
    def widths(self):
        """``{children: Γ}`` in ``unit`` (folded channels excluded)."""
        return {r.children: r.width for r in self.rows}

    @property
    def branching_ratios(self):
        """``{children: BR}`` (folded channels excluded)."""
        return {r.children: r.branching_ratio for r in self.rows}

    def __getitem__(self, children):
        key = tuple(self._names.resolve(c) if self._names else sp.sympify(c)
                    for c in children)
        for row in self.rows:
            if row.children == key or row.children == key[::-1]:
                return row
        raise KeyError(f"no channel {self.parent} -> "
                       f"{' '.join(map(str, key))} in this table")

    def _rows(self):
        rows = [(f"{self.parent} -> {' '.join(map(str, r.children))}",
                 _fmt(r.width), _fmt(r.branching_ratio)) for r in self.rows]
        if self.other is not None:
            rows.append(("other", _fmt(self.other[0]), _fmt(self.other[1])))
        return rows

    def _header(self):
        return ("channel", f"width [{self.unit}]", "BR")

    def __str__(self):
        out = _text_table(self._header(), self._rows())
        out += [out[1],
                f"total width {_fmt(self.total)} {self.unit}"]
        return "\n".join(out)

    def _repr_markdown_(self):
        out = _md_table(self._header(), self._rows())
        out += ["", f"total width **{_fmt(self.total)} {self.unit}**"]
        return "\n".join(out)

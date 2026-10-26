"""LaTeX vertex-table generation.

Unifies the three ``generate_latex_table_*`` variants from
``bsm-calc/models/DLRSM1/symbolic_tools.py`` into one configurable function.
"""

from sympy import latex, simplify

__all__ = ["latex_feynman_table"]


def _rows_from_interactions(interactions):
    """Yield ``(field_tuple, coefficient)`` from any of the dict layouts.

    Accepts the nested ``{n_fields: {fields: coeff}}`` output of
    ``extract_interaction_coefficients``, a flat ``{fields: coeff}`` dict, or
    a multi-structure ``{fields: {lorentz name: coeff}}`` dict (a VVVV vertex
    has one coupling per Lorentz structure — see
    :meth:`~feynlag.vertices.vertex.Vertex.structure_couplings`), which gets
    one row per structure with the name appended to the interaction label.

    The three are told apart by their inner keys: ints/tuples of symbols for
    the nested layout, strings for the multi-structure one.
    """
    for key in sorted(interactions.keys(), key=str):
        value = interactions[key]
        if isinstance(value, dict):
            if value and all(isinstance(k, str) for k in value):
                for name, coeff in sorted(value.items()):
                    yield tuple(key) + (_StructureLabel(name),), coeff
            else:
                for fields, coeff in value.items():
                    yield fields, coeff
        else:
            yield key, value


class _StructureLabel:
    """A Lorentz-structure name rendered as a trailing bracket in a row."""

    def __init__(self, name):
        self.name = name

    def _latex(self, printer=None):
        return rf"[\mathrm{{{self.name}}}]"

    def __str__(self):
        return f"[{self.name}]"


def latex_feynman_table(interactions, simplify_coeff=None, extra_column=None,
                        extra_header=r"\textbf{Simplified}"):
    """Generate a LaTeX table of interactions and their coefficients.

    Args:
        interactions: nested dict from ``extract_interaction_coefficients``
            or a flat ``{field_tuple: coefficient}`` dict.
        simplify_coeff: optional callable applied to each coefficient before
            printing (default: print raw).
        extra_column: optional callable producing a third column from each
            coefficient (e.g. ``sympy.simplify`` or a series approximation).
            ``None`` gives a two-column table.
        extra_header: header of the third column.

    Returns:
        str: LaTeX ``array`` environment.
    """
    ncols = 3 if extra_column is not None else 2
    header_cells = [r"\textbf{Interaction}", r"\textbf{Coefficient}"]
    if extra_column is not None:
        header_cells.append(extra_header)

    table = r"\begin{array}{|" + "c|" * ncols + "}\n"
    table += "\\hline\n"
    table += " & ".join(header_cells) + r" \\" + "\n"
    table += "\\hline\n"

    for fields, coefficient in _rows_from_interactions(interactions):
        interaction_str = " ".join(latex(f) for f in fields)
        coeff = simplify_coeff(coefficient) if simplify_coeff else coefficient
        cells = [f"${interaction_str}$", f"${latex(coeff)}$"]
        if extra_column is not None:
            cells.append(f"${latex(extra_column(coefficient))}$")
        table += " & ".join(cells) + " \\\\\n"
        table += "\\hline\n"

    table += r"\end{array}"
    return table

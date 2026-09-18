"""LaTeX vertex-table export tests."""

import sympy as sp

from feynlag import extract_interaction_coefficients, latex_feynman_table

phi = sp.Symbol("phi", real=True)
lam = sp.Symbol("lambda", real=True)


def test_table_from_nested_dict():
    L = -lam / 24 * phi**4
    interactions = extract_interaction_coefficients(L, [phi])
    table = latex_feynman_table(interactions)
    assert table.startswith(r"\begin{array}")
    assert table.endswith(r"\end{array}")
    assert r"\phi" in table
    assert r"\lambda" in table


def test_table_from_flat_dict_with_extra_column():
    flat = {(phi, phi): lam * (sp.cos(phi) ** 2 + sp.sin(phi) ** 2)}
    table = latex_feynman_table(flat, extra_column=sp.simplify)
    # extra column simplifies to just lambda
    assert table.count("&") == 2 * 2  # header row + one data row, 3 columns


def test_table_from_multi_structure_dict():
    """A VVVV vertex has one coupling per Lorentz structure — each gets its
    own row, labelled, rather than being mistaken for the nested
    {n_fields: {fields: coeff}} layout."""
    Wm, Wp, Z = sp.symbols("Wm Wp Z")
    g = sp.Symbol("g")
    table = latex_feynman_table(
        {(Wm, Wp, Z, Z): {"VVVV2": sp.I * g ** 2, "VVVV3": -g}})
    assert r"[\mathrm{VVVV2}]" in table
    assert r"[\mathrm{VVVV3}]" in table
    assert table.count(r"\\") == 3        # header + two structure rows

"""Unit conversions for cross sections and widths, and the η acceptance.

Every calculator returns natural units: a cross section in GeV⁻² and a width
in GeV.  The conversion to barns uses the PDG conversion constant
``(ħc)² = 0.389 379 372 1 GeV² mbarn`` [PDG-PC], exact since the 2019 SI
redefinition fixed ħ and c.
"""

import math

__all__ = ["HBARC2_GEV2_MB", "CROSS_SECTION_UNITS", "ENERGY_UNITS",
           "cross_section_in", "energy_in", "eta_acceptance"]

#: (ħc)² in GeV² mb (PDG, "Physical Constants" review; exact)
HBARC2_GEV2_MB = 0.3893793721

#: ``{unit: factor}`` turning a GeV⁻² cross section into ``unit``
CROSS_SECTION_UNITS = {
    "GeV-2": 1.0,
    "mb": HBARC2_GEV2_MB,
    "ub": HBARC2_GEV2_MB * 1e3,
    "nb": HBARC2_GEV2_MB * 1e6,
    "pb": HBARC2_GEV2_MB * 1e9,
    "fb": HBARC2_GEV2_MB * 1e12,
}

#: ``{unit: factor}`` turning an energy in GeV into ``unit``
ENERGY_UNITS = {"TeV": 1e-3, "GeV": 1.0, "MeV": 1e3, "keV": 1e6, "eV": 1e9}


def _factor(table, unit, what):
    if unit in table:
        return table[unit]
    for name, factor in table.items():        # 'PB', 'mev': case-insensitive
        if name.lower() == str(unit).lower():
            return factor
    raise ValueError(f"unknown {what} unit {unit!r}; use one of "
                     f"{', '.join(table)}")


def cross_section_in(value_gev2, unit):
    """A cross section in GeV⁻² converted to ``unit`` (``'pb'``, ``'fb'``,
    ``'nb'``, ``'ub'``, ``'mb'`` or ``'GeV-2'``)."""
    return value_gev2 * _factor(CROSS_SECTION_UNITS, unit, "cross-section")


def energy_in(value_gev, unit):
    """An energy (a width, a mass) in GeV converted to ``unit``."""
    return value_gev * _factor(ENERGY_UNITS, unit, "energy")


def eta_acceptance(eta_max):
    """The ``cos θ`` range of ``|η| < eta_max``: ``(−tanh η, tanh η)``.

    Pseudorapidity ``η = −ln tan(θ/2)`` is a function of the angle alone, so
    ``cos θ = tanh η`` holds for massive particles too; the range is in the
    frame the angle is measured in (the CM frame of a 2→2 process, which is
    the lab frame of a symmetric collider).
    """
    if eta_max <= 0:
        raise ValueError(f"eta_max must be positive, got {eta_max!r}")
    c = math.tanh(eta_max)
    return (-c, c)

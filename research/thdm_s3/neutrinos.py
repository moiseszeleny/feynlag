"""The S₃ neutrino sector: Dirac Yukawas on H̃, a Majorana ν_R mass, the seesaw.

[LFVHD] declares three right-handed neutrinos, ``(ν₁R, ν₂R)`` an S₃ doublet and
``ν_SR`` a singlet, and never uses them; `02_s3_fermion_sector.ipynb` §8 found
that they allow exactly five Dirac structures on H̃ (the five charged-lepton
structures of `fermions._yukawa_terms`, with ``tilde=True``).  This module
builds that sector, adds the gauge-singlet Majorana mass
``−½ ν_Rᵀ C M_R ν_R + h.c.``, and supplies the type-I seesaw numerics that
`09_residual_z2.ipynb` uses:

- `build_neutrino_sector`: the ν_R fields (S₃-assigned) and the Dirac
  `fermions.FermionSector`;
- `invariant_majorana_basis`: the S₃-invariant symmetric ``M_R``, solved from
  the generators (not typed in);
- `majorana_lagrangian` / `majorana_matrix`: the Lagrangian term from
  feynlag's `MajoranaBilinear`, and the matrix read back off it;
- `light_spectrum`, `pmns_angles`: the seesaw light masses and mixing, from the
  analytic block formula ``m_ν = −m_D M_R⁻¹ m_Dᵀ``.  A numerical 6×6
  diagonalization would lose the light eigenvalues to round-off: with
  ``M_R ~ 10¹⁴`` GeV the double-precision floor is ``~10⁻² GeV``, eleven orders
  above ``m_ν``;
- `charged_scalar_loop`: the dipole function of a charged scalar with a neutral
  fermion of any mass, which decouples as ``1/(3t)`` for a heavy fermion.

Research code: `tests/` must never import this (see `research/README.md`).

References
----------
[LFVHD] see `fermions.py`.
[Lavoura03] L. Lavoura, Eur. Phys. J. C 29, 191 (2003), hep-ph/0302221: general
    formulae for f₁ → f₂γ, of which `charged_scalar_loop` is the neutral-fermion,
    photon-on-scalar case.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.integrate import quad

from feynlag import (ExternalParameter, MajoranaBilinear, WeylFermion, diracC,
                     diracPR)
from feynlag.vertices.bilinear import expand_bilinear

import fermions as F

__all__ = ["build_neutrino_sector", "invariant_majorana_basis", "majorana_lagrangian",
           "majorana_matrix", "light_spectrum", "pmns_angles", "charged_scalar_loop",
           "charged_scalar_loop_integral"]

#: ``C P_R``: the right-handed Majorana mass structure
CPR = diracC * diracPR


def build_neutrino_sector(s3, U1Y, lepton, scalars):
    """``(nuR, sector)``: ν_R = (ν₁R, ν₂R) ⊕ ν_SR and the Dirac sector on H̃.

    ``lepton`` is the charged-lepton `fermions.FermionSector`; its left-handed
    doublets are shared (the neutrino sits in the upper SU(2) slot).
    """
    nuR = tuple(WeylFermion(f"nu{g}R", reps={U1Y: 0}, chirality="R", nflavors=1,
                            component_names=[f"N_{g}"]) for g in F.GENERATIONS)
    s3.assign("2", nuR[0], nuR[1])
    s3.assign("1", nuR[2])
    Y = {k: ExternalParameter(f"Y^nu_{k}", 0.01 * k, real=True) for k in range(1, 6)}
    terms = F._yukawa_terms(lepton.left, nuR, scalars, tilde=True, ncolors=1)
    sector = F.FermionSector("neutrino", lepton.left, nuR, Y, terms,
                             tilde=True, su2_slot=0, ncolors=1)
    return nuR, sector


def invariant_majorana_basis(s3):
    """The S₃-invariant symmetric 3×3 ``M_R``: solve ``ρ(g)ᵀ M ρ(g) = M`` for both
    generators, ``ρ = diag(g₂, 1)`` on the (2 ⊕ 1) flavour space."""
    M = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"M_{min(i, j)}{max(i, j)}"))
    eqs = []
    for g in s3._irrep_generators["2"]:
        G = sp.diag(g, 1)
        eqs += list(G.T * M * G - M)
    sol = sp.solve(eqs, sorted(M.free_symbols, key=str), dict=True)
    if len(sol) != 1:
        raise ValueError("the invariance conditions have no unique solution family")
    return M.subs(sol[0])


def _legs(nuR):
    return [n.components[0][0] for n in nuR]


def majorana_lagrangian(nuR, M):
    """``−½ Σ_ij M_ij ν_iᵀ C P_R ν_j`` (the h.c. is not needed for the invariance
    checks or the mass matrix)."""
    legs = _legs(nuR)
    return -sp.Rational(1, 2) * sum(M[i, j] * MajoranaBilinear(legs[i], CPR, legs[j])
                                    for i in range(3) for j in range(3))


def majorana_matrix(L, nuR):
    """Read the symmetric ``M`` back off ``L ⊃ −½ M_ij ν_iᵀCP_Rν_j``.

    `MajoranaBilinear` canonicalises its legs, so ``i ≠ j`` gives one atom with
    coefficient ``−M_ij`` and ``i = j`` gives ``−½ M_ii``.  This is
    feynlag's `majorana_mass_matrix` for legs that are separate `WeylFermion`s,
    which that function (built for one flavour-indexed field) does not handle.
    """
    legs = _legs(nuR)
    L = sp.expand(expand_bilinear(sp.expand(L)))
    M = sp.zeros(3, 3)
    for i in range(3):
        for j in range(i, 3):
            c = L.coeff(MajoranaBilinear(legs[i], CPR, legs[j]))
            M[i, j] = M[j, i] = -2 * c if i == j else -c
    return M


def light_spectrum(mD, MR):
    """``(m_light, U)``: |eigenvalues| ascending and the orthogonal ν_L rotation of
    ``m_ν = −m_D M_R⁻¹ m_Dᵀ`` (real input; Majorana phases are not tracked)."""
    mnu = -mD @ np.linalg.solve(MR, mD.T)
    mnu = (mnu + mnu.T) / 2
    w, U = np.linalg.eigh(mnu)
    order = np.argsort(np.abs(w))
    return np.abs(w[order]), U[:, order]


def pmns_angles(U):
    """``(θ₁₂°, sin²θ₁₃, θ₂₃°)`` from |U| in the standard parametrization."""
    A = np.abs(U)
    s13 = A[0, 2]
    c13 = math.sqrt(max(1.0 - s13**2, 0.0))
    th12 = math.degrees(math.asin(min(A[0, 1] / c13, 1.0)))
    th23 = math.degrees(math.asin(min(A[1, 2] / c13, 1.0)))
    return th12, s13**2, th23


def charged_scalar_loop(t):
    """``f(t) = (1 − 6t + 3t² + 2t³ − 6t² ln t) / (6(1−t)⁴)``, ``t = m_F²/m_S²``.

    The non-flip dipole function of a charged scalar (photon on the scalar)
    and a neutral fermion of mass ``m_F``; the amplitude is
    ``A = −Σ_F Y_{eF}Y*_{μF} f(t_F) / (32π² m_S²)``.  ``f(0) = 1/6`` is the
    massless case of `radiative.one_loop_charged`; ``f → 1/(3t)`` decouples a heavy
    fermion.  Checked against `charged_scalar_loop_integral` in notebook 09.
    """
    if t == 0:
        return 1.0 / 6.0
    if abs(t - 1.0) < 1e-4:                     # removable singularity
        return 1.0 / 12.0 - (t - 1.0) / 30.0 + (t - 1.0) ** 2 / 60.0
    return (1 - 6 * t + 3 * t**2 + 2 * t**3 - 6 * t**2 * math.log(t)) / (6 * (1 - t) ** 4)


def charged_scalar_loop_integral(t):
    """The same function as a Feynman-parameter integral, ``∫₀¹ x²(1−x)/(x + (1−x)t) dx``."""
    return quad(lambda x: x * x * (1 - x) / (x + (1 - x) * t), 0.0, 1.0,
                epsabs=0.0, epsrel=1e-11, limit=200)[0]

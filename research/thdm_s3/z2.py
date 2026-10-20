"""The residual ℤ₂ of the aligned S₃ vacuum: generator, parities, soft-term split.

At the exact-S₃ vacuum the doublet VEV direction is (v₁, v₂) ∝ (1, √3)
(feynlag's basis, `model.build_model`).  Exactly one non-identity element of
S₃ leaves that direction fixed: a reflection, so it generates a ℤ₂ that
survives symmetry breaking.  `09_residual_z2.ipynb` derives it and its
consequences; this module holds the pieces later notebooks reuse:

- `vacuum_fixing_elements`: which S₃ elements fix a doublet vacuum, found by
  searching the group, not typed in;
- `residual_generator`: the 3×3 action ``G = diag(g, 1)`` on (doublet, singlet)
  flavour space, the same for scalars and for fermion multiplets assigned to
  **2 ⊕ 1**;
- `parity`: the ℤ₂ eigenvalue of a flavour vector (raises if the vector is not
  an eigenvector, so a mixed state cannot be given a parity by accident);
- `doublet_bilinear_rep`: how the 2-channel of 2⊗2 transforms, which splits the
  soft quadratics into ℤ₂-even and ℤ₂-odd combinations;
- `symmetric_split`: the ℤ₂-even and ℤ₂-odd parts of a symmetric flavour matrix
  (used for the Majorana mass in `neutrinos.py`).

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

__all__ = ["s3_doublet_elements", "vacuum_fixing_elements", "residual_generator",
           "parity", "doublet_bilinear_rep", "even_odd_directions", "symmetric_split"]


def s3_doublet_elements(s3):
    """All six S₃ elements in the doublet irrep, from the two generators."""
    a, b = s3._irrep_generators["2"]
    simp = lambda M: M.applyfunc(sp.nsimplify).applyfunc(sp.simplify)
    return {"e": sp.eye(2), "a": simp(a), "a2": simp(a * a), "b": simp(b),
            "ab": simp(a * b), "a2b": simp(a * a * b)}


def vacuum_fixing_elements(s3, vacuum):
    """Names of the S₃ elements ``g`` with ``g·v = v`` for the doublet VEVs ``v``."""
    v = sp.Matrix(vacuum)
    return [k for k, g in s3_doublet_elements(s3).items()
            if sp.simplify(g * v - v) == sp.zeros(2, 1)]


def residual_generator(s3, vacuum):
    """``(name, g, G)``: the unique non-identity vacuum-fixing element, and
    ``G = diag(g, 1)`` on (doublet₁, doublet₂, singlet) flavour space."""
    names = [k for k in vacuum_fixing_elements(s3, vacuum) if k != "e"]
    if len(names) != 1:
        raise ValueError(f"expected one non-identity element fixing the vacuum, got {names}")
    g = s3_doublet_elements(s3)[names[0]]
    return names[0], g, sp.diag(g, 1)


def parity(G, vec, tol=1e-9):
    """The ℤ₂ eigenvalue (±1) of a flavour vector under ``G``.

    Works on sympy (exact) or numpy (numeric) input.  Raises unless ``vec`` is an
    eigenvector, i.e. unless the state has a definite parity.
    """
    if isinstance(vec, sp.MatrixBase):
        p = sp.simplify((G * vec).dot(vec) / vec.dot(vec))
        if sp.simplify(G * vec - p * vec) != sp.zeros(*vec.shape):
            raise ValueError("not a ℤ₂ eigenvector: the state has no definite parity")
        return int(p)
    G = np.asarray(G, dtype=float)
    vec = np.asarray(vec, dtype=float)
    p = vec @ G @ vec / (vec @ vec)
    if np.abs(G @ vec - p * vec).max() > tol * max(np.abs(vec).max(), 1.0):
        raise ValueError("not a ℤ₂ eigenvector: the state has no definite parity")
    return int(round(p))


def doublet_bilinear_rep(g):
    """The 2×2 matrix ``T`` with ``d₂(gXgᵀ) = T d₂(X)`` for the 2-channel
    ``d₂(X) = (X₁₁ − X₂₂, −(X₁₂ + X₂₁))`` of 2⊗2 (the soft ``m_D`` pair, and the
    ``λ₄`` structure).  Solved from the transformation, not assumed."""
    X = sp.Matrix(2, 2, sp.symbols("X11 X12 X21 X22"))
    d2 = lambda Y: sp.Matrix([Y[0, 0] - Y[1, 1], -(Y[0, 1] + Y[1, 0])])
    t = sp.symbols("t0:4")
    T = sp.Matrix(2, 2, t)
    eqs = sp.expand(d2(g * X * g.T) - T * d2(X))
    coeffs = [c for e in eqs for c in sp.Poly(e, *X).coeffs()]
    sol = sp.solve(coeffs, t, dict=True)
    if len(sol) != 1:
        raise ValueError("the 2-channel transformation is not unique")
    return T.subs(sol[0]).applyfunc(sp.simplify)


def even_odd_directions(T):
    """``{+1: v_even, −1: v_odd}`` coefficient directions for a soft pair that
    enters the potential as ``c·s`` with ``s → T s``: ``c·s`` is invariant iff
    ``Tᵀc = c``, and flips sign iff ``Tᵀc = −c``."""
    out = {}
    for val, _, vecs in T.T.eigenvects():
        if len(vecs) != 1:
            raise ValueError("degenerate eigenvalue: no unique direction")
        out[int(val)] = sp.simplify(vecs[0] / vecs[0].norm())
    return out


def symmetric_split(G, M):
    """``(M_even, M_odd)`` with ``GᵀM_evenG = M_even``, ``GᵀM_oddG = −M_odd``."""
    if isinstance(M, sp.MatrixBase):
        back = (G.T * M * G).applyfunc(sp.simplify)
        return ((M + back) / 2).applyfunc(sp.simplify), ((M - back) / 2).applyfunc(sp.simplify)
    G = np.asarray(G, dtype=float)
    back = G.T @ M @ G
    return (M + back) / 2, (M - back) / 2

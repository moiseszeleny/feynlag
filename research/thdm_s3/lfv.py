"""Charged-lepton LFV couplings of the 3HDM-S₃, at the aligned or any vacuum.

Shared by `03_scalar_decays.ipynb` (exact-S₃ vacuum) and `06_soft_decays.ipynb`
(soft-broken vacuum).  Everything here is numeric; the one symbolic step —
splitting the mass matrix into per-doublet Yukawa matrices — is done once, on the
**Lagrangian**, and lambdified.

**Why the split comes from the Lagrangian.**  ``M_ℓ = Σ_k v_k G_k`` with ``G_k``
the Yukawa matrix of doublet k, and the scalar mass eigenstate i couples through
``Q_i = Σ_j (R_S)_{ji} OᵀG_jO`` ([LFVHD] Eq. Q_general).  Writing ``M_ℓ`` with the
alignment's √3 typed in (as notebook 03 §7 first did) hides one VEV, so
`fermions.g_matrices` silently returns ``G₂ = 0`` — every doublet Yukawa then sits
on ``G₁``.  ``Q₁`` and ``Q₃`` survive that (they depend on ``Σ v_kG_k = M``
only), but ``Q₂`` — the gauge-phobic ``h₀``'s couplings — comes out wrong.
Reading ``M_ℓ`` off the Lagrangian with all three VEVs symbolic cannot make that
mistake, and since it is feynlag's basis throughout, the scalar rotations
`decays.build_decay_model` registers apply with no basis map.

**The μ dictionary.**  At the alignment the draft's analytic inversion
(`lepton_mus`, notebook 02 §5–§6, ``μ₅ = μ₄``) fixes μ₁, μ₂, μ₄ from the three
lepton masses and the free dial μ₃.  feynlag's Yukawas follow from
``μ_k = Y_k v/√2`` (notebook 02 §4) with the doublet VEV ``|v₂^draft| = v₁``.  The
draft basis has ``v₂^draft = −v₁ < 0`` (notebook 02 §4.2), so ``Y₂`` picks up
that sign — `aligned_yukawas` asserts the result reproduces (m_e, m_μ, m_τ)
rather than trusting it.

**Away from the alignment** the electron no longer decouples, but the inversion
is still exact.  In feynlag's basis the doublet block of ``M_ℓ`` is
``a·1 + b·v₁₂[[cosφ, −sinφ], [−sinφ, −cosφ]]`` (a reflection), whose
eigenvectors depend on φ alone, and the third generation couples to them through
``(cos 3φ/2, sin 3φ/2)``.  So ``M_ℓ`` is orthogonally an *arrowhead* matrix
``[[A, 0, D cos χ], [0, B, D sin χ], [D cos χ, D sin χ, μ₃]]`` with ``χ = 3φ/2``,
and matching its characteristic polynomial to ``Π(λ − s_i m_i)`` leaves **one
cubic** in ``Δ = A − B`` per sign pattern ``s`` (`lepton_branches`).  Every
solution is found, none by luck; `fit_yukawas` (least squares) is kept only as
an independent check.  Two consequences, both depending on φ only and not on θ:

* at fixed μ₃ there are several discrete solutions (branches), differing in the
  eigenvalue signs and in *which* diagonal entry the electron grows out of;
* away from the residual-Z₂ directions (cos 3φ = ±1, i.e. φ = π/3 and φ → 0) a
  real solution exists only for ``|μ₃|`` near ``m_τ`` or below ~``m_μ`` —
  `mu3_allowed`.  The draft's dial is no longer free over ``(m_μ, m_τ)``.

References
----------
[LFVHD] M. Zeleny-Mora, M. Mondragón, T. A. Valencia-Pérez, "Exploring LFV Higgs
    decays in the Three Higgs Doublet Model", draft (`paper_lfvhd/LFVHD_3HDMS3.tex`).
[CMS21] CMS Collaboration, Phys. Rev. D 104, 032013 (2021), arXiv:2105.03007.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

__all__ = [
    "LEPTON_MASSES", "lepton_mus", "lepton_g_function", "aligned_yukawas",
    "lepton_mass_basis", "fit_yukawas", "lepton_branches", "branch_kind",
    "mu3_allowed",
    "q_matrices",
    "lfv_width", "IDX",
]

#: (m_e, m_μ, m_τ) in GeV [PDG].
LEPTON_MASSES = (0.51099895e-3, 0.1056584, 1.77686)

#: mass-basis index of each lepton (lightest first).
IDX = {"e": 0, "mu": 1, "tau": 2}


def lepton_mus(mu3, masses=LEPTON_MASSES):
    """The draft's analytic inversion at the alignment: ``(μ₁, μ₂, μ₃, μ₄, θ_ℓ)``.

    Notebook 02 §5–§6 with ``μ₅ = μ₄``: ``m_e = μ₁ − 2μ₂`` is the O₁₂-decoupled
    state and the residual 2–3 block fixes the rest.  ``μ₃ ∈ (m_μ, m_τ)``.
    """
    me, mmu, mtau = masses
    mu1 = (me + mmu + mtau - mu3) / 2
    mu2 = (mmu + mtau - me - mu3) / 4
    p1, p2 = mu3 - mmu, mtau - mu3
    if p1 <= 0 or p2 <= 0:
        raise ValueError(f"mu3 = {mu3} is outside (m_mu, m_tau)")
    mu4 = 0.5 * math.sqrt(p1 * p2)
    return mu1, mu2, mu3, mu4, math.atan2(math.sqrt(p2), math.sqrt(p1))


def lepton_g_function(lep, scalar):
    """Lambdified ``Y₁…Y₅ → (G₁, G₂, G_S)``, read off the Lagrangian.

    ``lep`` is a `fermions.FermionSector`, ``scalar`` the `model.S3Model` whose
    doublets it couples to.  ``M_ℓ`` is extracted at the symbolic vacuum with all
    three VEVs free and split by `fermions.g_matrices` — which raises unless
    ``Σ v_kG_k`` rebuilds ``M_ℓ`` exactly.  Each ``G_k`` is the coupling matrix of
    the *real* neutral field of doublet k (the ``1/√2`` of the VEV and of the
    field expansion cancel between ``M`` and ``L``), i.e. exactly the weak-basis
    field the CP-even rotation acts on.
    """
    from feynlag import Vacuum
    import fermions as F

    bars, rights = lep.mass_legs()
    M = F.mass_matrix_from_bilinears(lep.lagrangian(), bars, rights,
                                     Vacuum(list(scalar.doublets)))
    vevs = (scalar.v1.s, scalar.v2.s, scalar.vS.s)
    G = F.g_matrices(M, vevs)
    if any(g.free_symbols & set(vevs) for g in G):
        raise ValueError("a G_k still depends on the VEVs: M_l is not linear in them")
    fn = sp.lambdify([lep.coupling_symbols], [g.tolist() for g in G], "numpy")

    def g_of(Y):
        return [np.array(g, dtype=float) for g in fn(list(Y))]
    return g_of


def lepton_mass_basis(G, vevs):
    """``(masses, O)`` for ``M = Σ v_kG_k``: |eigenvalues| ascending, O orthogonal.

    Requires ``M`` symmetric (the ``μ₅ = μ₄`` ansatz, ``Y₅ = Y₄``), so one
    orthogonal ``O`` serves both chiralities.  A negative eigenvalue is a chiral
    phase on one leg; it flips the sign of a row/column of ``Q`` and leaves every
    ``|Q_ab|`` — hence every rate — unchanged.
    """
    M = sum(v * g for v, g in zip(vevs, G))
    if np.abs(M - M.T).max() > 1e-12 * max(np.abs(M).max(), 1e-300):
        raise ValueError("M_l is not symmetric: set Y5 = Y4 (the mu5 = mu4 ansatz)")
    w, O = np.linalg.eigh(M)
    order = np.argsort(np.abs(w))
    return np.abs(w[order]), O[:, order]


def aligned_yukawas(g_of, mu3, v1, vS, masses=LEPTON_MASSES, rtol=1e-9):
    """feynlag's ``(Y₁…Y₅)`` on the draft branch at the √3-aligned vacuum.

    The vacuum is ``(v₁, √3v₁, v_S)`` **exactly** — imposed here rather than read
    from rounded stored VEVs, so the masses come out to ``rtol``.  Asserts that.
    """
    mu1, mu2, mu3_, mu4, _ = lepton_mus(mu3, masses)
    s2 = math.sqrt(2)
    Y4 = s2 * mu4 / v1
    Y = [s2 * mu1 / vS, -s2 * mu2 / v1, s2 * mu3_ / vS, Y4, Y4]
    got, _ = lepton_mass_basis(g_of(Y), (v1, math.sqrt(3) * v1, vS))
    if not np.allclose(got, masses, rtol=rtol):
        raise AssertionError(f"draft dictionary does not reproduce the lepton "
                             f"masses: {got} vs {masses}")
    return Y


def _residual(x, g_of, Y3, vevs, target):
    Y = [x[0], x[1], Y3, x[2], x[2]]
    got, _ = lepton_mass_basis(g_of(Y), vevs)
    return np.log(np.maximum(got, 1e-300) / target)


def fit_yukawas(g_of, mu3, vevs, Y_start, masses=LEPTON_MASSES, tol=1e-10):
    """Solve ``(Y₁, Y₂, Y₄)`` at fixed ``μ₃ = Y₃v_S/√2`` so ``M_ℓ`` has the lepton masses.

    Least squares on ``log(|m_i|/m_i^PDG)`` (the masses span 3.5 decades),
    warm-started from ``Y_start``.  Returns ``(Y, max|log residual|)`` and
    raises if that exceeds ``tol``.
    """
    from scipy.optimize import least_squares

    target = np.asarray(masses, dtype=float)
    Y3 = math.sqrt(2) * mu3 / vevs[2]
    x0 = np.array([Y_start[0], Y_start[1], Y_start[3]], dtype=float)
    sol = least_squares(_residual, x0, args=(g_of, Y3, vevs, target),
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    res = float(np.abs(sol.fun).max())
    if res > tol:
        raise RuntimeError(f"lepton fit failed: max |log residual| = {res:.2e}")
    x = sol.x
    return [x[0], x[1], Y3, x[2], x[2]], res


def _branch_roots(c, k, masses):
    """``[(signs, Δ, S, D²)]``: every real arrowhead solution at μ₃ = c, cos 3φ = k.

    For eigenvalues ``λ_i = s_i m_i`` with elementary symmetric polynomials
    ``e₁, e₂, e₃``, ``S = A + B = e₁ − c`` and ``q = cS − e₂``, the condition on
    ``e₃`` is the cubic (in ``Δ = A − B``)

        −(k/8)Δ³ + (S/8 − c/4)Δ² + (S²/4 + q)(k/2)Δ
            + cS²/4 − (S²/4 + q)S/2 − e₃ = 0,

    which is ``c·AB − D²(B cos²χ + A sin²χ) − e₃`` with ``AB = (S² − Δ²)/4`` and
    ``D² = AB + q``, ``cos 2χ = k``.  A real root with ``D² ≥ 0`` is a branch.
    """
    import itertools

    m = np.asarray(masses, dtype=float)
    out = []
    for signs in itertools.product((1, -1), repeat=3):
        lam = np.array(signs) * m
        e1, e3 = lam.sum(), lam.prod()
        e2 = lam[0] * lam[1] + lam[0] * lam[2] + lam[1] * lam[2]
        S = e1 - c
        q = c * S - e2
        coeffs = [-k / 8, S / 8 - c / 4, (S * S / 4 + q) * k / 2,
                  c * S * S / 4 - (S * S / 4 + q) * S / 2 - e3]
        dcoeffs = np.polyder(coeffs)
        for root in np.roots(coeffs):
            if abs(root.imag) > 1e-10 * max(1.0, abs(root)):
                continue
            dl = root.real
            for _ in range(3):              # Newton polish: np.roots loses digits
                slope = np.polyval(dcoeffs, dl)    # near a double root (a branch edge)
                if slope == 0:
                    break
                dl -= np.polyval(coeffs, dl) / slope
            D2 = (S * S - dl * dl) / 4 + q
            if D2 < -1e-12 * max(1.0, abs(e2)):
                continue
            out.append((signs, dl, S, max(D2, 0.0)))
    return out


def lepton_branches(g_of, mu3, vevs, masses=LEPTON_MASSES, tol=1e-7):
    """Every ``(Y₁…Y₅)`` with ``Y₅ = Y₄`` and ``μ₃ = Y₃v_S/√2`` giving the lepton masses.

    Exact, via the arrowhead form of the module docstring.  For each sign
    pattern ``s`` of the eigenvalues ``λ_i = s_i m_i`` the characteristic
    polynomial of ``[[A,0,x],[0,B,y],[x,y,c]]`` (``x² + y² = D²``,
    ``x = D cos χ``) has

        e₁ = A + B + c,   e₂ = AB + c(A+B) − D²,
        e₃ = ABc − D²(B cos²χ + A sin²χ),

    so with ``S = e₁ − c``, ``Δ = A − B`` and ``AB = (S² − Δ²)/4`` the last line is
    a cubic in Δ (`_branch_roots`; ``cos 2χ = cos 3φ``).  Real roots with
    ``D² ≥ 0`` are the branches.  Each is mapped back to Yukawas and
    **verified** against `lepton_mass_basis` to ``tol`` — an algebra slip here
    cannot pass silently.

    Returns a list of dicts ``{"signs", "Y", "A", "B", "D", "kind"}``, with
    ``kind`` from `branch_kind`.  Only ``D ≥ 0`` is returned: ``Y₄ → −Y₄`` (a sign
    flip of the third generation) leaves the masses and every ``|Q|`` unchanged,
    so it is quotiented out.
    """
    v1, v2, vS = vevs
    w = math.hypot(v1, v2)
    k = math.cos(3 * math.atan2(v2, v1))
    c = float(mu3)
    m = np.asarray(masses, dtype=float)
    out = []
    for signs, dl, S, D2 in _branch_roots(c, k, masses):
        A, B = (S + dl) / 2, (S - dl) / 2
        a, b = (A + B) / 2, (A - B) / (2 * w)
        d = math.sqrt(D2) / w
        s2 = math.sqrt(2)
        Y = [s2 * a / vS, s2 * b, s2 * c / vS, s2 * d, s2 * d]
        got, _ = lepton_mass_basis(g_of(Y), vevs)
        if np.max(np.abs(got / m - 1)) > tol:
            raise AssertionError(f"branch {signs} does not reproduce the masses: {got}")
        out.append({"signs": signs, "Y": Y, "A": A, "B": B, "D": d * w,
                    "kind": branch_kind(A, masses)})
    return out


def branch_kind(A, masses=LEPTON_MASSES):
    """Which lepton the ``e₊`` diagonal entry ``A`` belongs to: ``"e"``, ``"mu"`` or ``"tau"``.

    ``e₊`` is the doublet direction whose coupling to the third generation is
    ``D cos(3φ/2)``, so at the alignment it is an exact eigenvector, with
    eigenvalue ``A``.  The draft's branch (`lepton_mus`) has ``|A| = m_e`` there:
    the electron decouples.  Other branches decouple the muon or the tau
    instead.  Off the alignment ``A`` is no longer an eigenvalue, but it moves
    continuously, so the label "the lepton |A| is closest to (in log)" follows
    each branch away from φ = π/3.
    """
    logm = np.log(np.asarray(masses, dtype=float))
    return ("e", "mu", "tau")[int(np.argmin(np.abs(math.log(max(abs(A), 1e-300)) - logm)))]


def mu3_allowed(phi, grid, masses=LEPTON_MASSES):
    """Boolean mask over ``grid``: does any branch exist at this φ and μ₃?

    Depends on φ only (the cubic sees the vacuum through cos 3φ alone), so one
    call serves every point with the same φ.
    """
    k = math.cos(3 * phi)
    return np.array([bool(_branch_roots(float(c), k, masses)) for c in grid])


def q_matrices(G, O, mixing):
    """``Q_i = Σ_j mixing[j, i] · OᵀG_jO`` — the CP-even state i's lepton couplings.

    ``mixing[j, i]`` is state i's component along the real neutral field of
    doublet j (feynlag's (H₁, H₂, H_S) order) — `decays.DecayModel.RS` at the
    alignment, ``cp_even_mixing`` of `results/viable_points_soft.json` away from it.
    """
    Gt = [O.T @ g @ O for g in G]
    mixing = np.asarray(mixing, dtype=float)
    return [sum(mixing[j, i] * Gt[j] for j in range(3)) for i in range(mixing.shape[1])]


def lfv_width(q_ab, q_ba, m_h):
    """``Γ(h → ℓ_a ℓ̄_b + ℓ̄_a ℓ_b)`` in the ``m_ℓ ≪ m_h`` limit (notebook 03 §7)."""
    return m_h / (8 * math.pi) * (abs(q_ab) ** 2 + abs(q_ba) ** 2)

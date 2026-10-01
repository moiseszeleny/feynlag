"""Decays and LFV rates of the 3HDM-S₃ in the **soft-broken** vacuum.

Notebook 03 computed these observables on the exact-S₃ vacuum, where the
CP-even sector is one angle δ on top of the geometric rotation and h₀ is
gauge-phobic.  The soft-broken points of `results/viable_points_soft.json`
(notebook 05) have none of that structure: the vacuum (θ, φ) is general, the
CP-even 3×3 is fully mixed, and the CP-odd and charged 2×2 blocks are no longer
diagonal.  This module recomputes every observable **numerically per point**,
from the stored λ's, vacuum angles and soft terms.

Why it can be numeric and still trustworthy:

* **The soft terms are dimension 2**, so they enter only the mass matrices, and
  hence only the rotations.  Every cubic coupling (`trilinear_function`) comes
  from the quartics alone and is lambdified **once** from the Lagrangian; at a
  point it is contracted with that point's rotations.  The soft model's
  Lagrangian is used, so the quartics are literally the scan's.
* **The kinetic couplings are pure geometry**: with the electroweak kinetic
  terms ``Σ|D_μH_i|²`` every doublet couples to Z and W identically, so the VSS
  couplings are ``g/(2c_W)·(R_oddᵀR_even)`` and ``(g/2)·(R_Cᵀ R_even)``.  Their
  normalization is pinned against the feynlag extractor in notebook 06, not
  assumed.
* **The lepton couplings** come from `lfv.py`: per-doublet Yukawa matrices read
  off the Lagrangian, and the exact branch enumeration of the lepton fit.

`06_soft_decays.ipynb` checks all of it against the library extractor and
against corrected notebook 03 on the exact-S₃ slice.

References
----------
See `decays.py` ([CMS21], [CMS23], [LFVHD]) and `lfv.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field

import numpy as np
import sympy as sp

import fermions as F
import lfv
from model import (SOFT_SPECTRUM_KEYS, V_EW, build_model, soft_free_names,
                   soft_mass_function, soft_vevs, _geometric_rotation_numeric)

__all__ = [
    "SoftTools", "build_tools", "point_inputs", "point_rotations",
    "trilinear_function", "hhh_charged", "vss_overlaps", "vss_width",
    "lfv_point", "diphoton_ratio", "sm_like_index", "image_point", "scan_point",
]

#: The S₃ reflection about the 60° axis (the element ab), acting on both soft
#: doublets: (mD1², mD2²) and (mS1², mS2²).  Maps φ → 2π/3 − φ
#: (`tests/test_thdm_s3.py::test_soft_s3_image_is_the_same_physics`).
S3_IMAGE = np.array([[-0.5, math.sqrt(3) / 2], [math.sqrt(3) / 2, 0.5]])


@dataclass(eq=False)
class SoftTools:
    """The lambdified pieces every point reuses — build once with `build_tools`."""

    model: object                  # model.S3Model, soft=True, solved for mD1sq
    mass_fn: object                # soft_mass_function(model)
    tri_fn: object                 # trilinear_function(model)
    g_of: object                   # lfv.lepton_g_function
    _cache: dict = _field(default_factory=dict, repr=False)


def trilinear_function(m):
    """Lambdified ``(λ₁…λ₈, v₁, v₂, v_S) → T[a, j, k]`` (3×3×3).

    ``T[a,j,k] = ∂³V/∂r_a ∂c_j ∂c̄_k`` at the vacuum, with ``r_a`` the real
    neutral fluctuation of doublet a (``H_a⁰ = (v_a + r_a)/√2``) and ``c_j``,
    ``c̄_k`` the charged component of doublet j and its conjugate.  Only the
    quartic part of V contributes — a quadratic (μ², soft) term has no cubic
    piece after the shift — so this is λ- and v-dependent only.

    The CP-odd fluctuations are set to zero: this is the CP-even–charged–charged
    tensor the ``h H⁺H⁻`` couplings need, nothing more.
    """
    V = -sum(t.expr for t in m.model.lagrangian.terms)
    lam = m.lam_symbols
    V4 = sum(t for t in sp.expand(V).args if any(t.has(l) for l in lam))

    vs = [m.v1.s, m.v2.s, m.vS.s]
    r = sp.symbols("r0:3", real=True)
    c = sp.symbols("c0:3")
    cb = sp.symbols("cb0:3")
    charged = [H.components[0] for H in m.doublets]
    neutral = [H.components[1] for H in m.doublets]
    sub_conj = {sp.conjugate(x): cb[k] for k, x in enumerate(charged)}
    sub_conj.update({sp.conjugate(x): (vs[k] + r[k]) / sp.sqrt(2)
                     for k, x in enumerate(neutral)})
    sub = {x: c[k] for k, x in enumerate(charged)}
    sub.update({x: (vs[k] + r[k]) / sp.sqrt(2) for k, x in enumerate(neutral)})
    Vs = sp.expand(V4.xreplace(sub_conj).xreplace(sub))
    stray = Vs.free_symbols - set(lam) - set(vs) - set(r) - set(c) - set(cb)
    if stray:
        raise ValueError(f"field symbols survived the substitution: {sorted(map(str, stray))}")
    zero = {x: 0 for x in (*r, *c, *cb)}
    T = [[[sp.diff(Vs, r[a], c[j], cb[k]).subs(zero) for k in range(3)]
          for j in range(3)] for a in range(3)]
    return sp.lambdify([lam, vs], T, "numpy")


def build_tools() -> SoftTools:
    """The soft model (tadpoles solved for mD1², as `scan_soft.py`) and its tools."""
    m = build_model(soft=True, soft_solve_for="mD1sq")
    lep = F.build_lepton_sector(m.s3, m.SU2L, m.U1Y, m.doublets)
    return SoftTools(model=m, mass_fn=soft_mass_function(m),
                     tri_fn=trilinear_function(m), g_of=lfv.lepton_g_function(lep, m))


# --------------------------------------------------------------------------
# one point: inputs and rotations
# --------------------------------------------------------------------------

def point_inputs(tools: SoftTools, point):
    """``(lam, vevs, softs)`` for one stored point, VEVs rebuilt from (θ, φ)."""
    lam = [point["lambdas"][f"lambda_{k + 1}"] for k in range(8)]
    v1, v2, vS = (float(x) for x in soft_vevs(point["theta"], point["phi"]))
    softs = [point["softs_GeV2"][n] for n in soft_free_names(tools.model)]
    return lam, (v1, v2, vS), softs


def point_rotations(tools: SoftTools, point, check=True, check_mixing=True,
                    rtol=1e-2):
    """Mass-basis rotations and masses at one soft point.

    Returns a dict with

    ``R_even``  (3,3) columns = h1 < h2 < h3 in the (H₁⁰, H₂⁰, H_S⁰) real basis,
                sign fixed so each state's hVV coupling is ≥ 0 (the stored
                ``cp_even_mixing`` convention);
    ``R_odd``   (3,3) columns = (G⁰, A1 < A2) in the imaginary basis;
    ``R_C``     (3,3) columns = (G⁺, H⁺1 < H⁺2) in the charged basis;
    ``mass``    {`SOFT_SPECTRUM_KEYS`: GeV};
    ``vevs``    (v₁, v₂, v_S).

    With ``check`` the masses (and, with ``check_mixing``, the CP-even mixing)
    are compared with the stored ones.  The tolerance is set by **storage**:
    λ's and (θ, φ) are rounded to 5 decimals, which moves every mass² by a
    fraction of the largest one — up to ~3e-3 near φ → 0, where the solved
    ``m_D1² ∝ 1/(v₁v₂)`` amplifies a rounding of φ.  The check is there to catch
    an order-one error (a wrong rotation or soft-term order), not rounding.
    """
    lam, vevs, softs = point_inputs(tools, point)
    M_S, M_A, M_C, _ = tools.mass_fn(*lam, *vevs, *softs)
    M_S, M_A, M_C = (np.array(M, dtype=float) for M in (M_S, M_A, M_C))
    R = _geometric_rotation_numeric(*(np.atleast_1d(x) for x in vevs))[0]

    w, U = np.linalg.eigh(R.T @ M_S @ R)
    U = U * np.where(U[0, :] < 0, -1.0, 1.0)          # hVV coupling ≥ 0
    R_even = R @ U

    def two_block(M):
        D = R.T @ M @ R
        wb, Ub = np.linalg.eigh(D[1:, 1:])             # ascending
        full = np.eye(3)
        full[1:, 1:] = Ub
        return wb, R @ full

    wA, R_odd = two_block(M_A)
    wC, R_C = two_block(M_C)
    mass2 = dict(zip(SOFT_SPECTRUM_KEYS, (*w, *wA, *wC)))
    if min(mass2.values()) <= 0:
        raise ValueError("tachyonic state at a stored viable point")
    mass = {k: math.sqrt(v) for k, v in mass2.items()}

    if check:
        # rounding of (θ, φ, λ) moves every mass² by a fraction of the LARGEST
        # one, so the tolerance scales with it, not with each state's own mass
        scale = max(mass2.values())
        for k in SOFT_SPECTRUM_KEYS:
            stored = point["masses_GeV"][k]
            if abs(mass2[k] - stored**2) > rtol * scale:
                raise AssertionError(f"{k}: recomputed {mass[k]:.3f} vs stored {stored:.3f}")
        if check_mixing:
            stored_mix = np.array(point["cp_even_mixing"])
            if np.abs(R_even - stored_mix).max() > 5e-3:
                raise AssertionError("recomputed CP-even mixing differs from the stored one")
    return {"R_even": R_even, "R_odd": R_odd, "R_C": R_C, "mass": mass,
            "vevs": vevs}


def sm_like_index(rot):
    """The CP-even state carrying the most hVV coupling — a coupling, not a label."""
    vhat = np.array(rot["vevs"]) / np.linalg.norm(rot["vevs"])
    return int(np.argmax((vhat @ rot["R_even"]) ** 2))


# --------------------------------------------------------------------------
# couplings
# --------------------------------------------------------------------------

def hhh_charged(tools: SoftTools, point, rot):
    """``g[k, a, b]``: the ``h_k H⁺_a H⁻_b`` coupling, Feynman rule with its ``i`` stripped.

    The Lagrangian coefficient is ``−T`` contracted with the rotations; the
    feynlag rule is ``i × coefficient × ∏(multiplicity)!`` and all three legs are
    distinct, so ``g = −Σ T[a,j,k] R_even[a,·] R_C[j,·] R_C[k,·]``.  Indices
    a, b run over (G⁺, H⁺1, H⁺2).
    """
    lam, vevs, _ = point_inputs(tools, point)
    T = np.array(tools.tri_fn(lam, list(vevs)), dtype=float)
    return -np.einsum("ajk,ai,jb,kc->ibc", T, rot["R_even"], rot["R_C"], rot["R_C"])


def vss_overlaps(rot):
    """``(O_Z, O_W)`` with ``O_Z[a,k] = (R_oddᵀR_even)[a,k]`` and ``O_W = R_CᵀR_even``.

    The VSS coupling of ``Z A_a h_k`` is ``(g/2c_W)·O_Z[a,k]`` and of
    ``W⁺ H⁻_a h_k`` is ``(g/2)·O_W[a,k]`` (normalization checked against the
    extractor in notebook 06).  Row 0 is the Goldstone; its column overlap with
    a CP-even state is that state's hVV coupling — the Goldstone-equivalence
    shadow of ``h → VV``.
    """
    return rot["R_odd"].T @ rot["R_even"], rot["R_C"].T @ rot["R_even"]


def kallen(a, b, c):
    return a * a + b * b + c * c - 2 * (a * b + a * c + b * c)


def vss_width(c, m_parent, m_vector, m_scalar):
    """``Γ(S → V S') = |c|² λ^{3/2}(M², m_V², m_S'²)/(16π M³ m_V²)``; 0 below threshold.

    ``c`` is the coefficient of ``(p(S) − p(S'))`` in the Feynman rule — the form
    `feynlag.pheno.amplitudes.vss_squared` squares (CLAUDE.md, decay widths).
    """
    if m_parent <= m_vector + m_scalar:
        return 0.0
    lam = kallen(m_parent**2, m_vector**2, m_scalar**2)
    return abs(c) ** 2 * lam ** 1.5 / (16 * math.pi * m_parent**3 * m_vector**2)


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------

def lfv_point(tools: SoftTools, point, rot, mu3_values):
    """LFV coupling matrices of all three CP-even states, every lepton branch.

    Returns a list of ``{"mu3", "branch", "signs", "Q"}`` with ``Q`` the three
    3×3 coupling matrices (mass-ordered CP-even states).  A μ₃ with no real
    branch at this φ contributes nothing (`lfv.mu3_allowed`).
    """
    out = []
    for mu3 in mu3_values:
        for n, br in enumerate(lfv.lepton_branches(tools.g_of, mu3, rot["vevs"])):
            G = tools.g_of(br["Y"])
            _, O = lfv.lepton_mass_basis(G, rot["vevs"])
            out.append({"mu3": float(mu3), "branch": n, "signs": br["signs"],
                        "kind": br["kind"], "Q": lfv.q_matrices(G, O, rot["R_even"])})
    return out


def diphoton_ratio(g_hpp, m_h, m_hp, kappa_v, kappa_t, m_t, m_w, v=V_EW):
    """``Γ(h→γγ)/Γ_SM`` with W, top and both charged Higgses in the loop.

    ``g_hpp`` are the two diagonal ``h H⁺_a H⁻_a`` couplings (GeV), entering as
    ``c = g v/(2m²)`` ([ChakrabartyEtAl21] Eq. 27, the normalization
    `higgs_diphoton_amplitude` documents).
    """
    from feynlag.pheno.loop import higgs_diphoton_amplitude

    sm = [(1, 1, m_w, 1, 1), (1, 0.5, m_t, 2 / 3, 3)]
    bsm = [(kappa_v, 1, m_w, 1, 1), (kappa_t, 0.5, m_t, 2 / 3, 3)]
    bsm += [(g * v / (2 * m * m), 0, m, 1, 1) for g, m in zip(g_hpp, m_hp)]
    return (abs(higgs_diphoton_amplitude(m_h, bsm)) ** 2
            / abs(higgs_diphoton_amplitude(m_h, sm)) ** 2)


def image_point(tools: SoftTools, point):
    """The same physics at the S₃-image vacuum φ → 2π/3 − φ.

    Both soft doublets transform with `S3_IMAGE`.  The doublet one needs the
    point's mD1², which the tadpoles fix: it is **re-solved** here rather than
    read from the JSON, whose value is rounded to 0.01 GeV² — feeding that
    rounding through the reflection into the image's free mD2² shows up at the
    1e-3 level in the smallest heavy-state overlaps.  The CP-even mixing is
    *not* the stored one (the doublet rows are reflected), so compare with
    ``check_mixing=False``.
    """
    from model import soft_solved_names

    lam, vevs, softs = point_inputs(tools, point)
    solved = tools.mass_fn(*lam, *vevs, *softs)[3]
    mD1 = float(solved[soft_solved_names(tools.model).index("mD1sq")])
    sf = point["softs_GeV2"]
    mD = S3_IMAGE @ [mD1, sf["mD2sq"]]
    mS = S3_IMAGE @ [sf["mS1sq"], sf["mS2sq"]]
    out = dict(point)
    out["phi"] = 2 * math.pi / 3 - point["phi"]
    out["softs_GeV2"] = {"mD1sq": float(mD[0]), "mD2sq": float(mD[1]),
                         "mS1sq": float(mS[0]), "mS2sq": float(mS[1])}
    return out


#: PDG inputs for the loop and VSS widths [PDG].
M_W, M_Z, M_T = 80.377, 91.1876, 172.69
G_W = 2 * M_W / V_EW                        # g, from m_W = g v / 2
C_W = M_W / M_Z


def scan_point(tools: SoftTools, point, mu3_grid, image=False):
    """Every observable notebook 06 reports, at one soft point.

    Returns a dict: the SM-like index and its κ_V, the ``R_γγ`` ratio, the
    ``h_k H⁺_a H⁻_a`` couplings, the VSS widths ``A_a → Z h_SM`` and
    ``H⁺_a → W⁺ h_SM`` (GeV), and the LFV entries (one per allowed μ₃ and
    lepton branch) carrying ``|Q|`` for all three CP-even states.
    """
    pt = image_point(tools, point) if image else point
    rot = point_rotations(tools, pt, check_mixing=not image)
    mass = rot["mass"]
    k = sm_like_index(rot)
    keys_even = ("h1", "h2", "h3")
    vhat = np.array(rot["vevs"]) / np.linalg.norm(rot["vevs"])
    kappa = float(vhat @ rot["R_even"][:, k])          # ≥ 0 by the sign convention

    g = hhh_charged(tools, pt, rot)
    g_hpp = (g[k, 1, 1], g[k, 2, 2])
    m_hp = (mass["Hpm1"], mass["Hpm2"])
    r_gg = diphoton_ratio(g_hpp, 125.25, m_hp, kappa, kappa, M_T, M_W)

    O_Z, O_W = vss_overlaps(rot)
    m_sm = mass[keys_even[k]]
    vss = {}
    for a, key in ((1, "A1"), (2, "A2")):
        vss[f"{key}->Z h"] = vss_width(G_W / (2 * C_W) * O_Z[a, k], mass[key], M_Z, m_sm)
    for a, key in ((1, "Hpm1"), (2, "Hpm2")):
        vss[f"{key}->W h"] = vss_width(G_W / 2 * O_W[a, k], mass[key], M_W, m_sm)

    lfv_rows = []
    for e in lfv_point(tools, pt, rot, mu3_grid):
        lfv_rows.append({"mu3": e["mu3"], "branch": e["branch"], "signs": e["signs"],
                         "kind": e["kind"], "absQ": [np.abs(q) for q in e["Q"]]})
    return {"sm_like": k, "kappa_v": kappa, "r_gammagamma": r_gg,
            "g_hpp": [float(x) for x in g_hpp], "vss": vss, "lfv": lfv_rows,
            "mass": mass, "phi": pt["phi"], "theta": pt["theta"]}

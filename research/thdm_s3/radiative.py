"""ℓ_i → ℓ_j γ (μ → eγ, τ → μγ, τ → eγ) from scalar exchange: one-loop and Barr–Zee two-loop dipole amplitudes.

Pure numerics, no model knowledge.  A model supplies its scalars' masses and
their Yukawa matrices **in the positive-mass charged-lepton basis**, in the
convention

    L ⊃ −ē_L Y^φ e_R φ − ν̄_L Y_C^a e_R H⁺_a + h.c.,      (e, μ, τ) = (0, 1, 2)

of [OST16] Eq. (6) and [HKZ13].  `lfv.scalar_lepton_couplings` +
`lfv.mass_basis_couplings` produce exactly this.

Every amplitude takes the decaying lepton ``i`` and the final one ``j`` as
indices (default μ → e), with ``m_μ → m_i`` in the formulas below.

**Normalization** ([OST16] Eqs. (28)–(29)).  The amplitude is
``T = e ε*_α ū_e m_μ iσ^{αβ}q_β (A_L P_L + A_R P_R) u_μ`` and

    BR(μ → eγ) = 48π³ α (|A_L|² + |A_R|²) / G_F²,

which is ``Γ = α m_μ⁵ (|A_L|²+|A_R|²)/4`` over ``Γ(μ→eνν̄) = G_F² m_μ⁵/192π³``.
[HKZ13]'s Wilson coefficients are ``c = 4π² A``.  That follows from their
Eq. (15) against the above, and their one-loop Eq. (13) equals [OST16]'s
Eq. (19) term by term under it, sign included.

**What is summed.**

1. *One loop, neutral scalars* (`one_loop_neutral`).  This is [HKZ13] Eq. (A3)
   at q² = 0 with the external masses dropped from the denominator
   (m_μ²/m_φ² ≲ 10⁻⁶).  The x, y Feynman parameters then integrate in closed
   form, leaving the two functions `I1`, `I2` of ``t = m_f²/m_φ²``.  These are
   exact in t, so a light scalar is handled with no ``m_f ≪ m_φ`` expansion.
   In that limit they reproduce [OST16] Eq. (30) and [HKZ13] Eq. (16).  One
   exception: [OST16] Eq. (30) multiplies its *non-flip* term ``y_{iμ}/6`` by
   ``m_i/m_μ`` too.  Chirality says that term flips on the external muon line
   and so carries ``m_μ``, not ``m_τ``.  [HKZ13] Eq. (A3) has it so, and
   [OST16]'s own τ→μγ Eq. (19) (where i = τ is the decaying lepton) agrees.
2. *One loop, charged scalars* (`one_loop_charged`).  Massless neutrinos give
   [OST16] Eq. (32): ``A_L = −(Y_C†Y_C)_{eμ}/(192π² m_{H±}²)``, ``A_R = 0``.
3. *Barr–Zee, effective hγγ vertex* (`barr_zee`), for each neutral scalar:
   - fermion loops: [OST16] Eq. (20), first line, which is [HKZ13] Eq. (A4);
   - W loop with its Goldstones: [OST16] Eq. (20), lines 3–4, which is [HKZ13]
     Eq. (A5) with ``G ≡ h``;
   - charged-scalar loop: not in [OST16] or [HKZ13].  It is built from [Ilisie15]
     Eqs. (11) and (13).  Both one-loop φγγ vertices there have the same
     k²-dependence, ``1/(M² − k²x(1−x))``, so the second loop is identical, and
     the charged-scalar term follows from the fermion term by the substitution
     ``m_f y_ff N_c Q² (1 − 2x(1−x)) → (vλ/2) x(1−x)``, with ``m_f → M_{H±}``.
     With the trilinear written ``L ⊃ g φH⁺H⁻`` (feynlag's Feynman-rule
     coefficient, `soft_decays.hhh_charged`) it is ``λ = −g/v``.  This sign is
     *checked*, not assumed: in the heavy-loop limit the scalar/top ratio of
     the [Ilisie15] vertices must equal feynlag's `higgs_diphoton_amplitude`
     ratio (validated against [ChakrabartyEtAl21]), and notebook 07 §2 tests
     that.

Neglected, with reasons:
- Barr–Zee diagrams with an effective φZγ vertex: suppressed by the lepton
  vector coupling ``1 − 4s_W² ≈ 0.08`` ([HKZ13] below Eq. (A9), [OST16] §III A:
  a ~10% effect);
- the H±W∓γ Barr–Zee diagrams of [Ilisie15] Fig. 3;
- [HKZ13]'s "set C".

References
----------
[OST16] Y. Omura, E. Senaha, K. Tobe, "τ- and μ-physics in a general two Higgs
    doublet model with μ−τ flavor violation", Phys. Rev. D 94, 055019 (2016),
    arXiv:1511.08880.
[HKZ13] R. Harnik, J. Kopp, J. Zupan, "Flavor violating Higgs decays",
    JHEP 03 (2013) 026, arXiv:1209.1397.
[Ilisie15] V. Ilisie, "New Barr-Zee contributions to (g−2)_μ in
    two-Higgs-doublet models", JHEP 04 (2015) 077, arXiv:1502.04199.
[Lavoura03] L. Lavoura, "General formulae for f1 → f2 γ", Eur. Phys. J. C 29,
    191 (2003), hep-ph/0302221.  Used in notebook 07 to cross-check the
    charged/neutral one-loop ratio.
[ChakrabartyEtAl21] see `feynlag.pheno.loop`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

__all__ = [
    "I1", "I2", "FH", "FA", "GW", "FS", "w_loop",
    "one_loop_neutral", "one_loop_charged", "barr_zee", "branching_ratio",
]

E, MU, TAU = 0, 1, 2


# --------------------------------------------------------------------------
# one-loop functions
# --------------------------------------------------------------------------

def _near_one(fn_exact, fn_quad, t, eps=0.05):
    # the closed forms divide by (1 - t)^3 or (1 - t)^4: at |t - 1| = 1e-3 they lose ~4 digits,
    # at 0.05 they are good to 1e-10 (notebook 07 §2.1).  Here t = m_f^2/m_phi^2 < 0.01 anyway.
    return fn_quad(t) if abs(t - 1) < eps else fn_exact(t)


def I1(t):
    """``∫₀¹ z(1−z)² / (z + (1−z)t) dz``, the non-flip one-loop function.

    Closed form ``(t³/6 − t² + t ln t + t/2 + 1/3)/(1 − t)⁴``, which → 1/3 as t → 0.
    It is evaluated by quadrature within 0.05 of t = 1, where the closed form cancels.
    """
    return _near_one(
        lambda t: (t**3 / 6 - t**2 + t * math.log(t) + t / 2 + 1 / 3) / (1 - t) ** 4,
        lambda t: quad(lambda z: z * (1 - z) ** 2 / (z + (1 - z) * t), 0, 1)[0], t)


def I2(t):
    """``∫₀¹ (1−z)² / (z + (1−z)t) dz``, the chirality-flip one-loop function.

    Closed form ``(t² − 4t + 2 ln t + 3)/(2(t − 1)³)``, which → ``ln(1/t) − 3/2`` as t → 0.
    """
    return _near_one(
        lambda t: (t**2 - 4 * t + 2 * math.log(t) + 3) / (2 * (t - 1) ** 3),
        lambda t: quad(lambda z: (1 - z) ** 2 / (z + (1 - z) * t), 0, 1)[0], t)


# --------------------------------------------------------------------------
# Barr–Zee functions ([OST16] Eqs. (22)–(24); [Ilisie15] Eq. (13) for FS)
# --------------------------------------------------------------------------
#
# All four have the form ∫ dx R(x) / (x(1−x) − z) · {log(x(1−x)/z), ...}, with a
# removable singularity where x(1−x) = z.  Written in u = x(1−x)/z − 1 they
# become smooth functions of log1p(u)/u, which is evaluated by its series near u = 0.

def _l1(u):
    """``log(1+u)/u``, stable at u → 0."""
    return 1 - u / 2 + u * u / 3 if abs(u) < 1e-4 else math.log1p(u) / u


def _g1(u):
    """``(1 − log(1+u)/u)/u``, stable at u → 0."""
    return 0.5 - u / 3 + u * u / 4 if abs(u) < 1e-4 else (1 - math.log1p(u) / u) / u


def _bz(integrand, z):
    # Every integrand here is symmetric under x → 1−x, so integrate [0, ½] and double.  On
    # [0, 1] the nodes next to the root (1+s)/2 ≈ 1 − z round to x = 1.0 for tiny z (a τ
    # loop), which makes x(1−x) = 0 and log1p(−1) undefined.
    pts = None
    if z < 0.25:                               # where x(1−x) = z: integrable, smooth in u
        pts = [(1 - math.sqrt(1 - 4 * z)) / 2]
    return 2 * quad(integrand, 0, 0.5, points=pts, limit=200, epsabs=0, epsrel=1e-10)[0]


@lru_cache(maxsize=None)
def FH(z):
    """[OST16] Eq. (22): ``(z/2)∫ (1−2x(1−x))/(x(1−x)−z) log(x(1−x)/z)``."""
    return 0.5 * _bz(lambda x: (1 - 2 * x * (1 - x)) * _l1(x * (1 - x) / z - 1), z)


@lru_cache(maxsize=None)
def FA(z):
    """[OST16] Eq. (23): ``(z/2)∫ 1/(x(1−x)−z) log(x(1−x)/z)``."""
    return 0.5 * _bz(lambda x: _l1(x * (1 - x) / z - 1), z)


@lru_cache(maxsize=None)
def GW(z):
    """[OST16] Eq. (24): ``−(z/2)∫ 1/(x(1−x)−z) [1 − z/(x(1−x)−z) log(x(1−x)/z)]``."""
    return -0.5 * _bz(lambda x: _g1(x * (1 - x) / z - 1), z)


@lru_cache(maxsize=None)
def FS(z):
    """Charged-scalar loop: ``½∫ x(1−x)/(x(1−x)−z) log(x(1−x)/z)``.

    This is minus [Ilisie15] Eq. (27).  It enters `barr_zee` as
    ``(vλ/2m_φ²) FS(M²/m_φ²)``, the image of ``(m_f y_ff N_cQ²/m_f²) FH(m_f²/m_φ²)``
    (module docstring).
    """
    return 0.5 * _bz(lambda x: (x * (1 - x) / z) * _l1(x * (1 - x) / z - 1), z)


def w_loop(m_phi, m_w):
    """The W-loop bracket of [OST16] Eq. (20), lines 3–4 (= [HKZ13] Eq. (A5))."""
    x = m_w**2 / m_phi**2
    fh, fa = FH(x), FA(x)
    return 3 * fh + 23 / 4 * fa + 3 / 4 * GW(x) + m_phi**2 / (2 * m_w**2) * (fh - fa)


# --------------------------------------------------------------------------
# amplitudes
# --------------------------------------------------------------------------

def one_loop_neutral(Y, masses, m_phi, i=MU, j=E):
    """``(A_L, A_R)`` from one neutral scalar, summed over internal ℓ_f = e, μ, τ.

    [HKZ13] Eq. (A3) for ℓ_i → ℓ_j γ (defaults i = μ decaying, j = e), after the
    x, y integration:

        c_L = Σ_f [ ½ I1(t_f) (m_j Y_jf Y*_if + m_i Y*_fj Y_fi)
                    + I2(t_f) m_f Y*_fj Y*_if ] / (4 m_i m_φ²),

    with ``c_R`` the same with ``Y → Y†``, and ``A = c/4π²``.  ``Y`` is a complex 3×3
    in the (e, μ, τ) mass basis and ``masses`` are the positive lepton masses.
    """
    m_j, m_i = masses[j], masses[i]

    def c(Y):
        out = 0j
        for f in range(3):
            t = masses[f] ** 2 / m_phi**2
            out += (0.5 * I1(t) * (m_j * Y[j, f] * np.conj(Y[i, f])
                                   + m_i * np.conj(Y[f, j]) * Y[f, i])
                    + I2(t) * masses[f] * np.conj(Y[f, j]) * np.conj(Y[i, f]))
        return out / (4 * m_i * m_phi**2)

    Y = np.asarray(Y, dtype=complex)
    return c(Y) / (4 * math.pi**2), c(Y.conj().T) / (4 * math.pi**2)


def one_loop_charged(Yc, m_hpm, i=MU, j=E):
    """``(A_L, A_R)`` from one charged scalar with massless neutrinos ([OST16] Eq. (32)),
    for ℓ_i → ℓ_j γ: ``A_L = −(Y_C†Y_C)_{ji}/(192π² m_{H±}²)``."""
    Yc = np.asarray(Yc, dtype=complex)
    return -(Yc.conj().T @ Yc)[j, i] / (192 * math.pi**2 * m_hpm**2), 0j


def barr_zee(y_mue, y_emu, m_phi, alpha, v, m_w=None, kappa_w=0.0,
             fermions=(), charged=(), m_mu=None):
    """Barr–Zee ``{"fermion", "W", "Hpm"}: (A_L, A_R)`` for one neutral scalar φ.

    Args:
        y_mue, y_emu: φ's ``Y_{μe}`` (coefficient of ``μ̄_L e_R``) and ``Y_{eμ}``.
            For a general ℓ_i → ℓ_j γ pass ``Y_{ij}`` and ``Y_{ji}``, with ``m_mu``
            the decaying lepton's mass ``m_i``.
        kappa_w: ``g_{φVV}/g^{SM}``.
        fermions: ``(N_c, Q, m_f, y_ff)`` with ``y_ff`` φ's (complex) ``f̄_L f_R``
            coupling in the same convention.  An imaginary part is a pseudoscalar
            coupling and enters through ``FA``.
        charged: ``(g, M)``, where ``g`` is the ``φH⁺H⁻`` coefficient in L (GeV).

    [OST16] Eqs. (20)–(21): A_R is A_L with ``y*_{μe} → y_{eμ}`` and ``i → −i``.
    """
    pre_l, pre_r = np.conj(y_mue), y_emu
    out = {"fermion": [0j, 0j], "W": [0j, 0j], "Hpm": [0j, 0j]}
    for n_c, q, m_f, y in fermions:
        x = m_f**2 / m_phi**2
        k = -n_c * q**2 * alpha / (8 * math.pi**3 * m_mu * m_f)
        fh, fa = FH(x), FA(x)
        out["fermion"][0] += k * pre_l * (y.real * fh - 1j * y.imag * fa)
        out["fermion"][1] += k * pre_r * (y.real * fh + 1j * y.imag * fa)
    if kappa_w:
        k = alpha * kappa_w / (16 * math.pi**3 * m_mu * v) * w_loop(m_phi, m_w)
        out["W"] = [k * pre_l, k * pre_r]
    for g, M in charged:
        k = alpha * g / (16 * math.pi**3 * m_mu * m_phi**2) * FS(M**2 / m_phi**2)
        out["Hpm"][0] += k * pre_l
        out["Hpm"][1] += k * pre_r
    return {key: tuple(val) for key, val in out.items()}


def branching_ratio(A_L, A_R, alpha, G_F, br_lnn=1.0):
    """[OST16] Eq. (29): ``48π³α(|A_L|² + |A_R|²)/G_F² × B(ℓ_i → ℓ_k νν̄)``.

    The ratio is to ``Γ(ℓ_i → ℓ_k νν̄) = G_F² m_i⁵/192π³``, so ``br_lnn = 1`` for the
    muon and ``B(τ → ℓνν̄)`` for the tau ([OST16] Eq. (17)).
    """
    return 48 * math.pi**3 * alpha * (abs(A_L) ** 2 + abs(A_R) ** 2) / G_F**2 * br_lnn

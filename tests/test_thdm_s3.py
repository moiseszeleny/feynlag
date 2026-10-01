"""Phase 6 stress test: 3HDM with S₃ symmetry.

The S₃-invariant three-Higgs-doublet potential, with (H1, H2) an S₃ doublet
and HS the S₃ singlet, built entirely from the library's own CG products:

V = μ1² (H1†H1 + H2†H2) + μ0² HS†HS
  + λ1 (H1†H1 + H2†H2)²          [1 of 2⊗2, squared]
  + λ2 (H1†H2 − H2†H1)²          [1' of 2⊗2, squared]
  + λ3 [(H1†H1 − H2†H2)² + (H1†H2 + H2†H1)²]   [2 of 2⊗2, squared]
  + λ4 [HS†H_CG-doublet contraction + h.c.]
  + λ5 (HS†HS)(H1†H1 + H2†H2)
  + λ6 [(HS†H1)(H1†HS) + (HS†H2)(H2†HS)]
  + λ7 [(HS†H1)² + (HS†H2)² + h.c.]
  + λ8 (HS†HS)²

Physics checks:
- every term passes gauge (SU2×U1) AND S₃ invariance; a forbidden term fails;
- with all three VEVs, the tadpole conditions over-constrain (μ0², μ1²) and
  force the alignment v1² = 3 v2² (or the equivalent branch) — the relation
  the user's 3HDM-S₃ papers rely on (v₁ = √3 v₂);
- CP-even 3×3 mass matrix is symmetric with the expected structure.
"""

import sympy as sp
import pytest

from feynlag import (
    ExternalParameter, InternalParameter, Lagrangian, Model, S3, SU2,
    Scalar, U1, dag,
)


@pytest.fixture(scope="module")
def s3_model():
    return _build_s3_model()


def _build_s3_model(soft=False):
    """The S₃-invariant 3HDM; with ``soft=True`` also the four soft quadratics.

    The soft terms are the S₃-breaking, gauge-invariant dimension-2 operators:
    the 2 of 2⊗2, ``(x11 − x22, −(x12 + x21))``, and the doublet ``H_S†H_i +
    h.c.``.  They are returned as a 7th tuple entry ``{name: parameter}`` (empty
    when ``soft=False``) so the exact-S₃ tests see the same 6-tuple as before.
    """
    gw = ExternalParameter("gw", 0.6535, positive=True)
    g1 = ExternalParameter("g1", 0.3580, positive=True)
    SU2L, U1Y = SU2("SU2L", coupling=gw), U1("U1Y", coupling=g1)
    s3 = S3()

    v1 = ExternalParameter("v1", 200.0, positive=True, unit_dim=1)
    v2 = ExternalParameter("v2", 115.0, positive=True, unit_dim=1)
    vS = ExternalParameter("vS", 80.0, positive=True, unit_dim=1)
    lams = {k: ExternalParameter(f"lm{k}", 0.05 * k) for k in range(1, 9)}
    mu0sq = InternalParameter("mu0sq", unit_dim=2)
    mu1sq = InternalParameter("mu1sq", unit_dim=2)

    def doublet(name):
        return Scalar(name, reps={SU2L: 2, U1Y: sp.Rational(1, 2)},
                      component_names=[f"{name}p", f"{name}0"])

    H1, H2, HS = doublet("H1"), doublet("H2"), doublet("HS")
    s3.assign("2", H1, H2)
    s3.assign("1", HS)

    H1.expand_vev({H1.components[1]: v1})
    H2.expand_vev({H2.components[1]: v2})
    HS.expand_vev({HS.components[1]: vS})

    def bra(a, b):
        return (dag(a) * b.mat)[0]

    x11, x22, x12, x21 = bra(H1, H1), bra(H2, H2), bra(H1, H2), bra(H2, H1)
    s11, s22 = bra(HS, H1), bra(HS, H2)          # HS†H_i : S3 doublet
    s11c, s22c = bra(H1, HS), bra(H2, HS)
    sss = bra(HS, HS)

    # CG contractions of the (H1,H2) doublet with itself
    # 1  : x11 + x22 ; 1' : x12 − x21 ; 2 : (x11 − x22, −(x12 + x21))
    cg = s3.doublet_product((sp.Symbol("_a1"), sp.Symbol("_a2")),
                            (sp.Symbol("_b1"), sp.Symbol("_b2")))
    # build with actual bilinears: bra-side (H1†, H2†) and ket-side (H1, H2)
    sub = {sp.Symbol("_a1") * sp.Symbol("_b1"): x11,
           sp.Symbol("_a1") * sp.Symbol("_b2"): x12,
           sp.Symbol("_a2") * sp.Symbol("_b1"): x21,
           sp.Symbol("_a2") * sp.Symbol("_b2"): x22}

    def cg_sub(expr):
        return sp.expand(expr).subs(sub, simultaneous=True)

    inv1 = cg_sub(cg["1"])            # x11 + x22
    inv1p = cg_sub(cg["1p"])          # x12 − x21
    d2_1, d2_2 = cg_sub(cg["2"][0]), cg_sub(cg["2"][1])

    # λ4 invariant: (HS†H)₂ ⊗ (H†H)₂ → 1 :  s11·d1 + s22·d2, + h.c.
    lam4_term = s11 * d2_1 + s22 * d2_2
    lam4_term = lam4_term + sp.conjugate(lam4_term)
    # λ7: (HS†H)₂ ⊗ (HS†H)₂ → 1 : s11² + s22², + h.c.
    lam7_term = s11**2 + s22**2
    lam7_term = lam7_term + sp.conjugate(lam7_term)

    l = {k: lams[k].s for k in lams}
    V = (mu1sq.s * inv1 + mu0sq.s * sss
         + l[1] * inv1**2
         + l[2] * inv1p**2
         + l[3] * (d2_1**2 + d2_2**2)
         + l[4] * lam4_term
         + l[5] * sss * inv1
         + l[6] * (s11 * s11c + s22 * s22c)
         + l[7] * lam7_term
         + l[8] * sss**2)

    softs = {}
    if soft:
        herm = lambda e: e + sp.conjugate(e)
        for name, struct in (("mD1sq", d2_1), ("mD2sq", d2_2),
                             ("mS1sq", herm(s11)), ("mS2sq", herm(s22))):
            softs[name] = ExternalParameter(name, 0.0, unit_dim=2)
            V += softs[name].s * struct

    L = Lagrangian().add(-V, sector="potential")
    model = Model("3HDM-S3", gauge_groups=[SU2L, U1Y], discrete_groups=[s3],
                  fields=[H1, H2, HS],
                  parameters=[gw, g1, v1, v2, vS, mu0sq, mu1sq,
                              *lams.values(), *softs.values()],
                  lagrangian=L)
    out = (model, s3, (H1, H2, HS), (v1, v2, vS), (mu0sq, mu1sq), l)
    return out + (softs,) if soft else out


def test_invariance_full_potential(s3_model):
    model, s3, fields, vevs, mus, l = s3_model
    report = model.check_invariance()
    assert report.ok, report.failures


def test_forbidden_term_fails(s3_model):
    from feynlag import check_discrete_invariance
    model, s3, (H1, H2, HS), *_ = s3_model
    bad = (dag(HS) * H1.mat)[0] * (dag(H1) * H1.mat)[0]
    bad = bad + sp.conjugate(bad)
    ok, _ = check_discrete_invariance(bad, s3)
    assert not ok


def test_tadpole_alignment_sqrt3(s3_model):
    """Solving t2, tS for (μ0², μ1²) leaves a residual third condition whose
    non-trivial solution is the S₃ alignment  v² ratio = 3.

    Note on basis: the literature (Gómez-Bock et al.) quotes v₁ = √3 v₂; in
    feynlag's real-orthogonal S₃ basis the roles of the doublet components
    are swapped (an equivalent irrep, related by the reflection), so the
    alignment appears as v₂ = √3 v₁ — the ratio squared is 3 either way."""
    model, s3, fields, (v1, v2, vS), (mu0sq, mu1sq), l = s3_model

    tadpoles = model.tadpoles()
    t1, t2, tS = tadpoles[v1.s], tadpoles[v2.s], tadpoles[vS.s]

    # solve the v2 and vS conditions for the two mass parameters
    sol = sp.solve([sp.Eq(t2, 0), sp.Eq(tS, 0)], [mu0sq.s, mu1sq.s],
                   dict=True)
    assert len(sol) == 1
    residual = sp.factor(sp.expand(t1.subs(sol[0])))

    # the residual must vanish only on alignment: find its v1 solutions
    solutions = sp.solve(sp.Eq(residual, 0), v1.s)
    ratios = set()
    for s_v1 in solutions:
        r = sp.simplify((s_v1 / v2.s) ** 2)
        if not r.free_symbols:               # pure number
            ratios.add(sp.nsimplify(r))
    assert ratios & {sp.Integer(3), sp.Rational(1, 3)}, (solutions, ratios)


def test_cp_even_mass_matrix_structure(s3_model):
    model, s3, fields, (v1, v2, vS), (mu0sq, mu1sq), l = s3_model
    # impose the alignment (v2 = √3 v1 in this basis) and solve all
    # tadpoles consistently
    align = {v1.s: v2.s / sp.sqrt(3)}

    tadpoles = model.tadpoles()
    sol = sp.solve([sp.Eq(tadpoles[v2.s].subs(align), 0),
                    sp.Eq(tadpoles[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]
    # the v1 tadpole is then automatically satisfied
    assert sp.simplify(tadpoles[v1.s].subs(align).subs(sol)) == 0

    h1, h2, hS = (sp.Symbol("H10_r", real=True),
                  sp.Symbol("H20_r", real=True),
                  sp.Symbol("HS0_r", real=True))
    M = model.mass_matrix([h1, h2, hS])
    M = M.subs(sol).subs(align)
    M = M.applyfunc(lambda e: sp.simplify(sp.expand(e)))

    # symmetric, and no vanishing diagonal in general
    assert sp.simplify(M - M.T) == sp.zeros(3, 3)
    assert M[0, 0] != 0 and M[1, 1] != 0 and M[2, 2] != 0


# ---------------------------------------------------------------------------
# All three scalar sectors + the geometric rotation.
#
# These pin what previously lived only as print statements inside
# examples/THDM_S3_Tutorial.ipynb §6-§9.  Built here from this file's own
# fixture, deliberately NOT importing research/thdm_s3/model.py — research
# code churns, and the suite must not be hostage to it (see research/README.md).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s3_sectors(s3_model):
    """(M_S, M_A, M_C, R, v-symbols) on the aligned vacuum, tadpoles solved."""
    model, s3, (H1, H2, HS), (v1, v2, vS), (mu0sq, mu1sq), l = s3_model
    align = {v1.s: v2.s / sp.sqrt(3)}
    tad = model.tadpoles()
    sol = sp.solve([sp.Eq(tad[v2.s].subs(align), 0),
                    sp.Eq(tad[vS.s].subs(align), 0)],
                   [mu0sq.s, mu1sq.s], dict=True)[0]

    def build(fields, charged=False):
        M = (model.mass_matrix(fields, charged=True) if charged
             else model.mass_matrix(fields))
        return M.subs(sol).subs(align).applyfunc(
            lambda e: sp.simplify(sp.expand(e)))

    M_S = build([sp.Symbol(f"{n}0_r", real=True) for n in ("H1", "H2", "HS")])
    M_A = build([sp.Symbol(f"{n}0_i", real=True) for n in ("H1", "H2", "HS")])
    M_C = build([H1.components[0], H2.components[0], HS.components[0]],
                charged=True)

    # the Gomez-Bock-Mondragon-Perez-Martinez geometric ansatz, [GomezBock21]
    # Eq. (29): first column is the vacuum direction.
    v12 = sp.sqrt(v1.s**2 + v2.s**2)
    vtot = sp.sqrt(v1.s**2 + v2.s**2 + vS.s**2)
    cphi, sphi = v1.s / v12, v2.s / v12
    cth, sth = vS.s / vtot, v12 / vtot
    R = sp.simplify(sp.Matrix([
        [sth * cphi, -sphi, -cth * cphi],
        [sth * sphi, cphi, -cth * sphi],
        [cth, 0, sth],
    ]).subs(align))
    return M_S, M_A, M_C, R, (v1, v2, vS), l


def test_pseudoscalar_and_charged_mass_matrices(s3_sectors):
    """M_A and M_C are symmetric with non-vanishing diagonals, like M_S."""
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    for M in (M_A, M_C):
        assert sp.simplify(M - M.T) == sp.zeros(3, 3)
        assert all(M[i, i] != 0 for i in range(3))


def test_geometric_rotation_isolates_goldstones(s3_sectors):
    """RᵀMR puts an EXACT zero at [0,0] in the CP-odd and charged sectors.

    One Goldstone each (eaten by Z and W±); the ansatz is purely geometric —
    built from the VEVs, with no reference to the quartics — so this vanishing
    is a nontrivial statement about the potential, not an identity.
    """
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    assert sp.simplify(R.T * R) == sp.eye(3)          # orthogonal

    for M in (M_A, M_C):
        D = sp.simplify(R.T * M * R)
        assert D[0, 0] == 0
        # and fully diagonal in the remaining 2x2
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert sp.simplify(D[i, j]) == 0


def test_cp_even_block_diagonalizes_to_2x2(s3_sectors):
    """D_S = RᵀM_S R keeps exactly one off-diagonal pair, the (0,2) block."""
    M_S, M_A, M_C, R, vevs, l = s3_sectors
    D = sp.simplify(R.T * M_S * R)
    assert sp.simplify(D[0, 1]) == 0 and sp.simplify(D[1, 2]) == 0
    assert sp.simplify(D[0, 2]) != 0                   # the surviving mixing
    assert sp.simplify(D[0, 2] - D[2, 0]) == 0


def test_block_overlap_is_the_hvv_coupling(s3_sectors):
    """A CP-even state's hVV coupling² is its overlap with block index 0.

    The coupling of a mass eigenstate to W/Z is its component along the vacuum
    direction v̂ = (v1, v2, vS)/v, because every doublet carries the same gauge
    charges.  R's first column IS v̂, so that direction is index 0 of D_S = RᵀM_SR
    and, with h0 decoupled, the two block states share the SM strength as

        g²± = ½ (1 ± (a−c)/√((a−c)² + 4b²)),   a, b, c = D_S[0,0], D_S[0,2], D_S[2,2].

    Checked against the definition (numerically diagonalize the *un-rotated*
    M_S, project each eigenvector on v̂) at random points, not derived from the
    block; and the opposite assignment fails, so the test has teeth.  The
    research scan (research/thdm_s3/model.py::hvv_function) tags every viable
    point with this quantity and relies on it.
    """
    import numpy as np
    M_S, M_A, M_C, R, (v1, v2, vS), l = s3_sectors
    D = sp.simplify(R.T * M_S * R)
    a, b, c = D[0, 0], D[0, 2], D[2, 2]

    rng = np.random.default_rng(20260920)
    wrong_matches = 0
    for _ in range(8):
        vals = {l[k]: float(rng.uniform(-2.0, 2.0)) for k in l}
        vals[v2.s] = float(rng.uniform(40.0, 200.0))
        vals[vS.s] = float(rng.uniform(40.0, 200.0))
        Mn = np.array(M_S.subs(vals).evalf().tolist(), dtype=float)
        w, U = np.linalg.eigh((Mn + Mn.T) / 2)
        v1n = vals[v2.s] / np.sqrt(3.0)
        vhat = np.array([v1n, vals[v2.s], vals[vS.s]])
        vhat /= np.linalg.norm(vhat)
        true_g2 = np.sort((U.T @ vhat) ** 2)

        an, bn, cn = (float(x.subs(vals)) for x in (a, b, c))
        root = np.hypot(an - cn, 2.0 * bn)
        ours = np.sort([0.0, 0.5 * (1 + (an - cn) / root),
                        0.5 * (1 - (an - cn) / root)])
        assert np.allclose(true_g2, ours, atol=1e-9), (true_g2, ours)
        assert abs(true_g2.sum() - 1.0) < 1e-12               # Σ g² = (g^SM)²

        # the eigenvalue each overlap belongs to: (tr + root)/2 carries the + sign
        eig_plus = 0.5 * (an + cn + root)
        assert any(np.isclose(eig_plus, x, rtol=1e-9) for x in w)

        # swapping the two block states' couplings must be caught whenever they differ
        g_plus = 0.5 * (1 + (an - cn) / root)
        k_plus = int(np.argmin(np.abs(w - eig_plus)))
        assert np.isclose((U[:, k_plus] @ vhat) ** 2, g_plus, atol=1e-9)
        if abs(g_plus - 0.5) > 1e-3:
            wrong_matches += int(np.isclose((U[:, k_plus] @ vhat) ** 2,
                                            1 - g_plus, atol=1e-6))
    assert wrong_matches == 0


def test_delta_is_minus_the_block_angle(s3_sectors):
    """R_S = R·R_H(δ) diagonalizes M_S for δ = −ψ, and for neither ψ nor ψ−θ_v.

    Two conventions compose here and the research code got their combination
    wrong (it used δ = ψ − θ_v), so both halves are pinned:

    1. ``R_A(φ,θ)·R_H(δ) = R_A(φ,θ+δ)`` identically, so δ is the rotation *on
       top of* the geometric basis — there is no θ_v to subtract from a block
       angle that is already measured from that basis.
    2. ``R_H(δ)``'s (0,2) submatrix is ``rotation_2x2(δ)`` and the physical basis
       needs ``R_Hᵀ B R_H`` diagonal, while `solve_mixing_angle_2x2` returns ψ
       with ``R B Rᵀ`` diagonal — the transpose.  Hence δ = −ψ.

    The wrong candidates are asserted to *fail*, so the test has teeth.
    """
    import math

    import numpy as np

    from feynlag.vacuum.diagonalize import rotation_2x2, solve_mixing_angle_2x2

    M_S, M_A, M_C, R, (v1, v2, vS), l = s3_sectors

    def R_H(d):
        c, s = sp.cos(d), sp.sin(d)
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # (1) the composition identity, symbolically
    phi, th, d = sp.symbols("phi theta d")
    R_A = sp.Matrix([[sp.sin(th) * sp.cos(phi), -sp.sin(phi), -sp.cos(th) * sp.cos(phi)],
                     [sp.sin(th) * sp.sin(phi), sp.cos(phi), -sp.cos(th) * sp.sin(phi)],
                     [sp.cos(th), 0, sp.sin(th)]])
    shifted = R_A.subs(th, th + d)
    assert sp.simplify(R_A * R_H(d) - shifted) == sp.zeros(3, 3)

    # the VV coupling of each column: overlap with the vacuum direction, R_A's
    # first column.  cos δ, 0, sin δ -- h_0 gauge-phobic for ANY δ.
    nhat = R_A[:, 0]
    overlaps = sp.simplify((R_A * R_H(d)).T * nhat)
    assert sp.simplify(overlaps[0] - sp.cos(d)) == 0
    assert sp.simplify(overlaps[1]) == 0
    assert sp.simplify(overlaps[2] - sp.sin(d)) == 0

    # (2) the sign, numerically on the real mass matrix
    D = sp.simplify(R.T * M_S * R)
    block = sp.Matrix([[D[0, 0], D[0, 2]], [D[2, 0], D[2, 2]]])
    psi_expr = solve_mixing_angle_2x2(block)[0]
    assert sp.simplify(rotation_2x2(-psi_expr) - rotation_2x2(psi_expr).T) == sp.zeros(2, 2)

    rng = np.random.default_rng(20260920)
    for _ in range(5):
        vals = {l[k]: float(rng.uniform(-2.0, 2.0)) for k in l}
        vals[v2.s] = float(rng.uniform(40.0, 200.0))
        vals[vS.s] = float(rng.uniform(40.0, 200.0))
        psi = float(psi_expr.subs(vals))
        v1n = vals[v2.s] / math.sqrt(3.0)
        theta_v = math.atan2(math.hypot(v1n, vals[v2.s]), vals[vS.s])
        Rn = np.array(R.subs(vals).evalf().tolist(), dtype=float)
        Mn = np.array(M_S.subs(vals).evalf().tolist(), dtype=float)

        def off(delta):
            RH = np.array(R_H(sp.Float(delta)).evalf().tolist(), dtype=float)
            RS = Rn @ RH
            Dn = RS.T @ Mn @ RS
            return np.abs(Dn - np.diag(np.diag(Dn))).max() / np.abs(np.diag(Dn)).max()

        assert off(-psi) < 1e-12                      # the fix
        assert off(psi) > 1e-3                        # the un-transposed sign
        assert off(psi - theta_v) > 1e-3              # the old research formula

    # the branch of atan pins the labelling: |cos δ| >= |sin δ| always, i.e.
    # h_1 IS the state carrying the larger hVV coupling.
    assert sp.atan(sp.Symbol("x", real=True)).is_real
    for x in (-50.0, -1.0, 0.0, 1.0, 50.0):
        assert abs(math.atan(x) / 2) <= math.pi / 4


def test_masses_match_gomezbock_closed_forms(s3_sectors):
    """feynlag's derived masses reproduce [GomezBock21] Eqs. (30)-(33) exactly.

    This is the end-to-end validation of the 3HDM-S₃ chain against published
    closed forms — potential, tadpoles, mass matrices and rotation at once —
    and it is what *fixes the parameter dictionary* to the literature:

        a=2λ₈, b=λ₅, c=2λ₁, d=2λ₂, e=−λ₄, f=λ₆, g=2λ₃, h=2λ₇

    The e = −λ₄ sign matters: feynlag's real-orthogonal S₃ doublet basis is not
    the literature's (the λ₄ invariant transcribed literally from [DasDey14]
    is not even S₃-invariant here).  Because the [DasDey14] boundedness and
    unitarity conditions involve λ₄ only as |λ₄| or λ₄², they nevertheless
    transfer to feynlag's λ's unchanged — which is what
    research/thdm_s3/constraints.py relies on.

    [GomezBock21] M. Gómez-Bock, M. Mondragón, A. Pérez-Martínez,
        Eur. Phys. J. C 81, 942 (2021), arXiv:2102.02800,
        doi:10.1140/epjc/s10052-021-09731-3.
    [DasDey14] D. Das, U. K. Dey, Phys. Rev. D 89, 095025 (2014),
        arXiv:1404.2491, doi:10.1103/PhysRevD.89.095025.
    """
    M_S, M_A, M_C, R, (v1, v2, vS), l = s3_sectors
    D_A = sp.simplify(R.T * M_A * R)
    D_C = sp.simplify(R.T * M_C * R)

    # the paper's vacuum parametrization: v12 = v sinθ, vS = v cosθ, with the
    # alignment fixing v2 = √3 v12/2 in feynlag's basis ([GomezBock21] Eq. 24)
    v, th = sp.symbols("v theta", positive=True)
    vac = {v2.s: v * sp.sin(th) * sp.sqrt(3) / 2, vS.s: v * sp.cos(th)}

    a, b, c, d, e, f, g, h = sp.symbols("a b c d e f g h")
    dictionary = {a: 2 * l[8], b: l[5], c: 2 * l[1], d: 2 * l[2],
                  e: -l[4], f: l[6], g: 2 * l[3], h: 2 * l[7]}

    published = {                                        # [GomezBock21]
        D_A[1, 1]: -v**2 * ((d + g) * sp.sin(th)**2
                            + sp.Rational(5, 4) * e * sp.sin(2 * th)
                            + h * sp.cos(th)**2),                     # Eq. (30)
        D_A[2, 2]: -v**2 * (e / 2 * sp.tan(th) + h),                  # Eq. (31)
        D_C[1, 1]: -v**2 / 4 * (5 * e * sp.sin(2 * th)
                                + 2 * (f + h) * sp.cos(th)**2
                                + 4 * g * sp.sin(th)**2),             # Eq. (32)
        D_C[2, 2]: -v**2 / 2 * (e * sp.tan(th) + (f + h)),            # Eq. (33)
    }

    for derived, closed_form in published.items():
        diff = sp.simplify(sp.expand_trig(sp.simplify(
            derived.subs(vac) - closed_form.subs(dictionary))))
        assert diff == 0, diff

    # and the opposite sign genuinely fails, so the test has teeth
    wrong = {**dictionary, e: l[4]}
    bad = sp.simplify(sp.expand_trig(sp.simplify(
        D_A[2, 2].subs(vac) - published[D_A[2, 2]].subs(wrong))))
    assert bad != 0


def test_aligned_vacuum_can_meet_the_electroweak_scale(s3_model):
    """The alignment and √Σvᵢ² = 246 GeV are simultaneously satisfiable.

    Regression for a real bug found in examples/THDM_S3_Tutorial.ipynb: it
    picks (v1,v2,vS) = (200,115,80) so that √Σvᵢ² ≈ 246 ([GomezBock21] Eq. 8),
    then imposes the alignment as the substitution v1 → v2/√3, which *replaces*
    v1 = 200 by 66.4 and silently drops the vacuum to 155 GeV.  Imposing both
    conditions at once leaves θ free: v12 = v sinθ, vS = v cosθ, v1 = v12/2,
    v2 = √3 v12/2.
    """
    import math
    v_ew = 246.0
    for theta in (0.3, 0.8, 1.0286, 1.4):
        v12 = v_ew * math.sin(theta)
        v1, v2, vS = v12 / 2, v12 * math.sqrt(3) / 2, v_ew * math.cos(theta)
        assert math.isclose(v2 / v1, math.sqrt(3), rel_tol=1e-12)     # alignment
        assert math.isclose(math.sqrt(v1**2 + v2**2 + vS**2), v_ew,
                            rel_tol=1e-12)                            # Eq. (8)

    # the tutorial's aligned point demonstrably misses the scale
    v2_t, vS_t = 115.0, 80.0
    v1_t = v2_t / math.sqrt(3)
    assert not math.isclose(math.sqrt(v1_t**2 + v2_t**2 + vS_t**2), v_ew,
                            rel_tol=1e-3)


# ---------------------------------------------------------------------------
# The SOFT-BROKEN vacuum.
#
# Four S₃-breaking quadratics release the √3 alignment: the tadpoles become
# ordinary equations and a general (v1, v2, vS) is a stationary point.  These
# pin the structural facts research/thdm_s3/scan_soft.py rests on, built from
# the fixture's own model (never importing research code).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s3_soft():
    """Soft model, tadpoles solved for (μ0², μ1², mD1²), numeric mass matrices."""
    import numpy as np

    model, s3, (H1, H2, HS), vevs, (mu0sq, mu1sq), l, softs = \
        _build_s3_model(soft=True)
    v = [p.s for p in vevs]
    tad = model.tadpoles()
    sol = sp.solve([sp.Eq(tad[x], 0) for x in v],
                   [mu0sq.s, mu1sq.s, softs["mD1sq"].s], dict=True)[0]

    fields = {"S": [sp.Symbol(f"{n}0_r", real=True) for n in ("H1", "H2", "HS")],
              "A": [sp.Symbol(f"{n}0_i", real=True) for n in ("H1", "H2", "HS")]}
    Ms = {k: model.mass_matrix(f).subs(sol) for k, f in fields.items()}
    Ms["C"] = model.mass_matrix([H1.components[0], H2.components[0],
                                 HS.components[0]], charged=True).subs(sol)
    free = [softs[n].s for n in ("mD2sq", "mS1sq", "mS2sq")]
    args = [l[k] for k in range(1, 9)] + v + free
    fn = sp.lambdify(args, [Ms["S"], Ms["A"], Ms["C"], sol[softs["mD1sq"].s]],
                     "numpy")

    def at(lam, theta, phi, free_vals):
        vv = (246 * np.sin(theta) * np.cos(phi), 246 * np.sin(theta) * np.sin(phi),
              246 * np.cos(theta))
        M_S, M_A, M_C, mD1 = fn(*lam, *vv, *free_vals)
        return (np.array(M_S, float), np.array(M_A, float), np.array(M_C, float),
                float(mD1), np.array(vv))

    return model, tad, softs, (mu0sq, mu1sq), v, sol, at


def _geometric_R(vv):
    import numpy as np
    v1, v2, vS = vv
    v12, vt = np.hypot(v1, v2), np.linalg.norm(vv)
    cphi, sphi, cth, sth = v1 / v12, v2 / v12, vS / vt, v12 / vt
    return np.array([[sth * cphi, -sphi, -cth * cphi],
                     [sth * sphi, cphi, -cth * sphi],
                     [cth, 0.0, sth]])


def test_soft_tadpoles_are_regular_only_when_solved_for_mD1(s3_soft):
    """Which soft term the tadpoles fix is not cosmetic.

    Solving for mD2² divides by v1² − v2², singular on the φ = 45° line in the
    middle of the scan's domain; solving for mD1² divides only by v1·v2 (and
    μ0² by vS), regular throughout φ ∈ (0, π/3).
    """
    model, tad, softs, (mu0sq, mu1sq), (v1, v2, vS), sol, at = s3_soft
    for k, e in sol.items():
        den = sp.factor(sp.denom(sp.together(e)))
        assert den.free_symbols <= {v1, v2, vS}
        assert not sp.factor(den).has(v1 - v2) and not sp.factor(den).has(v1 + v2)
        assert sp.simplify(den.subs(v1, v2)) != 0

    sol2 = sp.solve([sp.Eq(tad[x], 0) for x in (v1, v2, vS)],
                    [mu0sq.s, mu1sq.s, softs["mD2sq"].s], dict=True)[0]
    den2 = sp.denom(sp.together(sol2[softs["mD2sq"].s]))
    assert sp.simplify(den2.subs(v1, v2)) == 0


def test_soft_breaking_off_recovers_the_alignment(s3_soft):
    """On the √3-aligned vacuum with the free soft terms at zero, mD1² = 0.

    So the exact-S₃ model is the φ = π/3, zero-soft slice of the soft one —
    the limit notebook 01's scan lives in.
    """
    model, tad, softs, mus, (v1, v2, vS), sol, at = s3_soft
    zero = {softs[n].s: 0 for n in ("mD2sq", "mS1sq", "mS2sq")}
    mD1 = sol[softs["mD1sq"].s].subs(zero).subs(v2, sp.sqrt(3) * v1)
    assert sp.simplify(mD1) == 0


def test_soft_vacuum_goldstones_and_hvv_sum_rule(s3_soft):
    """For a GENERAL vacuum the geometric rotation still isolates the Goldstones.

    R's first column is the vacuum direction for any (θ, φ), so RᵀM_A R and
    RᵀM_C R have a vanishing first row.  The CP-even sector is now fully mixed:
    every state couples to VV (overlap with v̂), and the overlaps² sum to 1.
    """
    import numpy as np

    at = s3_soft[-1]
    rng = np.random.default_rng(11)
    for _ in range(5):
        lam = rng.uniform(-2, 2, 8)
        theta, phi = rng.uniform(0.2, 1.3), rng.uniform(0.1, 1.0)
        M_S, M_A, M_C, _, vv = at(lam, theta, phi, rng.uniform(-2e5, 2e5, 3))
        R = _geometric_R(vv)
        for M in (M_A, M_C):
            D = R.T @ M @ R
            assert np.abs(D[0]).max() < 1e-9 * np.abs(D).max()
            assert np.sum(np.abs(np.linalg.eigvalsh(M)) < 1e-9 * np.abs(M).max()) == 1
        w, V = np.linalg.eigh(M_S)
        overlaps2 = (vv / np.linalg.norm(vv) @ V) ** 2
        assert abs(overlaps2.sum() - 1) < 1e-12
        assert overlaps2.min() > 1e-8          # no gauge-phobic state in general


def test_soft_exact_limit_has_a_gauge_phobic_state(s3_soft):
    """At φ = π/3 with zero free soft terms one CP-even state decouples from VV."""
    import numpy as np

    at = s3_soft[-1]
    rng = np.random.default_rng(12)
    M_S, _, _, mD1, vv = at(rng.uniform(-2, 2, 8), 0.8, np.pi / 3, [0.0, 0.0, 0.0])
    assert abs(mD1) < 1e-8 * 246**2
    w, V = np.linalg.eigh(M_S)
    overlaps2 = (vv / np.linalg.norm(vv) @ V) ** 2
    assert overlaps2.min() < 1e-20


def test_soft_s3_image_is_the_same_physics(s3_soft):
    """φ ∈ (0, π/3) is a fundamental domain for the soft-broken scan.

    The S₃ element ab — the reflection P about the 60° axis — maps a vacuum at
    φ to one at 2π/3 − φ.  Both soft doublets (the 2 of 2⊗2 and H_S†H_i)
    transform with the same P, and the transformed point has *identical*
    spectra in all three sectors, with the tadpole-solved mD1² coming out as
    the first component of P·(mD1², mD2²).  So scanning φ past π/3 only
    revisits physics already covered.
    """
    import numpy as np

    at = s3_soft[-1]
    P = np.array([[-0.5, np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    rng = np.random.default_rng(13)
    for _ in range(3):
        lam = rng.uniform(-2, 2, 8)
        theta, phi = rng.uniform(0.2, 1.3), rng.uniform(0.05, 1.0)
        mD2, mS1, mS2 = rng.uniform(-2e5, 2e5, 3)
        M1 = at(lam, theta, phi, [mD2, mS1, mS2])
        mD = P @ [M1[3], mD2]
        mS = P @ [mS1, mS2]
        M2 = at(lam, theta, 2 * np.pi / 3 - phi, [mD[1], mS[0], mS[1]])
        assert abs(M2[3] - mD[0]) < 1e-8 * max(1.0, abs(mD[0]))
        for a, b in zip(M1[:3], M2[:3]):
            wa, wb = np.linalg.eigvalsh(a), np.linalg.eigvalsh(b)
            assert np.allclose(wa, wb, rtol=1e-10, atol=1e-6)


# ---------------------------------------------------------------------------
# The S₃ FERMION sector (leptons and quarks).
#
# Physics input: the S₃ irrep assignment of research/thdm_s3/paper_lfvhd/LFVHD_3HDMS3.tex —
# (F1, F2) a doublet and F_S a singlet, for every left- and right-handed
# species.  Rebuilt here from scratch, deliberately NOT importing
# research/thdm_s3/fermions.py (see the note above the s3_sectors fixture).
#
# The headline result these pin: O12 depends on the VACUUM alone, so it is the
# same matrix in every fermion sector — which forces the CKM matrix to be block
# diagonal unless the vacuum is moved off the S₃ alignment.
# ---------------------------------------------------------------------------

MU1, MU2, MU3, MU4, MU5 = sp.symbols("mu1 mu2 mu3 mu4 mu5", real=True)


def _s3_mass_matrix(r):
    """The S₃ mass matrix at vacuum ratio r = v1/v2 (r = √3 is the alignment)."""
    return sp.Matrix([[MU1 + MU2, r * MU2, r * MU5],
                      [r * MU2, MU1 - MU2, MU5],
                      [r * MU4, MU4, MU3]])


def _o12(r):
    """The 1–2 block-diagonalizing rotation: tan 2ψ = −r, independent of the μ's."""
    psi = (sp.pi - sp.atan(r)) / 2
    c, s = sp.cos(psi), sp.sin(psi)
    return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def test_s3_lepton_yukawa_is_complete(s3_model):
    """feynlag's own enumeration finds exactly the five Yukawa structures.

    The draft writes Y₁…Y₅; character theory says the trivial rep appears once
    in 2⊗2⊗2, giving 5 invariants over the eight (L̄, H, e_R) irrep
    assignments.  `suggest_yukawa` knows nothing about either argument.
    """
    from feynlag import WeylFermion, suggest_yukawa

    model, s3, (H1, H2, HS), _, _, _ = s3_model
    SU2L, U1Y = model.gauge_groups

    def lepL(name):
        return WeylFermion(name, reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                           chirality="L", nflavors=1,
                           component_names=[f"nu{name}", f"e{name}"])

    def lepR(name):
        return WeylFermion(name, reps={U1Y: -1}, chirality="R", nflavors=1,
                           component_names=[name])

    left = [lepL(f"tL{t}") for t in ("1", "2", "S")]
    right = [lepR(f"tR{t}") for t in ("1", "2", "S")]
    s3.assign("2", left[0], left[1])
    s3.assign("1", left[2])
    s3.assign("2", right[0], right[1])
    s3.assign("1", right[2])

    terms = suggest_yukawa(left + right, [H1, H2, HS], [SU2L, U1Y],
                           discrete_groups=[s3], max_dim=4, verify=True)
    assert len(terms) == 5


def test_o12_block_diagonalizes_and_gives_m_e():
    """O₁₂ᵀ M O₁₂ is block diagonal with m_e = μ₁ − 2μ₂ at the alignment."""
    M = _s3_mass_matrix(sp.sqrt(3))
    D = (_o12(sp.sqrt(3)).T * M * _o12(sp.sqrt(3))).applyfunc(sp.simplify)

    assert sp.simplify(D[0, 0] - (MU1 - 2 * MU2)) == 0
    for i, j in ((0, 1), (1, 0), (0, 2), (2, 0)):
        assert sp.simplify(D[i, j]) == 0
    # the residual block is [[μ₁+2μ₂, 2μ₅], [2μ₄, μ₃]]
    assert sp.simplify(D[1, 1] - (MU1 + 2 * MU2)) == 0
    assert sp.simplify(D[1, 2] - 2 * MU5) == 0
    assert sp.simplify(D[2, 1] - 2 * MU4) == 0
    assert sp.simplify(D[2, 2] - MU3) == 0


def test_first_generation_decouples_only_on_the_s3_alignment():
    """The 1st generation decouples **iff** v1 = √3 v2 — the Z₂-preserving vacuum.

    This is what makes the exact-S₃ CKM matrix block diagonal: O₁₂ depends on
    the vacuum alone, so every sector shares it, but the (1,3) entry only
    vanishes at r = √3.  Asserting that it does NOT vanish elsewhere is what
    gives the test teeth.
    """
    r = sp.Symbol("r", positive=True)
    D = (_o12(r).T * _s3_mass_matrix(r) * _o12(r)).applyfunc(sp.simplify)

    # the 1-2 block is diagonalized for ANY r (the angle has no μ dependence)
    assert sp.simplify(D[0, 1]) == 0 and sp.simplify(D[1, 0]) == 0

    assert sp.simplify(D[0, 2].subs(r, sp.sqrt(3))) == 0
    assert sp.simplify(D[2, 0].subs(r, sp.sqrt(3))) == 0
    for bad in (sp.Integer(2), sp.Rational(3, 2), sp.sqrt(5)):
        assert sp.simplify(D[0, 2].subs(r, bad)) != 0

    # the orthogonality condition behind it has the single positive root √3
    lam = sp.sqrt(1 + r**2)
    cond = sp.expand((sp.Matrix([r, -lam - 1]).T * sp.Matrix([r, 1]))[0])
    assert sp.solve(sp.Eq(cond, 0), r) == [sp.sqrt(3)]


def test_exact_s3_ckm_is_a_pure_23_rotation():
    """V_CKM = O_uᵀO_d is a 2–3 rotation, so V_us = V_ub = V_cd = V_td = 0."""
    import numpy as np

    def diagonalize(pars, r_val):
        a, b, c, d = pars                       # (μ₁, μ₂, μ₃, μ₄), with μ₅ = μ₄
        Mn = np.array([[a + b, r_val * b, r_val * d],
                       [r_val * b, a - b, d],
                       [r_val * d, d, c]], dtype=float)
        w, O = np.linalg.eigh(Mn)
        order = np.argsort(np.abs(w))
        return O[:, order]

    pu, pd = (60.0, 25.0, 90.0, 30.0), (2.0, 0.7, 3.0, 1.1)
    root3 = float(sp.sqrt(3))
    V = diagonalize(pu, root3).T @ diagonalize(pd, root3)
    for i, j in ((0, 1), (0, 2), (1, 0), (2, 0)):
        assert abs(V[i, j]) < 1e-12

    # off the alignment the block structure is destroyed
    V_soft = diagonalize(pu, root3 + 0.1).T @ diagonalize(pd, root3 + 0.1)
    assert abs(V_soft[0, 1]) > 1e-3


def test_singular_value_relation_needs_mu4_plus_mu5_squared():
    """(m_τ−m_μ)² = (ρ−μ₃)² + 4(μ₄+μ₅)², not + 16μ₄μ₅.

    The compact form is what the draft prints; it is only correct once
    μ₄ = μ₅, which is *derived* from this relation — so using it there is
    circular.  Both statements are pinned.
    """
    A = sp.Matrix([[MU1 + 2 * MU2, 2 * MU5], [2 * MU4, MU3]])
    rho = MU1 + 2 * MU2
    exact = sp.expand(sp.trace(A.T * A) - 2 * sp.det(A))    # (σ₁ − σ₂)²

    correct = (rho - MU3)**2 + 4 * (MU4 + MU5)**2
    printed = (rho - MU3)**2 + 16 * MU4 * MU5
    assert sp.expand(exact - correct) == 0
    assert sp.expand(exact - printed) != 0
    assert sp.simplify((correct - printed).subs(MU5, MU4)) == 0

    # and the non-circular route still gives μ₅ = μ₄
    p1, p2 = sp.symbols("p1 p2", positive=True)
    combined = sp.expand((MU4 + MU5)**2 - p1 * p2
                         - 2 * (MU4**2 + MU5**2 - p1 * p2 / 2))
    assert sp.factor(combined) == -(MU4 - MU5)**2


# --------------------------------------------------------------------------
# the LFV couplings of research/thdm_s3 notebook 03 §7: which CP-even state
# couples to which lepton flavours at the exact-S₃ vacuum
# --------------------------------------------------------------------------

_ME, _MMU, _MTAU = 0.51099895e-3, 0.1056584, 1.77686


def _draft_lepton_couplings(mu3, v2d=89.0, v3=170.0, split_with_root3=False):
    """(Q_vac, Q_perp, Q_S) in the draft basis at v₁ = √3 v₂, μ₅ = μ₄.

    ``Q_vac``/``Q_perp`` are the couplings of the real neutral field along /
    orthogonal to the doublet vacuum direction, ``Q_S`` of the singlet's —
    i.e. ``Σ_j n_j OᵀG_jO`` for the three orthonormal directions.  With
    ``split_with_root3`` the mass matrix is written with the √3 typed in, every
    doublet entry ∝ v₁ — the split notebook 03 first used, which gives G₂ = 0.
    """
    import math
    import numpy as np

    mu1 = (_ME + _MMU + _MTAU - mu3) / 2
    mu2 = (_MMU + _MTAU - _ME - mu3) / 4
    p1, p2 = mu3 - _MMU, _MTAU - mu3
    mu4 = 0.5 * math.sqrt(p1 * p2)
    tl = math.atan2(math.sqrt(p2), math.sqrt(p1))
    v1d = math.sqrt(3) * v2d
    if split_with_root3:
        s3 = math.sqrt(3)
        G1 = np.array([[mu2, s3 * mu2, s3 * mu4], [s3 * mu2, -mu2, mu4],
                       [s3 * mu4, mu4, 0]]) / v1d
        G2 = np.zeros((3, 3))
    else:
        g2, g4 = mu2 / v2d, mu4 / v2d
        G1 = np.array([[0, g2, g4], [g2, 0, 0], [g4, 0, 0]])
        G2 = np.array([[g2, 0, 0], [0, -g2, g4], [0, g4, 0]])
    GS = np.diag([mu1 / v3, mu1 / v3, mu3 / v3])
    M = v1d * G1 + v2d * G2 + v3 * GS
    c, s = 0.5, math.sqrt(3) / 2
    O12 = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    O23 = np.array([[1, 0, 0], [0, math.cos(tl), math.sin(tl)],
                    [0, -math.sin(tl), math.cos(tl)]])
    O = O12 @ O23
    assert np.allclose(np.abs(np.diag(O.T @ M @ O)), (_ME, _MMU, _MTAU), rtol=1e-9)
    Gt = [O.T @ G @ O for G in (G1, G2, GS)]
    u = np.array([v1d, v2d]) / math.hypot(v1d, v2d)
    q_vac = u[0] * Gt[0] + u[1] * Gt[1]
    q_perp = -u[1] * Gt[0] + u[0] * Gt[1]
    return q_vac, q_perp, Gt[2]


@pytest.mark.parametrize("mu3", [0.5, 0.9, 1.4])
def test_exact_s3_lfv_channels_split_by_state(mu3):
    """At the S₃ vacuum the h₀ direction couples ONLY through the electron.

    Along the vacuum (and for the singlet) the electron is O₁₂-decoupled, so
    those couplings have no e entry: τμ only.  Orthogonal to the vacuum — the
    gauge-phobic h₀ — every non-zero entry has an electron leg: eμ and eτ only,
    and it equals the draft's printed Q₂(A).  The √3-typed-in split
    (G₂ = 0) gets h₀ wrong while leaving the other two untouched.
    """
    import math
    import numpy as np

    v2d = 89.0
    q_vac, q_perp, q_S = _draft_lepton_couplings(mu3, v2d=v2d)
    for q in (q_vac, q_S):
        assert np.abs(q[0, 1:]).max() < 1e-16 and np.abs(q[1:, 0]).max() < 1e-16
    assert abs(q_vac[1, 2]) > 1e-6                       # τμ is there
    assert np.abs(q_perp[1:, 1:]).max() < 1e-16          # no τμ, μμ, ττ for h₀
    assert abs(q_perp[0, 0]) < 1e-16
    assert abs(q_perp[0, 1]) > 1e-4 and abs(q_perp[0, 2]) > 1e-4

    # the draft's printed Q₂(A), [LFVHD] Scenario A
    p1, p2 = mu3 - _MMU, _MTAU - mu3
    q_emu = math.sqrt(p1) * (_ME - mu3 + p1 - 3 * p2) / (4 * v2d * math.sqrt(p1 + p2))
    q_etau = math.sqrt(p2) * (_ME - mu3 + 3 * p1 - p2) / (4 * v2d * math.sqrt(p1 + p2))
    assert np.isclose(abs(q_perp[0, 1]), abs(q_emu), rtol=1e-10)
    assert np.isclose(abs(q_perp[0, 2]), abs(q_etau), rtol=1e-10)

    # the old split: h₀ wrong, the vacuum direction right
    o_vac, o_perp, _ = _draft_lepton_couplings(mu3, v2d=v2d, split_with_root3=True)
    assert np.allclose(np.abs(o_vac), np.abs(q_vac), atol=1e-15)
    assert np.abs(np.abs(o_perp) - np.abs(q_perp)).max() > 1e-4


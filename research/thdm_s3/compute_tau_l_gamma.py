"""The two per-point τ→ℓγ loops of notebook 08, run offline and cached.

`08_tau_to_l_gamma.ipynb` derives and validates the τ rates (against notebook
07's μ→eγ and [HKZ13]'s bounds).  What it no longer runs are its two loops,
here verbatim:

1. **the exact-S₃ vacuum** (notebook 08 §3): every viable exact point of
   ``cache/viable_points.json`` at φ = π/3 with every free soft term zero, on the
   draft lepton branch, over 60 values of μ₃ ∈ (m_μ, m_τ);
2. **the soft scan** (notebook 08 §5): every soft point × every μ₃ on notebook
   06's 73-value grid × every lepton branch, all three radiative channels.

The raw outputs go to ``cache/tau_l_gamma_scan.pkl`` for the notebook to load
into the same variables (``ex_rows``, ``worst_e``, ... and ``scan``).  The
assertions on them stay in the notebook.

Usage::

    python compute_tau_l_gamma.py       # ~17 min

References
----------
[HKZ13] R. Harnik, J. Kopp, J. Zupan, JHEP 03 (2013) 026, arXiv:1209.1397.
"""

from __future__ import annotations

import math
import time

import numpy as np

import decays as D
import lfv
import radiative as RD
import soft_decays as S
import store
from compute_mu_e_gamma import w_hkz
from sm_inputs import ALPHA, G_F, GAMMA_H_SM

NAME = "tau_l_gamma_scan"
ME, MMU, MTAU = lfv.LEPTON_MASSES
E, MU, TAU = RD.E, RD.MU, RD.TAU
CH = S.RADIATIVE_CHANNELS                              # name: (i, j, limit)
PAIRS = {"taumu": (2, 1), "taue": (2, 0), "emu": (0, 1)}
SOFT0 = {n: 0.0 for n in ("mD1sq", "mD2sq", "mS1sq", "mS2sq")}
#: notebook 06's μ₃ grid
GRID = np.linspace(-0.9995 * MTAU, 0.9995 * MTAU, 73)
#: notebook 08 §3: the draft's whole μ₃ range
GRID_EXACT = np.linspace(MMU, MTAU, 62)[1:-1]


def z2_pattern(Yp, tol):
    "(odd states, even states) by their couplings: odd have no tau-mu, even have no e-anything."
    odd = [k for k in ("h1", "h2", "h3", "A1", "A2", "Hpm1", "Hpm2")
           if abs(Yp[k][TAU, MU]) + abs(Yp[k][MU, TAU]) < tol]
    even = [k for k in ("h1", "h2", "h3", "A1", "A2", "Hpm1", "Hpm2")
            if max(abs(Yp[k][E, MU]), abs(Yp[k][MU, E]), abs(Yp[k][E, TAU]), abs(Yp[k][TAU, E])) < tol]
    return odd, even


def exact_vacuum(T):
    """Notebook 08 §3's loop over the exact-S₃ points."""
    exact_pts = D.load_points()
    grid_exact = GRID_EXACT
    t0 = time.time()
    ex_rows, worst_e, worst_mue, worst_pattern, scale_e = [], 0.0, 0.0, 0, 0.0
    for n, pe in enumerate(exact_pts):
        lp = {"lambdas": pe["lambdas"], "theta": pe["theta"], "phi": math.pi / 3, "softs_GeV2": SOFT0}
        rl = S.point_rotations(T, lp, check=False)
        gl = S.hhh_charged(T, lp, rl)
        k_sm = ("h1", "h2", "h3")[S.sm_like_index(rl)]
        for mu3 in grid_exact:
            Y = lfv.aligned_yukawas(T.g_of, mu3, rl["vevs"][0], rl["vevs"][2])
            yuk = S.mass_basis_yukawas(T, Y, rl)
            res = {name: S.lfv_radiative(T, lp, rl, Y, i, j, gl, yukawas=yuk) for name, (i, j, _) in CH.items()}
            worst_e = max(worst_e, res["tau_e"]["br"])
            worst_mue = max(worst_mue, res["mu_e"]["br"])
            Yp = yuk[1]
            scale_e = max(scale_e, max(abs(Yp[k][TAU, E]) for k in Yp))
            odd, even = z2_pattern(Yp, 1e-12)
            # 3 odd (h0, one A, one H+), 4 even, and every state in exactly one class
            worst_pattern += not (len(odd) == 3 and len(even) == 4 and not set(odd) & set(even))
            Q = np.abs(Yp[k_sm])
            h_tm = rl["mass"][k_sm] / (8 * math.pi) * (Q[2, 1] ** 2 + Q[1, 2] ** 2) / GAMMA_H_SM
            light = min(rl["mass"][k] for k in even if k != k_sm and not k.startswith("Hpm"))
            ex_rows.append((n, mu3, res["tau_mu"]["br"], h_tm, light,
                            max(res["tau_mu"]["parts"], key=lambda q: abs(res["tau_mu"]["parts"][q][0]) ** 2
                                + abs(res["tau_mu"]["parts"][q][1]) ** 2)))
    print("%d exact points x %d mu_3 = %d evaluations in %.0f s"
          % (len(exact_pts), len(grid_exact), len(ex_rows), time.time() - t0), flush=True)
    return {"grid_exact": grid_exact, "n_exact": len(exact_pts), "ex_rows": ex_rows,
            "worst_e": worst_e, "worst_mue": worst_mue, "worst_pattern": worst_pattern,
            "scale_e": scale_e}


def soft_scan(T):
    """Notebook 08 §5's loop over the soft points."""
    points = store.points("viable_points_soft")
    grid = GRID
    rots = [S.point_rotations(T, p) for p in points]
    W_HKZ = w_hkz()

    def variants(res, lnn):
        AL, AR, Pq, Pw = res["A_L"], res["A_R"], res["parts"]["BZ_quark"], res["parts"]["BZ_W"]
        return {"no_quark": RD.branching_ratio(AL - Pq[0], AR - Pq[1], ALPHA, G_F, lnn),
                "top_flip": RD.branching_ratio(AL - 2 * Pq[0], AR - 2 * Pq[1], ALPHA, G_F, lnn),
                "w_hkz": RD.branching_ratio(AL - (1 - W_HKZ) * Pw[0], AR - (1 - W_HKZ) * Pw[1], ALPHA, G_F, lnn)}

    LNN = {name: (S.TAU_BR_LNN[j] if i == TAU else 1.0) for name, (i, j, _) in CH.items()}

    def lead_of(res):
        return max(res["parts"], key=lambda q: abs(res["parts"][q][0]) ** 2 + abs(res["parts"][q][1]) ** 2)

    def scan_point(i):
        p, r = points[i], rots[i]
        g = S.hhh_charged(T, p, r)
        k_sm = ("h1", "h2", "h3")[S.sm_like_index(r)]
        m_sm = r["mass"][k_sm]
        rows = []
        for c in grid:
            for b in lfv.lepton_branches(T.g_of, c, r["vevs"]):
                yuk = S.mass_basis_yukawas(T, b["Y"], r)
                res = {name: S.lfv_radiative(T, p, r, b["Y"], ii, jj, g, yukawas=yuk) for name, (ii, jj, _) in CH.items()}
                Q = np.abs(yuk[1][k_sm])
                hbr = {q: m_sm / (8 * math.pi) * (Q[a, bb] ** 2 + Q[bb, a] ** 2) / GAMMA_H_SM for q, (a, bb) in PAIRS.items()}
                rows.append({"mu3": float(c), "kind": b["kind"],
                             "br": {name: res[name]["br"] for name in CH},
                             "lead": {name: lead_of(res[name]) for name in CH},
                             "var": {name: variants(res[name], LNN[name]) for name in CH},
                             "h": hbr})
        return rows

    t0 = time.time()
    scan = []
    for i in range(len(points)):
        scan.append(scan_point(i))
        if i % 400 == 0:
            print("  point %4d / %d, %.0f s" % (i, len(points), time.time() - t0), flush=True)
    n_cfg = sum(len(rows) for rows in scan)
    print("scanned %d points: %d (point, mu_3, branch) configurations in %.0f s" % (len(points), n_cfg, time.time() - t0))
    return {"grid": grid, "n_points": len(points), "W_HKZ": W_HKZ, "scan": scan}


def run():
    T = S.build_tools()
    raw = {"exact": exact_vacuum(T), "soft": soft_scan(T)}
    store.save_raw(NAME, raw)
    return raw


if __name__ == "__main__":
    run()

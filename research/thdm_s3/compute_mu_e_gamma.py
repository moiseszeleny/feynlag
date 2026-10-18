"""The per-configuration μ→eγ scan of notebook 07, run offline and cached.

`07_mu_to_e_gamma.ipynb` derives and validates every piece of the rate (the
one-loop and Barr–Zee amplitudes against [HKZ13], the Goldstone couplings, the
exact-S₃ zero).  What it no longer does is the scan: every soft point × every μ₃
on notebook 06's 73-value grid × every lepton branch, ~4×10⁵ configurations.
That loop is here, verbatim (`scan_point` is notebook 07 §5's), and its raw
output goes to ``cache/mu_e_gamma_scan.pkl`` for the notebook to load into the
same ``scan`` variable.

``W_HKZ``, the ratio of [HKZ13]'s W Barr–Zee number to ours, is recomputed here
exactly as notebook 07 §2.3 computes it, and the notebook asserts the two agree.

Usage::

    python compute_mu_e_gamma.py        # ~11 min

References
----------
[HKZ13] R. Harnik, J. Kopp, J. Zupan, JHEP 03 (2013) 026, arXiv:1209.1397.
"""

from __future__ import annotations

import math
import time

import numpy as np

import lfv
import radiative as RD
import soft_decays as S
import store
from sm_inputs import ALPHA, G_F, M_W, GAMMA_H_SM

NAME = "mu_e_gamma_scan"
ME, MMU, MTAU = lfv.LEPTON_MASSES
PAIRS = {"taumu": (2, 1), "taue": (2, 0), "emu": (0, 1)}
#: notebook 06's μ₃ grid
GRID = np.linspace(-0.9995 * MTAU, 0.9995 * MTAU, 73)


def w_hkz():
    """[HKZ13]'s Eq. (14) W number over ours (notebook 07 §2.3)."""
    mt_bar, mh = 164.0, 125.0
    bz = RD.barr_zee(1.0, 0.0, mh, ALPHA, 246.0, m_w=M_W, kappa_w=1.0,
                     fermions=[(3, 2 / 3, mt_bar, 1.0)], m_mu=MTAU)
    c_w = (4 * math.pi**2 * bz["W"][0] * mh**2).real
    return 0.11 / c_w


def run():
    T = S.build_tools()
    points = store.points("viable_points_soft")
    grid = GRID
    rots = [S.point_rotations(T, p) for p in points]
    W_HKZ = w_hkz()

    def scan_point(i):
        p, r = points[i], rots[i]
        g = S.hhh_charged(T, p, r)
        k_sm = S.sm_like_index(r)
        m_sm = r["mass"][("h1", "h2", "h3")[k_sm]]
        rows = []
        for c in grid:
            for b in lfv.lepton_branches(T.g_of, c, r["vevs"]):
                res = S.mu_e_gamma(T, p, r, b["Y"], g)
                _, Yp = S.mass_basis_yukawas(T, b["Y"], r)
                Q = np.abs(Yp[("h1", "h2", "h3")[k_sm]])
                hbr = {q: m_sm / (8 * math.pi) * (Q[a, bb] ** 2 + Q[bb, a] ** 2) / GAMMA_H_SM for q, (a, bb) in PAIRS.items()}
                parts = {k_: abs(v_[0]) ** 2 + abs(v_[1]) ** 2 for k_, v_ in res["parts"].items()}
                # systematics from the same amplitudes: drop the quark-loop ansatz, flip its sign,
                # or scale the W loop down to [HKZ13]'s Eq. (14) number (the 14% gap of section 2.3)
                AL, AR, Pq, Pw = res["A_L"], res["A_R"], res["parts"]["BZ_quark"], res["parts"]["BZ_W"]
                var = {"no_quark": RD.branching_ratio(AL - Pq[0], AR - Pq[1], ALPHA, G_F),
                       "top_flip": RD.branching_ratio(AL - 2 * Pq[0], AR - 2 * Pq[1], ALPHA, G_F),
                       "w_hkz": RD.branching_ratio(AL - (1 - W_HKZ) * Pw[0], AR - (1 - W_HKZ) * Pw[1], ALPHA, G_F)}
                rows.append({"mu3": float(c), "kind": b["kind"], "br": res["br"], "h": hbr, "parts": parts, "var": var})
        return rows

    t0 = time.time()
    scan = []
    for i in range(len(points)):
        scan.append(scan_point(i))
        if i % 400 == 0:
            print("  point %4d / %d, %.0f s" % (i, len(points), time.time() - t0), flush=True)
    n_cfg = sum(len(rows) for rows in scan)
    print("scanned %d points: %d (point, mu_3, branch) configurations in %.0f s" % (len(points), n_cfg, time.time() - t0))
    store.save_raw(NAME, {"grid": grid, "n_points": len(points), "W_HKZ": W_HKZ, "scan": scan})
    return scan


if __name__ == "__main__":
    run()

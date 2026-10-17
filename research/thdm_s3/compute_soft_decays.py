"""The per-point LFV scan of notebook 06, run offline and cached.

`06_soft_decays.ipynb` derives and validates everything this script uses: the
point rebuild, the fast couplings against feynlag's extractor, the lepton fit at
any vacuum, the vacuum-direction identity.  What it no longer does is the loop
over every soft point: `soft_decays.scan_point` on each of the ~2000 points of
``cache/viable_points_soft.json`` over the notebook's 73-value μ₃ grid.  That
loop is here, verbatim, and its raw output goes to ``cache/soft_decays_scan.pkl``
for the notebook to load into the same ``results`` variable.

Usage::

    python compute_soft_decays.py       # ~4 min
"""

from __future__ import annotations

import time

import numpy as np

import lfv
import soft_decays as S
import store

NAME = "soft_decays_scan"
ME, MMU, MTAU = lfv.LEPTON_MASSES
#: notebook 06 §4: the μ₃ dial, wherever a branch exists
GRID = np.linspace(-0.9995 * MTAU, 0.9995 * MTAU, 73)


def run():
    t0 = time.time()
    T = S.build_tools()
    points = store.points("viable_points_soft")
    grid = GRID
    results = [S.scan_point(T, p, grid) for p in points]
    print("scanned %d points x %d mu_3 values in %.0f s: %d (point, mu_3, branch) entries"
          % (len(points), len(grid), time.time() - t0, sum(len(r["lfv"]) for r in results)))
    store.save_raw(NAME, {"grid": grid, "n_points": len(points), "results": results})
    return results


if __name__ == "__main__":
    run()

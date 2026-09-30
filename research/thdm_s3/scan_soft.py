"""The 3HDM-S₃ scalar parameter scan in the **soft-broken** vacuum.

`scan.py` scans the exactly-S₃ potential, whose tadpoles force the √3 alignment
(φ = π/3) and leave θ as the vacuum's only freedom.  The four soft quadratics
(`model.build_model(soft=True)`) release that: the tadpoles become ordinary
equations for (μ0², μ1², mD1²), and a general vacuum (θ, φ) is a stationary point
for any values of the other three soft terms (mD2², mS1², mS2²).  This module is
the same cut flow as `scan.py` over that larger space, and the sole writer of
``results/viable_points_soft.json``.  `05_soft_scalar_scan.ipynb` derives what
changes and runs chunk 0 of this scan itself.

What carries over unchanged, and why:

* **Boundedness and unitarity** are statements about the quartics alone; soft
  terms are dimension 2 and touch none of them.  `strict_bfb_mask`,
  `unitarity_mask` and the 12-d backstop are therefore applied to λ exactly as in
  `scan.py` — and applied *first*, before any spectrum is built, since ~95% of λ
  draws fail them and nothing about the vacuum can rescue those.
* **The neutral cuts** (`sm_like_cut`, `lep_neutral_mask`) already identify the
  SM-like state by its coupling, never by a label, so they apply as they are to a
  CP-even sector in which no state is gauge-phobic any more.

What is new is the spectrum (`model.soft_spectrum_function`): the geometric
rotation still isolates the Goldstones, but the CP-even 3×3 is fully mixed and the
CP-odd and charged 2×2 blocks are no longer diagonal.

**Prior** (a choice, recorded in the JSON, not a physics statement):

* λᵢ ~ U(−2, 2) and θ ~ U(0.05, π/2 − 0.05), as in `scan.py`;
* φ ~ U(0, π/3).  This is a **fundamental domain**: the S₃ reflection about the
  60° axis maps a vacuum at φ to one at 2π/3 − φ, together with an S₃ rotation of
  the soft terms, and the two are the same physics (pinned in
  `tests/test_thdm_s3.py`).  Both ends are residual-Z₂ directions — φ = π/3 is
  notebook 01's alignment — so the exact-S₃ limit sits at the domain's edge;
* the three free soft quadratics ~ U(−M², M²) with M = 500 GeV, signed.  *Soft*
  means dimension < 4, not *small*: soft terms are renormalized only in
  proportion to themselves, so any size is consistent.  The near-S₃ regime is read
  off afterwards from the per-point ``breaking`` tags rather than imposed here.

The draws are ordered λ, θ, φ, softs from ``default_rng([SOFT_SEED, k])``; there is
no notebook-01 sample to reproduce, so every chunk uses the list-seed form.

The vacuum requirement is **no tachyons**, i.e. a local minimum.  With soft terms
the potential can have several minima, and whether this one is the global one is
not checked (NOTES.md, open questions).

Usage::

    python scan_soft.py --chunks 1          # one 2M sample (notebook 05's chunk 0)
    python scan_soft.py                     # the full scan

References
----------
See `constraints.py` for [DasDey14], [GomezBock21] and [LEP2003].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from constraints import (KAPPA2_MIN, M_LEP_ZH, XI2_LEP_FLOOR, lep_neutral_mask,
                         sm_like_cut, strict_bfb_mask, unitarity_mask)
from model import (SOFT_CP_EVEN_KEYS, SOFT_SPECTRUM_KEYS, V_EW, build_model,
                   draft_r, numeric_bfb_min, soft_free_names, soft_solved_names,
                   soft_spectrum_function, soft_vevs)
from scan import BACKSTOP_DIRECTIONS, CHUNK_SIZE, LAMBDA_RANGE

#: Seed of the soft scan (the date it was opened).  Chunk k draws from
#: ``default_rng([SOFT_SEED, k])``.
SOFT_SEED = 20260929

#: Free soft quadratics are drawn from U(−SOFT_MASS², +SOFT_MASS²), GeV².
SOFT_MASS = 500.0

#: φ is drawn from (0, PHI_MAX): the fundamental domain under S₃.
PHI_MAX = np.pi / 3

#: Largest tolerated Goldstone residual (see `soft_spectrum_function`).  A
#: larger one means the tadpole solution and the mass matrices disagree about
#: where the vacuum is, which is a bug, not a physics outcome — so it raises.
GOLDSTONE_TOL = 1e-9

CUT_LABELS = ("sampled", "+ strict BFB", "+ unitarity", "+ no tachyons",
              "+ charged > 80", "+ SM-like at 125", "+ LEP neutral veto",
              "+ 12-d backstop")


def chunk_rng(chunk: int) -> np.random.Generator:
    return np.random.default_rng([SOFT_SEED, chunk])


def draw(chunk: int, n: int = CHUNK_SIZE):
    """``(lam, theta, phi, softs)`` for one chunk, in that draw order.

    ``softs`` is (n, 3) over `model.soft_free_names` — (mD2², mS1², mS2²).
    """
    rng = chunk_rng(chunk)
    lam = rng.uniform(*LAMBDA_RANGE, size=(n, 8))
    theta = rng.uniform(0.05, np.pi / 2 - 0.05, size=n)
    phi = rng.uniform(0.0, PHI_MAX, size=n)
    softs = rng.uniform(-SOFT_MASS**2, SOFT_MASS**2, size=(n, 3))
    return lam, theta, phi, softs


def scan_chunk(ms, m_exact, chunk: int, n: int = CHUNK_SIZE, backstop: bool = True):
    """Run every cut over one chunk.

    ``ms`` is ``build_model(soft=True, soft_solve_for="mD1sq")``; ``m_exact`` the
    plain model, used only for the 12-d backstop (whose quartic potential is the
    same in both).  Returns ``(records, counts)`` as `scan.scan_chunk` does.
    """
    lam, theta, phi, softs = draw(chunk, n)
    spectrum = soft_spectrum_function(ms)

    keep = strict_bfb_mask(lam)
    counts = {"sampled": n, "+ strict BFB": int(keep.sum())}
    keep &= unitarity_mask(lam)
    counts["+ unitarity"] = int(keep.sum())

    # the spectrum only where the quartics already pass: ~5% of the sample
    idx = np.flatnonzero(keep)
    v1, v2, vS = soft_vevs(theta[idx], phi[idx])
    out = spectrum([lam[idx, k] for k in range(8)], v1, v2, vS,
                   [softs[idx, k] for k in range(3)])
    worst = float(out["goldstone"].max(initial=0.0))
    if worst > GOLDSTONE_TOL:
        raise RuntimeError(f"Goldstone residual {worst:.2e} > {GOLDSTONE_TOL:.0e}: "
                           f"the tadpole solution does not put the vacuum where "
                           f"the mass matrices think it is")

    sel = np.ones(len(idx), dtype=bool)
    for k in SOFT_SPECTRUM_KEYS:
        sel &= out["mass2"][k] > 0
    counts["+ no tachyons"] = int(sel.sum())

    mass = {k: np.sqrt(np.clip(v, 0.0, None)) for k, v in out["mass2"].items()}
    sel &= (mass["Hpm1"] > 80) & (mass["Hpm2"] > 80)
    counts["+ charged > 80"] = int(sel.sum())

    xi = out["hvv"]
    m_even = {k: mass[k] for k in SOFT_CP_EVEN_KEYS}
    sel &= sm_like_cut(xi, m_even)
    counts["+ SM-like at 125"] = int(sel.sum())
    sel &= lep_neutral_mask(xi, m_even)
    counts["+ LEP neutral veto"] = int(sel.sum())

    loc = np.flatnonzero(sel)
    if backstop and len(loc):
        survive = numeric_bfb_min(m_exact, lam[idx[loc]],
                                  n_directions=BACKSTOP_DIRECTIONS,
                                  seed=9, subspace="all") >= -1e-9
        loc = loc[survive]
    counts["+ 12-d backstop"] = int(len(loc))

    free, solved = soft_free_names(ms), soft_solved_names(ms)
    records = []
    for j in loc:
        i = idx[j]
        soft_vals = {nm: float(softs[i, k]) for k, nm in enumerate(free)}
        soft_vals.update({nm: float(out["solved"][nm][j]) for nm in solved
                          if nm in ms.softs})
        r = float(draft_r(phi[i]))
        records.append({
            "chunk": chunk,
            "lambdas": {f"lambda_{k + 1}": round(float(lam[i, k]), 5) for k in range(8)},
            "theta": round(float(theta[i]), 5),
            "phi": round(float(phi[i]), 5),
            "r_draft": round(r, 5),
            "r_draft_image": round(float(draft_r(2 * PHI_MAX - phi[i])), 5),
            "vevs_GeV": {"v1": round(float(v1[j]), 3), "v2": round(float(v2[j]), 3),
                         "vS": round(float(vS[j]), 3)},
            "softs_GeV2": {nm: round(v, 2) for nm, v in sorted(soft_vals.items())},
            "mu_sq_GeV2": {nm: round(float(out["solved"][nm][j]), 2)
                           for nm in solved if nm not in ms.softs},
            "masses_GeV": {k: round(float(mass[k][j]), 2) for k in SOFT_SPECTRUM_KEYS},
            "hvv_squared": {k: round(float(xi[k][j]), 4) for k in SOFT_CP_EVEN_KEYS},
            "cp_even_mixing": np.round(out["mixing"][j], 5).tolist(),
            "breaking": {
                "max_soft_over_v2": round(max(abs(v) for v in soft_vals.values())
                                          / V_EW**2, 4),
                "pi3_minus_phi": round(float(PHI_MAX - phi[i]), 5),
            },
        })
    return records, counts


def run(chunks: int, out: Path, n: int = CHUNK_SIZE):
    """Sweep ``chunks`` samples and write the JSON.  Returns the payload."""
    ms = build_model(soft=True, soft_solve_for="mD1sq")
    m_exact = build_model()
    records, per_chunk, totals = [], [], {k: 0 for k in CUT_LABELS}
    t0 = time.time()
    for c in range(chunks):
        t1 = time.time()
        recs, counts = scan_chunk(ms, m_exact, c, n)
        records.extend(recs)
        per_chunk.append(counts)
        for k, v in counts.items():
            totals[k] += v
        print("chunk %2d/%d: %4d points  (%5.1f s, %d total)"
              % (c + 1, chunks, len(recs), time.time() - t1, len(records)),
              flush=True)

    payload = {
        "description": (
            "S3-3HDM scalar parameter points in the SOFT-BROKEN vacuum: the four "
            "CP-conserving soft S3-breaking quadratics release the sqrt(3) "
            "alignment, so the vacuum (theta, phi) is general and the tadpoles fix "
            "(mu0sq, mu1sq, mD1sq). Cuts as in viable_points.json (strict BFB + "
            "unitarity on the quartics, no tachyons = LOCAL minimum only, "
            "m_Hpm > 80 GeV, the hVV-carrying CP-even state at 125 +- 3 GeV with "
            "kappa^2 >= KAPPA2_MIN, the LEP neutral veto, the 12-d BFB backstop). "
            "CP-even states h1 < h2 < h3 are mass-ordered; none is gauge-phobic in "
            "general, and hvv_squared sums to 1. cp_even_mixing[i][k] is state h_k's "
            "component along the real part of (H1^0, H2^0, HS^0)[i], sign fixed so "
            "its hVV coupling is >= 0. r_draft = -cot(phi - pi/6) is [LFVHD]'s "
            "v1/v2 (notebook 02's r; the exact-S3 vacuum has |r| = sqrt3); "
            "r_draft_image = -tan(phi) is the same for the S3-image vacuum at "
            "2pi/3 - phi, which is the same physics, so each point stands for "
            "both draft ratios. "
            "breaking.max_soft_over_v2 is the largest |soft m^2|/v^2 over all four "
            "soft terms (mD1sq included)."),
        "electroweak_scale_GeV": V_EW,
        "n_points": len(records),
        "sample_size": chunks * n,
        "chunks": chunks,
        "rng_seed": SOFT_SEED,
        "rng_scheme": "chunk k: default_rng([seed, k]); draws lam, theta, phi, softs",
        "prior": {
            "lambda_range": list(LAMBDA_RANGE),
            "theta_range": [0.05, float(np.pi / 2 - 0.05)],
            "phi_range": [0.0, float(PHI_MAX)],
            "free_softs": soft_free_names(ms),
            "soft_range_GeV2": [-SOFT_MASS**2, SOFT_MASS**2],
            "solved_by_tadpoles": soft_solved_names(ms),
        },
        "thresholds": {"kappa2_min": KAPPA2_MIN, "m_lep_zh_GeV": M_LEP_ZH,
                       "xi2_lep_floor": XI2_LEP_FLOOR,
                       "goldstone_tol": GOLDSTONE_TOL},
        "vacuum_requirement": "local minimum (no tachyons); global stability not checked",
        "cut_flow": {k: totals[k] for k in CUT_LABELS},
        "cut_flow_per_chunk": per_chunk,
        "points": records,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("\nwrote %s: %d points from %d samples in %.1f min"
          % (out, len(records), chunks * n, (time.time() - t0) / 60))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunks", type=int, default=55,
                    help="samples of --size to sweep (default 55, ~2000 survivors)")
    ap.add_argument("--size", type=int, default=CHUNK_SIZE,
                    help="points per chunk (default 2,000,000)")
    ap.add_argument("--out", type=Path, default=Path("results/viable_points_soft.json"))
    args = ap.parse_args()
    run(args.chunks, args.out, args.size)


if __name__ == "__main__":
    main()

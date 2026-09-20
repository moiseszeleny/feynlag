"""The 3HDM-S₃ scalar parameter scan, run offline and cached.

`01_scalar_parameter_space.ipynb` derives every cut used here and runs the same
scan over one 2-million-point sample, which is what its narrative and its
cut-flow figure are built on.  This module is the same scan at scan size: it
sweeps many such samples in sequence and is the **sole writer** of
``results/viable_points.json``.  The split exists so the notebook keeps showing
one sample it can narrate while the cached file is swept over many; nothing here
is derived, it is the notebook's own cuts in a loop.

**Chunk 0 is the notebook's sample, exactly.**  It draws from
``default_rng(20260729)`` in the notebook's order (the eight λ's, then θ), so a
notebook run and ``scan.py``'s first chunk produce the same points and the same
survivors — asserted in the notebook rather than assumed here.  Chunk k ≥ 1 uses
``default_rng([20260729, k])``, an independent stream of the same generator.

Memory is the reason for chunking at all: one 2M sample already allocates ~130 MB
of mass arrays, and the unitarity eigenvalues another ~150 MB per million inside
`constraints`, so the samples are processed and discarded one at a time and only
the survivors are kept.

Cuts, in the order the cut flow reports them (see the notebook for the physics):

1. boundedness from below — [DasDey14] Eq. (4) **plus** the corrected
   neutral-direction condition derived in the notebook (`strict_bfb_mask`);
2. tree unitarity, [DasDey14] Eq. (36)–(37);
3. no tachyons;
4. ``m_H± > 80`` GeV, the LEP charged-scalar bound as [GomezBock21] applies it;
5. the hVV-carrying CP-even state at 125 ± 3 GeV with ξ² ≥ `KAPPA2_MIN`
   (`sm_like_cut`) — a *coupling* cut, not the mass-only condition the notebook's
   §8 applies;
6. no light, appreciably-coupled neutral state (`lep_neutral_mask`, [LEP2003]);
7. the full 12-dimensional boundedness backstop (`numeric_bfb_min`), affordable
   only on the handful of survivors.

Usage::

    python scan.py --chunks 30          # 60M points, ~3.5 min, ~2000 survivors
    python scan.py --chunks 1           # reproduces the notebook's sample alone

About 7 s per chunk: the cuts are vectorized and the only per-point work left is
the 12-d backstop, which by then runs on ~65 points rather than ~320.

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
from model import (SPECTRUM_KEYS, V_EW, build_model, delta_function,
                   hvv_function, numeric_bfb_min, spectrum_function)

#: The notebook's seed.  Chunk 0 must reproduce its sample exactly.
BASE_SEED = 20260729

#: Points per chunk.  The notebook's sample size, so chunk 0 IS that sample.
CHUNK_SIZE = 2_000_000

#: λ's are drawn uniformly from this box, as in the notebook.
LAMBDA_RANGE = (-2.0, 2.0)

#: Directions sampled by the 12-d boundedness backstop.
BACKSTOP_DIRECTIONS = 40_000

#: CP-even states, in the order `hvv_function` reports them.
CP_EVEN_KEYS = ("h0", "H1", "H2")

CUT_LABELS = ("sampled", "+ strict BFB", "+ unitarity", "+ no tachyons",
              "+ charged > 80", "+ SM-like at 125", "+ LEP neutral veto",
              "+ 12-d backstop")


def chunk_rng(chunk: int) -> np.random.Generator:
    """The generator for one chunk — chunk 0 is the notebook's own."""
    return np.random.default_rng(BASE_SEED if chunk == 0
                                 else [BASE_SEED, chunk])


def draw(chunk: int, n: int = CHUNK_SIZE):
    """``(lam, theta, v1, v2, vS)`` for one chunk, in the notebook's draw order.

    The order matters and is not cosmetic: drawing θ before the λ's would give a
    different stream from the same seed, and chunk 0 would stop reproducing the
    notebook.
    """
    rng = chunk_rng(chunk)
    lam = rng.uniform(*LAMBDA_RANGE, size=(n, 8))
    theta = rng.uniform(0.05, np.pi / 2 - 0.05, size=n)
    v12 = V_EW * np.sin(theta)
    return lam, theta, v12 / 2.0, v12 * np.sqrt(3) / 2.0, V_EW * np.cos(theta)


def scan_chunk(m, chunk: int, n: int = CHUNK_SIZE, backstop: bool = True):
    """Run every cut over one chunk.

    Returns ``(records, counts)``: the surviving points as JSON-ready dicts, and
    the cumulative survivor count after each cut in `CUT_LABELS`.
    """
    lam, theta, v1, v2, vS = draw(chunk, n)
    spectrum, hvv = spectrum_function(m, "numpy"), hvv_function(m)
    lam_cols = [lam[:, k] for k in range(8)]

    msq = spectrum(lam_cols, v2, vS)
    keep = strict_bfb_mask(lam)
    counts = {"sampled": n, "+ strict BFB": int(keep.sum())}

    keep &= unitarity_mask(lam)
    counts["+ unitarity"] = int(keep.sum())

    for k in SPECTRUM_KEYS:
        keep &= msq[k] > 0
    counts["+ no tachyons"] = int(keep.sum())

    mass = {k: np.sqrt(np.clip(v, 0.0, None)) for k, v in msq.items()}
    del msq                                   # ~130 MB, fully consumed above

    keep &= (mass["Hpm1"] > 80) & (mass["Hpm2"] > 80)
    counts["+ charged > 80"] = int(keep.sum())

    # the coupling cuts, on the survivors only: hvv_function is cheap, but the
    # masks stack three arrays and there is no reason to do it at full width
    idx = np.flatnonzero(keep)
    xi_s = hvv([lam[idx, k] for k in range(8)], v2[idx], vS[idx])
    m_s = {k: mass[k][idx] for k in CP_EVEN_KEYS}

    sel = sm_like_cut(xi_s, m_s)
    counts["+ SM-like at 125"] = int(sel.sum())
    sel &= lep_neutral_mask(xi_s, m_s)
    counts["+ LEP neutral veto"] = int(sel.sum())
    idx = idx[sel]

    if backstop and len(idx):
        survive = numeric_bfb_min(m, lam[idx], n_directions=BACKSTOP_DIRECTIONS,
                                  seed=9, subspace="all") >= -1e-9
        idx = idx[survive]
    counts["+ 12-d backstop"] = int(len(idx))

    lam_f = [lam[idx, k] for k in range(8)]
    xi_f = hvv(lam_f, v2[idx], vS[idx])
    delta = delta_function(m)(lam_f, v2[idx], vS[idx])
    records = [{
        "chunk": chunk,
        "lambdas": {f"lambda_{k + 1}": round(float(lam[i, k]), 5) for k in range(8)},
        "theta": round(float(theta[i]), 5),
        "vevs_GeV": {"v1": round(float(v1[i]), 3), "v2": round(float(v2[i]), 3),
                     "vS": round(float(vS[i]), 3)},
        "masses_GeV": {k: round(float(mass[k][i]), 2) for k in SPECTRUM_KEYS},
        "hvv_squared": {k: round(float(xi_f[k][j]), 4) for k in CP_EVEN_KEYS},
        "delta": round(float(delta[j]), 6),
    } for j, i in enumerate(idx)]
    return records, counts


def run(chunks: int, out: Path, n: int = CHUNK_SIZE):
    """Sweep ``chunks`` samples and write the JSON.  Returns the payload."""
    m = build_model()
    records, per_chunk, totals = [], [], {k: 0 for k in CUT_LABELS}
    t0 = time.time()
    for c in range(chunks):
        t1 = time.time()
        recs, counts = scan_chunk(m, c, n)
        records.extend(recs)
        per_chunk.append(counts)
        for k, v in counts.items():
            totals[k] += v
        print("chunk %2d/%d: %4d points  (%5.1f s, %d total)"
              % (c + 1, chunks, len(recs), time.time() - t1, len(records)),
              flush=True)

    payload = {
        "description": (
            "S3-3HDM scalar parameter points surviving boundedness-from-below "
            "([DasDey14] Eq.4 PLUS the corrected neutral condition derived in "
            "01_scalar_parameter_space.ipynb section 5, PLUS a full "
            "12-dimensional numerical backstop), tree unitarity [DasDey14] "
            "Eq.36-37, no tachyons, m_Hpm > 80 GeV, and the COUPLING-aware "
            "neutral conditions: the CP-even state carrying the hVV coupling "
            "sits at 125 +- 3 GeV with at least KAPPA2_MIN of the SM strength, "
            "and no CP-even state is both lighter than M_LEP_ZH and coupled "
            "above XI2_LEP_FLOOR ([LEP2003]).  hvv_squared is each CP-even "
            "state's hVV coupling^2 in units of the SM's; h0 is exactly 0 "
            "(gauge-phobic for any delta) and H1 + H2 sum to 1.  delta is the "
            "CP-even Higgs-basis angle, = -psi with psi the residual 2x2 block "
            "angle; it lies in (-pi/4, pi/4), so h_1 is by convention the state "
            "carrying the larger hVV coupling (cos^2 delta >= 1/2)."),
        "electroweak_scale_GeV": V_EW,
        "n_points": len(records),
        "sample_size": chunks * n,
        "chunks": chunks,
        "rng_seed": BASE_SEED,
        "rng_scheme": ("chunk 0: default_rng(seed) -- the notebook's own sample; "
                       "chunk k>0: default_rng([seed, k])"),
        "lambda_range": list(LAMBDA_RANGE),
        "thresholds": {"kappa2_min": KAPPA2_MIN, "m_lep_zh_GeV": M_LEP_ZH,
                       "xi2_lep_floor": XI2_LEP_FLOOR},
        "boundedness": {
            "quoted_conditions": "[DasDey14] Eq.(4a)-(4g), arXiv v2 (pre-erratum)",
            "correction": "min_{t>=0} [ l8 t^4 + (l5+l6+2 l7) t^2 - 2|l4| t + (l1+l3) ] >= 0",
            "numeric_backstop_directions": BACKSTOP_DIRECTIONS,
        },
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
    ap.add_argument("--chunks", type=int, default=30,
                    help="samples of --size to sweep (default 30, 60M points)")
    ap.add_argument("--size", type=int, default=CHUNK_SIZE,
                    help="points per chunk (default 2,000,000 -- the notebook's)")
    ap.add_argument("--out", type=Path, default=Path("results/viable_points.json"))
    args = ap.parse_args()
    run(args.chunks, args.out, args.size)


if __name__ == "__main__":
    main()

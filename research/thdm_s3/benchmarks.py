"""Three benchmark points of the soft-broken scan, tuned to m_h = M_H and followed through.

A histogram over the scan shows what is typical. It does not show what one point
looks like from end to end: its inputs, its spectrum, its couplings, and how it
fares against each limit. This script writes one card per benchmark to
``BENCHMARKS.md`` and the tuned records to ``results/benchmarks.json``.

**1. Pick** three points of ``results/viable_points_soft.json`` by fixed rules,
among the *clean* points: SM-like mass within 0.3 GeV of `sm_inputs.M_H`, and the
SM-like state is the lightest scalar.

- **decoupled**: MEG II-allowed, every non-SM scalar above 700 GeV, largest κ_V;
- **light H±**: MEG II-allowed, lightest charged scalar;
- **near S₃**: smallest soft breaking (largest |soft term| / v²).

**2. Tune** each to m_h = `M_H` exactly. Only one quartic moves: the λ_k with the
largest |∂m_h/∂λ_k|, solved by bisection. θ, φ and the free soft terms are held
fixed, so the tadpoles re-solve μ₀², μ₁² and m²_D1. The tuned point is rebuilt
as `scan_soft.scan_chunk` builds a record, and every cut of the scan is
re-checked on it (an assertion, not a hope).

**3. Recompute** the observables of notebooks 06, 07 and 08 at the tuned point,
replicating their per-point logic with the `soft_decays` functions. Before it
is trusted, the same code is run on the *untuned* scan points and must
reproduce what those notebooks stored for them (the validation gate).

Usage (from anywhere)::

    python research/thdm_s3/benchmarks.py

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RES = HERE / "results"

import decays as D  # noqa: E402
import lfv  # noqa: E402
import radiative as RD  # noqa: E402
import soft_decays as S  # noqa: E402
from constraints import (lep_neutral_mask, sm_like_cut, strict_bfb_mask,  # noqa: E402
                         unitarity_mask)
from model import (SOFT_CP_EVEN_KEYS, SOFT_SPECTRUM_KEYS, V_EW, build_model,  # noqa: E402
                   draft_r, numeric_bfb_min, soft_free_names, soft_solved_names,
                   soft_spectrum_function, soft_vevs)
from scan import BACKSTOP_DIRECTIONS  # noqa: E402
from sm_inputs import ALPHA, G_F, GAMMA_H_SM, M_H, M_W  # noqa: E402

CLEAN_WINDOW = 0.3          # GeV around M_H for a benchmark candidate
MTAU = lfv.LEPTON_MASSES[2]
GRID = np.linspace(-0.9995 * MTAU, 0.9995 * MTAU, 73)    # notebooks 06-08's mu_3 grid
PAIRS = {"taumu": (2, 1), "taue": (2, 0), "emu": (0, 1)}
CH = S.RADIATIVE_CHANNELS


def _w_hkz():
    """Notebook 07 §2.3: [HKZ13]'s W Barr-Zee number over ours (the w_hkz variant)."""
    bz = RD.barr_zee(1.0, 0.0, 125.0, ALPHA, 246.0, m_w=M_W, kappa_w=1.0,
                     fermions=[(3, 2 / 3, 164.0, 1.0)], m_mu=MTAU)
    return 0.11 / (4 * math.pi**2 * bz["W"][0] * 125.0**2).real


W_HKZ = _w_hkz()


def _load(name):
    return json.loads((RES / name).read_text())


# ----------------------------------------------------------------------- pick

def _sm_key(p):
    return max(SOFT_CP_EVEN_KEYS, key=lambda k: p["hvv_squared"][k])


def _non_sm(p):
    sm = _sm_key(p)
    return {k: m for k, m in p["masses_GeV"].items() if k != sm}


def clean(p):
    m_sm = p["masses_GeV"][_sm_key(p)]
    return abs(m_sm - M_H) < CLEAN_WINDOW and min(_non_sm(p).values()) > m_sm


def select(pts, meg):
    ok = [i for i, p in enumerate(pts) if clean(p)]
    allowed = [i for i in ok if meg[i]["n_meg_allowed"] > 0]
    kv = {i: meg[i]["kappa_v"] for i in ok}
    rules = [
        ("decoupled",
         max((i for i in allowed if min(_non_sm(pts[i]).values()) > 700), key=kv.get),
         "the decoupling limit: every new scalar is heavy and the 125 GeV state is SM-like "
         "to high precision. Rule: clean, MEG II-allowed, every non-SM scalar above 700 GeV, "
         "largest κ_V"),
        ("light H±",
         min(allowed, key=lambda i: pts[i]["masses_GeV"]["Hpm1"]),
         "the lightest charged scalar that survives MEG II, where its loop pulls "
         "R_γγ down. Rule: clean, MEG II-allowed, smallest m_H±"),
        ("near S₃",
         min(ok, key=lambda i: pts[i]["breaking"]["max_soft_over_v2"]),
         "the smallest soft breaking among clean points: the corner closest to the exact-S₃ "
         "model. Rule: clean, smallest largest-|soft term|/v²"),
    ]
    return rules, len(ok), len(allowed)


# ----------------------------------------------------------------------- tune

class Tuner:
    def __init__(self):
        self.ms = build_model(soft=True, soft_solve_for="mD1sq")
        self.m_exact = build_model()
        self.spectrum = soft_spectrum_function(self.ms)
        self.free, self.solved = soft_free_names(self.ms), soft_solved_names(self.ms)

    def run(self, lam, theta, phi, softs):
        v1, v2, vS = (np.atleast_1d(x) for x in soft_vevs(np.array([theta]), np.array([phi])))
        out = self.spectrum([np.array([x]) for x in lam], v1, v2, vS,
                            [np.array([x]) for x in softs])
        return out, (float(v1[0]), float(v2[0]), float(vS[0]))

    def m_sm(self, lam, theta, phi, softs):
        out, _ = self.run(lam, theta, phi, softs)
        k = max(SOFT_CP_EVEN_KEYS, key=lambda key: out["hvv"][key][0])
        return math.sqrt(out["mass2"][k][0])

    def tune(self, p):
        lam = [p["lambdas"][f"lambda_{k + 1}"] for k in range(8)]
        softs = [p["softs_GeV2"][n] for n in self.free]
        th, ph = p["theta"], p["phi"]
        f = lambda x, k: self.m_sm(lam[:k] + [x] + lam[k + 1:], th, ph, softs) - M_H
        h = 1e-4
        grads = [(f(lam[k] + h, k) - f(lam[k] - h, k)) / (2 * h) for k in range(8)]
        k = int(np.argmax(np.abs(grads)))
        # bracket around the root along the linear estimate, then bisect
        x0 = lam[k]
        step = -f(x0, k) / grads[k]
        lo, hi = sorted((x0, x0 + 2 * step))
        while f(lo, k) * f(hi, k) > 0:
            lo, hi = lo - abs(step), hi + abs(step)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(lo, k) * f(mid, k) <= 0:
                hi = mid
            else:
                lo = mid
        lam_t = lam[:k] + [0.5 * (lo + hi)] + lam[k + 1:]
        return self.record(p, lam_t, softs), k, x0, lam_t[k], grads[k]

    def record(self, p, lam, softs):
        """The tuned point as `scan_soft.scan_chunk` writes one, with every cut re-checked."""
        out, (v1, v2, vS) = self.run(lam, p["theta"], p["phi"], softs)
        L = np.array([lam])
        mass2 = {k: float(out["mass2"][k][0]) for k in SOFT_SPECTRUM_KEYS}
        mass = {k: math.sqrt(max(v, 0.0)) for k, v in mass2.items()}
        xi = {k: out["hvv"][k] for k in SOFT_CP_EVEN_KEYS}
        m_even = {k: np.array([mass[k]]) for k in SOFT_CP_EVEN_KEYS}
        cuts = {
            "strict BFB": bool(strict_bfb_mask(L)[0]),
            "unitarity": bool(unitarity_mask(L)[0]),
            "no tachyons": min(mass2.values()) > 0,
            "charged > 80": mass["Hpm1"] > 80 and mass["Hpm2"] > 80,
            "SM-like at M_H": bool(sm_like_cut(xi, m_even)[0]),
            "LEP neutral veto": bool(lep_neutral_mask(xi, m_even)[0]),
            "12-d backstop": bool(numeric_bfb_min(self.m_exact, L, n_directions=BACKSTOP_DIRECTIONS,
                                                  seed=9, subspace="all")[0] >= -1e-9),
            "Goldstones": float(out["goldstone"][0]) < 1e-9,
        }
        failed = [c for c, ok in cuts.items() if not ok]
        if failed:
            raise AssertionError(f"tuned point fails {failed}")
        soft_vals = dict(zip(self.free, softs))
        soft_vals.update({n: float(out["solved"][n][0]) for n in self.solved if n in self.ms.softs})
        return {
            "lambdas": {f"lambda_{k + 1}": float(lam[k]) for k in range(8)},   # unrounded
            "theta": p["theta"], "phi": p["phi"],
            "r_draft": float(draft_r(p["phi"])),
            "vevs_GeV": {"v1": v1, "v2": v2, "vS": vS},
            "softs_GeV2": dict(sorted(soft_vals.items())),
            "mu_sq_GeV2": {n: float(out["solved"][n][0]) for n in self.solved
                           if n not in self.ms.softs},
            "masses_GeV": mass,
            "hvv_squared": {k: float(xi[k][0]) for k in SOFT_CP_EVEN_KEYS},
            "cp_even_mixing": out["mixing"][0].tolist(),
            "breaking": {"max_soft_over_v2": max(abs(v) for v in soft_vals.values()) / V_EW**2,
                         "pi3_minus_phi": math.pi / 3 - p["phi"]},
            "cuts_passed": list(cuts),
        }


# ---------------------------------------------------------------- observables

def observables(T, p, check=True):
    """Notebooks 06, 07 and 08 at one point (their per-point logic, replicated)."""
    # notebook 06: kappa_V, R_gg, trilinears, VSS, SM-like h -> l l' (least constrained)
    r6 = S.scan_point(T, p, GRID) if check else _scan_point_unchecked(T, p)
    k6 = r6["sm_like"]
    lim_h = {f"{a}{b}": v for (a, b), v in D.LFV_LIMITS.items()}
    rows6 = []
    for e in r6["lfv"]:
        Q = e["absQ"][k6]
        rows6.append({q: M_H / (8 * math.pi) * (Q[a, b] ** 2 + Q[b, a] ** 2) / GAMMA_H_SM
                      for q, (a, b) in PAIRS.items()} | {"mu3": e["mu3"], "kind": e["kind"]})
    best6 = min(rows6, key=lambda x: max(x[q] / lim_h[q] for q in PAIRS)) if rows6 else None

    # notebooks 07 and 08: radiative decays over every (mu_3, branch)
    rot = S.point_rotations(T, p, check=check)
    g = S.hhh_charged(T, p, rot)
    k_sm = ("h1", "h2", "h3")[S.sm_like_index(rot)]
    m_sm = rot["mass"][k_sm]
    rows = []
    for c in GRID:
        for b in lfv.lepton_branches(T.g_of, c, rot["vevs"]):
            yuk = S.mass_basis_yukawas(T, b["Y"], rot)
            res = {n: S.lfv_radiative(T, p, rot, b["Y"], i, j, g, yukawas=yuk)
                   for n, (i, j, _) in CH.items()}
            Q = np.abs(yuk[1][k_sm])
            hbr = {q: m_sm / (8 * math.pi) * (Q[a, bb] ** 2 + Q[bb, a] ** 2) / GAMMA_H_SM
                   for q, (a, bb) in PAIRS.items()}
            me = res["mu_e"]
            AL, AR, Pq, Pw = me["A_L"], me["A_R"], me["parts"]["BZ_quark"], me["parts"]["BZ_W"]
            var = {"no_quark": RD.branching_ratio(AL - Pq[0], AR - Pq[1], ALPHA, G_F),
                   "top_flip": RD.branching_ratio(AL - 2 * Pq[0], AR - 2 * Pq[1], ALPHA, G_F),
                   "w_hkz": RD.branching_ratio(AL - (1 - W_HKZ) * Pw[0], AR - (1 - W_HKZ) * Pw[1],
                                               ALPHA, G_F)}
            lead = {n: max(res[n]["parts"], key=lambda q: abs(res[n]["parts"][q][0]) ** 2
                           + abs(res[n]["parts"][q][1]) ** 2) for n in CH}
            rows.append({"mu3": float(c), "kind": b["kind"],
                         "br": {n: res[n]["br"] for n in CH}, "lead": lead, "var": var, "h": hbr})

    def best_of(name):
        x = min(rows, key=lambda x: x["br"][name]) if rows else None
        return None if x is None else {"mu3": x["mu3"], "kind": x["kind"], "br": x["br"][name],
                                       "lead": x["lead"][name], "var": x["var"] if name == "mu_e" else None}

    return {
        "sm_like": k6, "kappa_v": r6["kappa_v"], "r_gammagamma": r6["r_gammagamma"],
        "g_hpp_GeV": r6["g_hpp"], "vss_widths_GeV": r6["vss"], "lfv_best_config": best6,
        "n_configurations": len(rows),
        "n_allowed": {n: sum(1 for x in rows if x["br"][n] < CH[n][2]) for n in CH},
        "best": {n: best_of(n) for n in CH},
    }


def _scan_point_unchecked(T, p):
    """`soft_decays.scan_point` without the stored-mass check (a tuned point has no 'stored')."""
    orig = S.point_rotations
    S.point_rotations = lambda tools, pt, check=True, check_mixing=True, rtol=1e-2: \
        orig(tools, pt, check=False)
    try:
        return S.scan_point(T, p, GRID)
    finally:
        S.point_rotations = orig


def validate(T, i, p, dec, meg, tau):
    """The validation gate: the replicated pipeline reproduces 06/07/08 at a scan point."""
    o = observables(T, p)
    sd, me, tl = dec["points"][i], meg["points"][i], tau["points"][i]
    rel = lambda a, b: abs(a - b) / max(abs(a), abs(b), 1e-300)
    checks = {
        "kappa_v": rel(o["kappa_v"], sd["kappa_v"]),
        "r_gammagamma": rel(o["r_gammagamma"], sd["r_gammagamma"]),
        "lfv_best taumu": rel(o["lfv_best_config"]["taumu"], sd["lfv_best_config"]["taumu"]),
        "mu_e best": rel(o["best"]["mu_e"]["br"], me["best"]["br"]),
        "n_meg_allowed": abs(o["n_allowed"]["mu_e"] - me["n_meg_allowed"]),
        "tau_mu best": rel(o["best"]["tau_mu"]["br"], tl["best_tau_mu"]["br"]["tau_mu"]),
        "tau_e best": rel(o["best"]["tau_e"]["br"], tl["best_tau_e"]["br"]["tau_e"]),
    } | {f"mu_e variant {k}": rel(o["best"]["mu_e"]["var"][k], me["best"]["var"][k])
         for k in me["best"]["var"]}
    tol = {"n_meg_allowed": 0}
    bad = {k: v for k, v in checks.items() if v > tol.get(k, 2e-3)}   # 07/08 store 4 figures
    if bad:
        raise AssertionError(f"point {i}: replicated pipeline disagrees with the notebooks: {bad}")
    return max(v for k, v in checks.items() if k != "n_meg_allowed")


# --------------------------------------------------------------------- render

def _e(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        return f"{x:.4g}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10^{int(e)}"


def _verdict(v, lim):
    return "✅ allowed" if v <= lim else f"❌ {v / lim:.1f}× limit"


def card(label, i, why, src, tuned, k, old, new, grad, ob_src, ob, conv):
    lim_h = conv["lfv_limits"]
    sm = _sm_key(tuned)
    lam = ", ".join(f"λ{key.split('_')[1]} = {v:+.4f}" for key, v in tuned["lambdas"].items())
    soft = ", ".join(f"{n} = {v / V_EW**2:+.2f} v²" for n, v in tuned["softs_GeV2"].items())
    names = list(tuned["masses_GeV"])
    out = [f"## {label}: scan point {i}, tuned", "", f"*Why this point:* {why}.", "",
           "**Tuning** (only one quartic moves; every scan cut re-checked)", "",
           f"- λ{k + 1}: {old:+.5f} → {new:+.5f} (Δ = {new - old:+.1e}; "
           f"∂m_h/∂λ{k + 1} = {grad:.0f} GeV)",
           f"- m_h: {src['masses_GeV'][_sm_key(src)]:.2f} → **{tuned['masses_GeV'][sm]:.3f} GeV**",
           f"- cuts passed: {', '.join(tuned['cuts_passed'])}", "",
           "**Inputs** (θ, φ and the free soft terms are the scan point's)", "",
           f"- vacuum: θ = {tuned['theta']:.3f}, φ = {math.degrees(tuned['phi']):.2f}° "
           f"(60° − φ = {math.degrees(tuned['breaking']['pi3_minus_phi']):.2f}°); "
           f"(v₁, v₂, v_S) = ({tuned['vevs_GeV']['v1']:.1f}, {tuned['vevs_GeV']['v2']:.1f}, "
           f"{tuned['vevs_GeV']['vS']:.1f}) GeV",
           f"- quartics: {lam}",
           f"- soft terms: {soft}; largest |soft|/v² = {tuned['breaking']['max_soft_over_v2']:.2f}",
           "", "**Spectrum and gauge couplings**", "",
           "| state | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
           "| mass [GeV] | " + " | ".join(f"{tuned['masses_GeV'][n]:.2f}" for n in names) + " |",
           "| $g_{hVV}^2/g_{\\rm SM}^2$ | " + " | ".join(
               _e(tuned["hvv_squared"][n]) if n in tuned["hvv_squared"] else "–" for n in names) + " |",
           "", "**Headline observables: scan point → tuned point**", "",
           "| observable | scan point | tuned | limit |", "|---|---|---|---|"]
    rows = [("κ_V", ob_src["kappa_v"], ob["kappa_v"], None),
            ("1 − κ_V²", 1 - ob_src["kappa_v"] ** 2, 1 - ob["kappa_v"] ** 2, None),
            ("R_γγ (κ_t = κ_V)", ob_src["r_gammagamma"], ob["r_gammagamma"], None)]
    for q, lab in (("taumu", "h→τμ"), ("taue", "h→τe"), ("emu", "h→eμ")):
        rows.append((f"{lab}, least constrained", ob_src["lfv_best_config"][q],
                     ob["lfv_best_config"][q], lim_h[q]))
    for n, lab in (("mu_e", "μ→eγ"), ("tau_mu", "τ→μγ"), ("tau_e", "τ→eγ")):
        rows.append((f"{lab}, least constrained", ob_src["best"][n]["br"], ob["best"][n]["br"], CH[n][2]))
    for lab, a, b, lim in rows:
        fmt = (lambda x: f"{x:.6f}") if lab == "κ_V" else _e
        out.append(f"| {lab} | {fmt(a)} | {fmt(b)} | {_e(lim) if lim else '–'} |")
    me = ob["best"]["mu_e"]
    vss = "; ".join(f"{key}: {_e(v)} GeV" for key, v in ob["vss_widths_GeV"].items())
    out += ["", "**Tuned point, in detail**", "",
            f"- $g_{{hH^+H^-}}$ = {', '.join(f'{x:.1f}' for x in ob['g_hpp_GeV'])} GeV; VSS widths: {vss}",
            f"- μ→eγ: least-constrained configuration μ₃ = {me['mu3']:.3f} GeV (*{me['kind']}*), "
            f"$\\mathcal B$ = {_e(me['br'])}: {_verdict(me['br'], CH['mu_e'][2])}. "
            f"{ob['n_allowed']['mu_e']} of {ob['n_configurations']} lepton configurations pass MEG II; "
            f"largest piece **{me['lead']}**; variants " + ", ".join(
                f"{key} {_e(v)}" for key, v in me["var"].items()),
            f"- τ→μγ best {_e(ob['best']['tau_mu']['br'])} (lead {ob['best']['tau_mu']['lead']}); "
            f"τ→eγ best {_e(ob['best']['tau_e']['br'])} (lead {ob['best']['tau_e']['lead']})", ""]
    meg_ok = ob["n_allowed"]["mu_e"] > 0
    out += [f"**Verdict:** the tuned point {'survives' if meg_ok else 'is excluded by'} MEG II. "
            "Assumptions it rests on: `A-prior-soft`, `A-local-min`, `A-quark-kappa`, "
            "`A-least-constrained`, `A-benchmark-tuning` (see [CLAIMS.md](CLAIMS.md)).", ""]
    return out


def main():
    scan = _load("viable_points_soft.json")
    dec, meg, tau = (_load(n) for n in ("soft_decay_benchmarks.json", "mu_e_gamma.json",
                                       "tau_l_gamma.json"))
    pts = scan["points"]
    for name, other in (("soft_decay_benchmarks", dec), ("mu_e_gamma", meg), ("tau_l_gamma", tau)):
        if len(other["points"]) != len(pts):
            raise AssertionError(f"{name}.json is not aligned with the soft scan; re-run its notebook")
    rules, n_clean, n_allowed = select(pts, meg["points"])
    T = S.build_tools()
    tuner = Tuner()

    cards, payload = [], {"m_h_target_GeV": M_H, "clean_window_GeV": CLEAN_WINDOW, "points": {}}
    summary = []
    for label, i, why in rules:
        gate = validate(T, i, pts[i], dec, meg, tau)
        ob_src = observables(T, pts[i])
        tuned, k, old, new, grad = tuner.tune(pts[i])
        assert abs(tuned["masses_GeV"][_sm_key(tuned)] - M_H) < 1e-3
        assert min(_non_sm(tuned).values()) > tuned["masses_GeV"][_sm_key(tuned)]
        assert abs(new - old) <= 0.01, f"{label}: Δλ = {new - old:.3g} is not a small tune"
        ob = observables(T, tuned, check=False)
        cards += card(label, i, why, pts[i], tuned, k, old, new, grad, ob_src, ob, dec["conventions"])
        payload["points"][label] = {"scan_point": i, "validation_max_rel_dev": gate,
                                    "tuned_quartic": f"lambda_{k + 1}", "lambda_old": old,
                                    "lambda_new": new, "record": tuned, "observables": ob}
        summary.append((label, i, tuned, ob))
        print(f"{label:10s} point {i}: λ{k + 1} {old:+.5f} → {new:+.5f}; m_h → "
              f"{tuned['masses_GeV'][_sm_key(tuned)]:.4f}; gate max rel dev {gate:.1e}")

    head = ["<!-- Generated by benchmarks.py from results/*.json; do not edit by hand. -->",
            "# 3HDM-S₃: three benchmark points at m_h = 125.20 GeV", "",
            f"Three points of the soft-broken scan, picked by fixed rules among its {n_clean} "
            f"*clean* points (SM-like mass within {CLEAN_WINDOW} GeV of M_H = {M_H} GeV, and the "
            f"SM-like state the lightest scalar; {n_allowed} of them MEG II-allowed). Each is then "
            "**tuned to m_h = 125.20 GeV exactly** by moving a single quartic by ~10⁻³, with every "
            "scan cut re-checked, and its observables are recomputed with the code of notebooks "
            "06–08. That code first reproduces what those notebooks stored for the untuned point "
            "(the validation gate), so the tuned numbers are the notebooks' numbers at a "
            "slightly moved point. The rules and the tuning are in `benchmarks.py`.", "",
            "| benchmark | scan point | tuned m_h [GeV] | κ_V | lightest non-SM scalar | R_γγ | μ→eγ (best) |",
            "|---|---|---|---|---|---|---|"]
    for label, i, t, ob in summary:
        name, mass = min(_non_sm(t).items(), key=lambda kv: kv[1])
        head.append(f"| {label} | {i} | {t['masses_GeV'][_sm_key(t)]:.3f} | {ob['kappa_v']:.6f} | "
                    f"{name} {mass:.0f} GeV | {ob['r_gammagamma']:.3f} | {_e(ob['best']['mu_e']['br'])} |")
    head.append("")
    (HERE / "BENCHMARKS.md").write_text("\n".join(head + cards).rstrip() + "\n")
    (RES / "benchmarks.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False,
                                                     default=float) + "\n")
    print("wrote BENCHMARKS.md and results/benchmarks.json")


if __name__ == "__main__":
    main()

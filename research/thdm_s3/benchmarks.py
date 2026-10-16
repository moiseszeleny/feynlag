"""Three benchmark points of the soft-broken scan, followed through every notebook.

A histogram over 2021 points shows what is typical. It does not show what one
point looks like from end to end: its inputs, its spectrum, its couplings, and
how it fares against each limit. This script picks three points by fixed rules
and writes one card per point to ``BENCHMARKS.md``, reading only the stored
results (`results/*.json`). Nothing is recomputed, so a card shows exactly what
the notebooks found.

The rules, applied to ``results/viable_points_soft.json``:

- **decoupled**: among points with a MEG II-allowed lepton configuration and
  every non-SM scalar above 700 GeV, the most aligned (largest $\\kappa_V$);
- **light H±**: among points with $R_{\\gamma\\gamma}<0.8$ (all have a light
  charged scalar) and a MEG II-allowed configuration, the one at the group's
  median $R_{\\gamma\\gamma}$, so the card shows the typical member, not the
  extreme one;
- **near S₃**: the point with the smallest soft breaking (largest soft term
  over $v^2$).

The rules are fixed so the same points come back after a re-run. If the
scan changes, the cards follow it.

Usage (from anywhere)::

    python research/thdm_s3/benchmarks.py

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
V_EW = 246.0


def _load(name):
    return json.loads((RES / name).read_text())


def load_all():
    scan = _load("viable_points_soft.json")
    dec = _load("soft_decay_benchmarks.json")
    meg = _load("mu_e_gamma.json")
    tau = _load("tau_l_gamma.json")
    pts = scan["points"]
    for name, other in (("soft_decay_benchmarks", dec), ("mu_e_gamma", meg),
                        ("tau_l_gamma", tau)):
        if len(other["points"]) != len(pts) or any(
                p.get("point", i) != i for i, p in enumerate(other["points"])):
            raise AssertionError(f"{name}.json is not aligned with the soft scan; "
                                 "re-run the notebook that writes it")
    return scan, dec, meg, tau


def _sm_label(d):
    return f"h{d['sm_like'] + 1}"


def _non_sm_masses(p, d):
    return {k: m for k, m in p["masses_GeV"].items() if k != _sm_label(d)}


def select(scan, dec, meg):
    pts, D, M = scan["points"], dec["points"], meg["points"]
    idx = range(len(pts))
    allowed = [i for i in idx if M[i]["n_meg_allowed"] > 0]

    heavy = [i for i in allowed if min(_non_sm_masses(pts[i], D[i]).values()) > 700]
    decoupled = max(heavy, key=lambda i: D[i]["kappa_v"])

    low = sorted((i for i in allowed if D[i]["r_gammagamma"] < 0.8),
                 key=lambda i: D[i]["r_gammagamma"])
    light = low[len(low) // 2]

    near = min(idx, key=lambda i: pts[i]["breaking"]["max_soft_over_v2"])
    return [("decoupled", decoupled,
             "the decoupling limit: every new scalar is heavy and the 125 GeV state "
             "is SM-like to high precision"),
            ("light H±", light,
             r"a light charged scalar interferes destructively with the $W$ in "
             r"$h\to\gamma\gamma$; the typical member of the $R_{\gamma\gamma}<0.8$ group"),
            ("near S₃", near,
             "the smallest soft breaking in the scan: the corner closest to the "
             "exact-S₃ model")]


# ------------------------------------------------------------------ render

def _e(x):
    """Compact scientific notation for a Markdown table."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        return f"{x:.4g}"
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10^{int(e)}"


def _pass(value, limit):
    return "✅ allowed" if value <= limit else f"❌ {value / limit:.1f}× limit"


def card(label, i, why, scan, dec, meg, tau):
    p, d = scan["points"][i], dec["points"][i]
    m, t = meg["points"][i], tau["points"][i]
    lim_h = dec["conventions"]["lfv_limits"]
    lim_r = tau["conventions"]["limits_90cl"]
    sm = _sm_label(d)
    out = [f"## {label}: point {i}", "", f"*Why this point:* {why}.", ""]

    # inputs
    soft = ", ".join(f"{k} = {v / V_EW**2:+.2f} v²" for k, v in p["softs_GeV2"].items())
    lam = ", ".join(f"λ{k.split('_')[1]} = {v:+.3f}" for k, v in p["lambdas"].items())
    out += ["**Inputs** (notebook 05)", "",
            f"- vacuum: θ = {p['theta']:.3f}, φ = {math.degrees(p['phi']):.2f}° "
            f"(60° − φ = {math.degrees(p['breaking']['pi3_minus_phi']):.2f}°); "
            f"VEVs (v₁, v₂, v_S) = ({p['vevs_GeV']['v1']:.1f}, {p['vevs_GeV']['v2']:.1f}, "
            f"{p['vevs_GeV']['vS']:.1f}) GeV",
            f"- quartics: {lam}",
            f"- soft terms: {soft}; largest |soft|/v² = "
            f"{p['breaking']['max_soft_over_v2']:.2f}", ""]

    # spectrum
    names = list(p["masses_GeV"])
    hvv = p["hvv_squared"]
    out += ["**Spectrum and gauge couplings** (notebook 05)", "",
            "| state | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
            "| mass [GeV] | " + " | ".join(f"{p['masses_GeV'][n]:.1f}" for n in names) + " |",
            "| $g_{hVV}^2/g_{\\rm SM}^2$ | " + " | ".join(
                _e(hvv[n]) if n in hvv else "–" for n in names) + " |", "",
            f"SM-like state: **{sm}**, with $\\kappa_V$ = {d['kappa_v']:.6f} and "
            f"$1-\\kappa_V^2$ = {_e(1 - d['kappa_v']**2)}. This is the admixture that "
            "carries all of the SM-like state's flavour violation (finding 12).", ""]

    # decays
    vss = "; ".join(f"{k}: {_e(v)} GeV" for k, v in d["vss_widths_GeV"].items())
    out += ["**Decays** (notebook 06)", "",
            f"- $R_{{\\gamma\\gamma}}$ = {d['r_gammagamma']:.3f} (top coupling assumed "
            "= κ_V)",
            f"- $g_{{hH^+H^-}}$ = {', '.join(f'{g:.1f}' for g in d['g_hpp_GeV'])} GeV",
            f"- VSS widths: {vss}", ""]

    # LFV
    b = d["lfv_best_config"]
    out += ["**LFV Higgs decays, least-constrained lepton configuration** (notebook 06)", "",
            f"μ₃ = {b['mu3']:.3f} GeV, branch kind *{b['kind']}*", "",
            "| channel | $\\mathcal B$ | limit | verdict |", "|---|---|---|---|"]
    for ch, lab in (("taumu", "h→τμ"), ("taue", "h→τe"), ("emu", "h→eμ")):
        out.append(f"| {lab} | {_e(b[ch])} | {_e(lim_h[ch])} | {_pass(b[ch], lim_h[ch])} |")
    out.append("")

    # radiative
    mb = m["best"]
    lead = max(mb["parts"], key=lambda k: abs(mb["parts"][k]))
    out += ["**Radiative LFV** (notebooks 07 and 08)", "",
            f"- μ→eγ, least-constrained configuration (μ₃ = {mb['mu3']:.3f} GeV, "
            f"*{mb['kind']}*): $\\mathcal B$ = {_e(mb['br'])} against "
            f"{_e(lim_r['mu_e'])}: {_pass(mb['br'], lim_r['mu_e'])}. "
            f"{m['n_meg_allowed']} of {m['n_configurations']} lepton configurations pass "
            f"MEG II. The largest piece is **{lead}**.",
            "- the same configuration under the three robustness variants: " + ", ".join(
                f"{k} {_e(v)}" for k, v in mb["var"].items()),
            f"- τ→μγ: best $\\mathcal B$ = {_e(t['best_tau_mu']['br']['tau_mu'])} "
            f"(limit {_e(lim_r['tau_mu'])}, lead {t['best_tau_mu']['lead']}); "
            f"τ→eγ: {_e(t['best_tau_e']['br']['tau_e'])} "
            f"(limit {_e(lim_r['tau_e'])}, lead {t['best_tau_e']['lead']})", ""]

    survives = m["n_meg_allowed"] > 0
    out += [f"**Verdict:** the point {'survives' if survives else 'is excluded by'} MEG II"
            + (" and is in the 610" if survives else "")
            + ". Assumptions it rests on: `A-prior-soft`, `A-local-min`, `A-quark-kappa`, "
            "`A-least-constrained` (see [CLAIMS.md](CLAIMS.md)).", ""]
    return out


def main():
    scan, dec, meg, tau = load_all()
    chosen = select(scan, dec, meg)
    out = ["<!-- Generated by benchmarks.py from results/*.json; do not edit by hand. -->",
           "# 3HDM-S₃: three benchmark points", "",
           f"Three of the {scan['n_points']} soft-broken points, each followed through "
           "notebooks 05–08. The values are read from the stored results, not recomputed. "
           "The selection rules are in `benchmarks.py`. Use the cards to build intuition "
           "and to spot-check: every number here can be found in a notebook.", "",
           "| benchmark | point | SM-like κ_V | lightest non-SM scalar | R_γγ | μ→eγ (best) |",
           "|---|---|---|---|---|---|"]
    for label, i, _ in chosen:
        p, d, m = scan["points"][i], dec["points"][i], meg["points"][i]
        name, mass = min(_non_sm_masses(p, d).items(), key=lambda kv: kv[1])
        out.append(f"| {label} | {i} | {d['kappa_v']:.6f} | {name} {mass:.0f} GeV | "
                   f"{d['r_gammagamma']:.3f} | {_e(m['best']['br'])} |")
    out.append("")
    for label, i, why in chosen:
        out += card(label, i, why, scan, dec, meg, tau)
    (HERE / "BENCHMARKS.md").write_text("\n".join(out).rstrip() + "\n")
    (RES / "benchmarks.json").write_text(json.dumps(
        {"rules": "see benchmarks.py", "points": {lab: i for lab, i, _ in chosen}},
        indent=1, ensure_ascii=False) + "\n")
    print("BENCHMARKS.md:", ", ".join(f"{lab} = point {i}" for lab, i, _ in chosen))


if __name__ == "__main__":
    main()

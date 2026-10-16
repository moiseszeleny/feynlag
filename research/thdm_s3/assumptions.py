"""The load-bearing assumptions the derivation ledgers do not record.

The notebooks' `derive.Ledger`s capture *derivation* steps: what was derived,
what was imported from a paper, and how each was checked. The methodological
choices the headline numbers rest on mostly live in prose instead: scan priors,
cuts, the vacuum criterion, approximations and the experimental limits. This
file lists those choices so `claims.py` can put them at the top of `CLAIMS.md`.

Each entry is a dict with these keys:

- ``id``: a stable key (``A-...``), used by ``review_status.csv``;
- ``statement``: what is assumed;
- ``introduced_in``: the pipeline node(s) that make the choice (keys of
  `pipeline.PIPELINE`, a string or a list); everything downstream is affected too;
- ``where``: where to read it (notebook §, NOTES finding);
- ``affects``: the headline results that depend on it;
- ``robustness``: what has been done to test it, or ``"untested"``.

**Going forward**, record a new notebook's choices in its own ledger, with
``ledger.step(..., obtained="assumed")``, and `claims.py` will pick them up. This
file then only needs entries for the 01-08 backlog and for choices made in
scripts, which have no ledger.

Text in ``$...$`` is maths, rendered on GitHub.
"""

ASSUMPTIONS = [
    {
        "id": "A-prior-exact",
        "statement": r"Exact-$S_3$ scan prior: the quartics are sampled uniformly, "
                     r"$\lambda\in U(-2,2)^8$.",
        "introduced_in": "scan.py",
        "where": "notebook 01; NOTES finding 5",
        "affects": "every *fraction* or *count* from the exact scan (1964 of 6e7 points, "
                   "the 16% with a non-SM scalar below 100 GeV, the 18% h→τμ exclusion). "
                   "Statements of *existence* do not depend on it.",
        "robustness": "untested: no alternative prior has been run",
    },
    {
        "id": "A-prior-soft",
        "statement": r"Soft-scan prior: the three free soft terms are signed uniform "
                     r"variables up to $(500\,\mathrm{GeV})^2$.",
        "introduced_in": "scan_soft.py",
        "where": "notebook 05; NOTES finding 11",
        "affects": "the 2021 soft points and every count built on them (610 MEG II, 2016 "
                   "h→ℓℓ′, 5 LFV-excluded), \"decoupling-like\", and \"light scalars do not "
                   "survive\". Only 15 points have every soft term below $v^2$.",
        "robustness": "untested: a log-uniform scan of the near-S₃ corner is an open item",
    },
    {
        "id": "A-local-min",
        "statement": "The soft scan requires only a local minimum (no tachyons), "
                     "not the global minimum of the potential.",
        "introduced_in": "scan_soft.py",
        "where": "NOTES finding 11 (Caveats); open questions",
        "affects": "the soft point set: some points may sit in a metastable vacuum",
        "robustness": "untested: a multi-start minimization of the survivors is an open item",
    },
    {
        "id": "A-bfb-slice",
        "statement": "The corrected boundedness condition is exact only on the real "
                     "neutral slice; complex-neutral and charged directions are "
                     "covered numerically, not analytically.",
        "introduced_in": ["scan.py", "scan_soft.py"],
        "where": "notebook 01 §5.3-§5.5, §8; notebook 04; NOTES finding 4",
        "affects": "both point sets (the soft terms do not touch the quartics, so the same "
                   "cut is inherited)",
        "robustness": "the full 12-dimensional numeric BFB removes a further 3 of 322 on the "
                      "benchmark points: real but small",
    },
    {
        "id": "A-hvv-cut",
        "statement": r"SM-likeness cut on the coupling: the state carrying $g_{hVV}$ sits at "
                     r"$125\pm3$ GeV with $\kappa^2\ge0.9$, and nothing below 114.4 GeV has "
                     r"$\xi^2\ge0.05$ [LEP2003].",
        "introduced_in": ["scan.py", "scan_soft.py"],
        "where": "notebooks 01 and 05; NOTES open questions (hVV entry)",
        "affects": "both point sets; in the soft scan this is the cut where most of the "
                   "parameter space dies",
        "robustness": "the result is unchanged for any LEP floor in [0.01, 0.10]; the "
                      r"$\kappa^2\ge0.9$ threshold itself has not been varied",
    },
    {
        "id": "A-least-constrained",
        "statement": r"A scalar point counts as allowed if *any* lepton configuration "
                     r"($\mu_3^\ell$, fit branch) passes the limit: the least-constrained one.",
        "introduced_in": ["06", "07", "08"],
        "where": "notebooks 06-08",
        "affects": "the per-point counts (5 LFV-excluded, 610 MEG II, 599 with h→ℓℓ′). "
                   "Per-configuration fractions are quoted separately.",
        "robustness": "a deliberate conservative choice; a per-configuration view exists "
                      "in the notebooks",
    },
    {
        "id": "A-quark-kappa",
        "statement": r"The quark couplings point along the vacuum direction: "
                     r"$\kappa_t=\kappa_V$. No quark sector is built per soft point.",
        "introduced_in": ["06", "07", "08"],
        "where": "notebook 06 (R_γγ); notebooks 07-08 (Barr-Zee); ledger 07 #8 and 08 #4",
        "affects": r"$R_{\gamma\gamma}$, the quark Barr-Zee terms in μ→eγ and τ→ℓγ, and the "
                   "missing heavy-state widths",
        "robustness": "three variants (no quark Barr-Zee, flipped quark sign, W loop at the "
                      "HKZ13 value) give 499, 456 and 675 survivors instead of 610",
    },
    {
        "id": "A-missing-diagrams",
        "statement": r"Barr-Zee diagrams not included: [HKZ13] set C, $\phi Z\gamma$ "
                     r"(about +6% on the $W$ term) and $H^\pm W\gamma$.",
        "introduced_in": "07",
        "where": "notebook 07 §2.3; NOTES finding 13",
        "affects": "μ→eγ and τ→ℓγ rates, the 610 count",
        "robustness": "only bounded through the W-loop variant above",
    },
    {
        "id": "A-exp-limits",
        "statement": "Experimental limits used: MEG II [MEGII25] for μ→eγ; Belle [Belle21] "
                     "and BaBar [BaBar10] for τ→ℓγ; CMS [CMS21], [CMS23] for h→ℓℓ′; "
                     "LEP [LEP2003] for light neutral scalars.",
        "introduced_in": ["03", "06", "07", "08"],
        "where": "NOTES References",
        "affects": "every exclusion count",
        "robustness": "review item: check each value, its year and confidence level against "
                      "the cited source",
    },
    {
        "id": "A-quark-fit-r",
        "statement": r"The quark fit's $r$ (and $V_{\rm abs}$) is an existence proof, not a "
                     "prediction: 9 parameters against 9 targets gives several exact branches, "
                     "and the one found depends on the restart seed and numpy/scipy version.",
        "introduced_in": "02",
        "where": "notebook 02 §11; NOTES finding 9 and the kernel note",
        "affects": "any quoted r or φ for the quark vacuum (φ = 53.54° in notebook 05)",
        "robustness": "documented; the CP phase is out of reach (real Yukawas, J = 0)",
    },
]

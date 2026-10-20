# Roadmap: lepton flavour violation in the S₃-symmetric 3HDM

**Working title:** *Lepton flavour violation in the S₃-symmetric 3HDM: residual-ℤ₂ selection
rules and the dominance of μ→eγ under soft breaking.*

This is the plan for turning notebooks 03 and 06–08 into one paper. It lists what is
established, what must still be done before the claims can be published, and in what order.
Findings are numbered as in [`NOTES.md`](NOTES.md); assumptions as in [`CLAIMS.md`](CLAIMS.md).

## 0. The thesis

1. **Selection rule.** At the aligned (exact-S₃) vacuum a residual ℤ₂ survives. The electron
   and the three in-plane non-vacuum scalars (h₀, one A, one H±) are odd; μ, τ and the other
   four scalars are even. Every amplitude whose external states have odd total parity
   vanishes **to all orders**: μ→eγ and τ→eγ are exactly zero, while τ→μγ is allowed. The
   SM-like Higgs (even) decays to τμ only; the odd h₀ decays to eμ and eτ only (nb03 §7.2).
2. **Soft breaking switches the forbidden channels on** in proportion to the ℤ₂ breaking,
   B ∝ ε². Off the alignment ε ∝ (60° − φ); in the soft scan the free soft terms break the ℤ₂
   whatever φ is.
3. **μ→eγ dominates.** In the soft vacuum, MEG II is the only LFV limit that cuts into the
   scalar sector: it keeps 618 of 2028 points, while h→ℓℓ′ keeps 2023 and τ→ℓγ removes
   nothing more. The reason is the W-loop Barr–Zee diagram through the SM-like Higgs. The
   τ rates allowed by every limit sit an order of magnitude below the Belle II reach, so the
   model predicts a null τ→ℓγ result there.

## 1. What is established

| claim | evidence | where |
|---|---|---|
| residual ℤ₂ forbids e-odd amplitudes at the exact vacuum | μ→eγ, τ→eγ are round-off (≤10⁻²⁹) at 1973 points; 3-odd/4-even coupling pattern at every evaluation | findings 13–14; nb07 §3, nb08 §3 |
| ε² switch-on off the alignment | log-log slopes 2.002 (μ→eγ) and 1.996 (τ→eγ), single-point probes | nb07 §3, nb08 §3 |
| μ→eγ is the binding LFV constraint | 618/2028 (MEG II) vs 2023/2028 (h→ℓℓ′); median 9×10⁴ more restrictive per configuration | finding 13; `report_mu_e_gamma` |
| robust to the two softest inputs | 504 / 453 / 684 survivors under the three variants | finding 13 |
| τ modes add nothing in the soft vacuum | 0.020 % / 0.029 % of configurations over Belle / BaBar; counts unchanged | finding 14; `report_tau_l_gamma` |
| exact-vacuum τ→μγ probes a light-scalar corner | 460 configurations at 20 points excluded by Belle, allowed by h→τμ; light even scalar (6.7–223 GeV) | finding 14 |
| amplitudes validated piece by piece | HKZ13 (A3), OST16 (17)/(19)/(20)/(32), Lavoura03, Ilisie15; extractor agreement 10⁻¹⁸ | nb07 §1–2, nb08 §1–2 |

## 2. Blocking gaps (must close before submission)

Each item names the claim it protects, the deliverable, and the gate that says it is done.

### B1. The selection rule as a theorem, not an observation
- **Why:** claim 1 is the paper's spine. Today it is a numerical pattern plus a symmetry
  argument in prose.
- **Do:**
  - Write the ℤ₂ generator explicitly: the S₃ reflection that fixes the aligned vacuum,
    with its action on the doublets, the lepton multiplets and the mass eigenstates.
  - Derive the parity of every physical field **symbolically**, from the Lagrangian, as a
    `derive.Ledger` step.
  - Prove that the Yukawa and scalar-potential couplings respect it. Then the vanishing to all
    orders follows from symmetry, not from the one- and two-loop diagrams that were computed.
- **Deliverable:** a new notebook `09_residual_z2.ipynb` (ledger `derivations_09.tex`).
- **Gate:**
  - The parity table is derived, not typed.
  - Its falsifier (a deliberately non-invariant coupling) fires.
  - It reproduces the 3-odd/4-even pattern of nb08 §3.

### B2. An order parameter for the ℤ₂ breaking in the soft vacuum
- **Why:** claim 2. The observation that φ is *not* the order parameter (finding 13) is the
  most interesting open thread, and a referee will ask what is.
- **Do:**
  - Expand the soft potential around the aligned vacuum.
  - Find the combination ε(m²_soft, φ) of soft terms that mixes ℤ₂-odd and ℤ₂-even states:
    the odd–even block of the mass matrices at first order.
  - Show that B(μ→eγ)/ε² collapses over the whole soft scan.
- **Deliverable:**
  - notebook `10_z2_order_parameter.ipynb`;
  - a per-point ε column, added through `store` (cache) plus a summary.
- **Gate:**
  - The collapse holds with a stated spread.
  - It reduces to (60° − φ)² on the zero-soft slice.

### B3. The missing two-loop pieces
- **Why:** claim 3 rests on the Barr–Zee W loop, and the W term carries a known 14 % gap
  to [HKZ13]'s number.
- **Do:** implement the φZγ Barr–Zee term, the H±W∓γ Barr–Zee term and [HKZ13]'s set C, or
  bound each one. Re-run `compute_mu_e_gamma.py` and `compute_tau_l_gamma.py`.
- **Gate:**
  - The W-loop gap to [HKZ13] closes, or is explained.
  - The 618 count moves by less than the existing variant band (453–684), or the claim is
    restated.

### B4. Quark couplings per soft point
- **Why:** the κ_t = κ_V ansatz (`A-quark-kappa`) feeds the Barr–Zee quark term, which
  interferes destructively with the W term.
- **Do:** build the soft-vacuum quark sector per point, at least the top and bottom Yukawa
  directions, reusing notebook 02's soft fit.
- **Gate:** the baseline count is recomputed; its shift is reported against the "no quark"
  and "flipped sign" variants.

### B5. Vacuum stability
- **Why:** `A-local-min`. The soft scan keeps local minima only.
- **Do:** run a multi-start minimization of the full potential on the 618 survivors.
- **Gate:** the number of survivors that sit in the global minimum, and whether that number
  changes any headline count.

### B6. The near-S₃ corner
- **Why:** the selection-rule story is most predictive where the ℤ₂ is approximately
  preserved. The uniform prior (`A-prior-soft`) has only 14 points with every soft term
  below v².
- **Do:** a targeted `scan_soft.py` mode: log-uniform soft terms, φ near π/3.
- **Gate:** B(μ→eγ) as a function of ε (B2) is mapped down to the MEG II limit, giving the
  smallest allowed ℤ₂ breaking.

## 3. Strengthening (should do)

### S1. A complete ℤ₂ parity table of LFV channels
Classify every channel by the total parity of its external states. The photon, Z, W and the
nuclei are even, so for charged-lepton decays only the electron count matters. For a scalar
decay the scalar's own parity enters.
- **Odd, so forbidden at the exact vacuum:**
  - μ→eγ, τ→eγ, μ→3e, μN→eN, τ→eμμ;
  - h_even→eμ and h_even→eτ;
  - h_odd→τμ (h₀, one A, one H± are the odd scalars).
- **Even, so allowed:**
  - τ→μγ, τ→3μ, τ→μee;
  - h_even→τμ;
  - h_odd→eμ and h_odd→eτ.

This turns claim 1 into a falsifiable pattern across experiments. Each assignment must be
checked against the B1 parity table, not taken from this list. The Higgs rows reproduce
nb03 §7.2.

### S2. μ→3e and μ–e conversion
- **What:** both share the dipole and add contact terms; both are ℤ₂-odd.
- **Prediction:** under soft breaking, their ratios to μ→eγ are a prediction. They are
  dipole-dominated if the contact terms are small.
- **Before quoting:** the experimental prospects (Mu3e, Mu2e/COMET) need verified,
  cited numbers.

### S3. τ→3μ and τ→μee
These are the even channels: the first allowed probes of the exact vacuum beyond τ→μγ. They
reuse the nb08 machinery plus tree-level scalar exchange.

### S4. The light-scalar corner
Find the direct-search status (LEP, LHC) of the light non-SM even scalars behind the
exact-vacuum τ→μγ exclusions. The scans only cut on the hVV coupling.

### S5. Consistency with lepton mixing (a key risk, see §6)
The residual ℤ₂ decouples the electron in the charged-lepton sector, so PMNS mixing involving
the electron must come from the neutrino sector. Build the neutrino sector (Dirac Yukawas on
H̃ plus S₃-allowed Majorana ν_R masses; feynlag has `seesaw_mass_matrix` and
`MajoranaRotation`). Then check whether it breaks the ℤ₂, and whether that breaking feeds
back into the scalar-sector LFV.

## 4. Paper outline

1. **Introduction:** S₃ 3HDMs, LFV as a probe; claims 1–3.
2. **Model and vacuum:** the potential, the aligned vacuum, soft breaking; the scan with its
   cuts (BFB correction, hVV coupling cut).
3. **The residual ℤ₂** (B1): parity table, all-orders selection rule, channel classification
   (S1).
4. **Breaking it softly** (B2): the order parameter ε, B ∝ ε², the collapse plot.
5. **Radiative decays:** the amplitude assembly and its validation (nb07–08, compressed).
6. **Results:**
   - the soft scan: μ→eγ against h→ℓℓ′ and τ→ℓγ;
   - the survivor counts;
   - the robustness (B3, B4, B5);
   - the near-S₃ corner (B6).
7. **Exact vacuum:** τ→μγ and the light-scalar corner (S4).
8. **Predictions:** the Belle II null result for τ→ℓγ; μ→3e and conversion ratios (S2).
9. **Conclusions.**

Figures:
- the parity table;
- the B/ε² collapse;
- the MEG II vs h→eμ scatter (`meg_scan.png`);
- the τ→ℓγ panels (`tau_l_gamma_scan.png`);
- the variant robustness bars.

## 5. Phases

| phase | content | gate (existing tooling) |
|---|---|---|
| A — theory backbone | B1, S1 | `derivations_09` ledger steps `derived`; falsifier fires |
| B — the order parameter | B2, then B6 | `check_quoted` pins the collapse spread; `run_pipeline.py` regenerates the new cache |
| C — robustness | B3, B4, B5 | survivor counts as `numbers.tex` macros; `provenance.py check` clean |
| D — extensions | S2, S3, S4, S5 | each as its own notebook and report |
| E — writing | the outline in §4 | every number through `numbers.tex`; `report_numbers.py check` clean; `CLAIMS.md` reviewed (`review_status.csv` marks) |

**What the paper needs.** Phases A–C are the minimum for the paper. D can be split: S1 and S2
belong in this paper, S4 and S5 could be follow-ups.

## 6. Risks

- **R1. Lepton mixing breaks the ℤ₂** (S5). If the neutrino sector needed for PMNS
  necessarily breaks it, claim 1 becomes "forbidden up to neutrino-sector effects". The size
  of those effects must then be shown to be negligible, or the selection rule restated.
  **Check this early**, in parallel with phase A.
- **R2. The missing diagrams shift the count** (B3) beyond the variant band. In that case the
  headline becomes a range rather than one number.
- **R3. Global-minimum cuts remove survivors** (B5). This changes the count, but not
  claims 1–2.
- **R4. Prior dependence** (`A-prior-soft`, `A-prior-exact`). Counts and fractions depend on
  the prior; statements of existence and of selection rules do not. The paper should lead with
  the latter.

## 7. Assumptions this paper inherits

`A-prior-soft`, `A-local-min`, `A-bfb-slice`, `A-hvv-cut`, `A-least-constrained`,
`A-quark-kappa`, `A-missing-diagrams`, `A-exp-limits`: see [`CLAIMS.md`](CLAIMS.md) §1.
- **Retired by this roadmap:** B4 retires `A-quark-kappa`, B3 retires `A-missing-diagrams`
  and B5 retires `A-local-min`.
- **Weakened:** B6 weakens `A-prior-soft`.

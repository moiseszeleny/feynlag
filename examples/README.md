# Examples

Runnable worked models, one physics topic per script. Each one runs from end
to end (`python examples/<name>.py`) and also serves as a manual smoke test. The
executed tutorial notebooks that walk these models step by step are in
[`docs/tutorials/`](../docs/tutorials/). The annotated gallery is
[`docs/examples.md`](../docs/examples.md).

| Script | Physics | Companion notebook | Pinned by |
|---|---|---|---|
| [`sm_scalar_gauge.py`](sm_scalar_gauge.py) | Complete SM: Higgs, electroweak gauge, leptons, quarks/QCD; LaTeX + UFO export | [`SM_Feynman_Rules_Tutorial`](../docs/tutorials/SM_Feynman_Rules_Tutorial.ipynb), [`SM_Walkthrough.md`](../docs/tutorials/SM_Walkthrough.md) | `test_scalar_pipeline_sm.py`, `test_gauge_sector_sm.py`, `test_qcd.py` |
| [`sm_vll.py`](sm_vll.py) | SM + vector-like lepton doublet: biunitary 2×2 diagonalization, RH Z-FCNC | [`SM_VLL_Tutorial`](../docs/tutorials/SM_VLL_Tutorial.ipynb) | `test_vll.py` |
| [`sm_u1x.py`](sm_u1x.py) | SM × U(1)_X with a Z′, symbolic charges; the guided basis finds the photon and one Z–Z′ angle | [`SM_U1X_Tutorial`](../docs/tutorials/SM_U1X_Tutorial.ipynb) | `test_u1x.py` |
| [`sm_ckm.py`](sm_ckm.py) | CKM quark mixing by insertion in the charged current, GIM; UFO export | — | `test_ckm.py` |
| [`thdm.py`](thdm.py) | Softly-broken Z₂ 2HDM scalar sector: guided basis, `tan 2α` and the Goldstone-fixed `tan β` | — | `test_scalar_pipeline_thdm.py` |
| [`thdm_s3.py`](thdm_s3.py) | 3HDM with S₃ flavour symmetry, forced √3 vacuum alignment | [`THDM_S3_Tutorial`](../docs/tutorials/THDM_S3_Tutorial.ipynb) | `test_thdm_s3.py` |
| [`fermi_theory.py`](fermi_theory.py) | Dim-6 four-fermion muon-decay operator (FFFF track); UFO export | — | `test_four_fermion.py` |
| [`sm_weinberg.py`](sm_weinberg.py) | Dim-5 Weinberg operator: Majorana ν mass `m_ν=−c v²/Λ`, Takagi, `term_coverage(max_dim=5)` | — | `test_majorana.py` |
| [`sm_seesaw.py`](sm_seesaw.py) | Type-I seesaw: light + heavy spectrum, `MajoranaRotation`, heavy-ν couplings | [`SM_Seesaw_Tutorial`](../docs/tutorials/SM_Seesaw_Tutorial.ipynb) | `test_seesaw.py` |
| [`sm_decays.py`](sm_decays.py) | Two-body widths `Γ(Z→νν̄)`, `Γ(Z→ℓℓ)`, `Γ(W→ℓν)`, `Γ(h→ℓℓ)` vs PDG | [`Particle_Decays_Tutorial`](../docs/tutorials/Particle_Decays_Tutorial.ipynb) | `test_pheno.py`, `test_models.py`, `test_offshell.py` |
| [`sm_higgs_decays.py`](sm_higgs_decays.py) | Full Higgs BR table: tree `f f̄` via `DiracParticle`, off-shell `WW*/ZZ*`, loop `gg/γγ/Zγ` | [`Particle_Decays_Tutorial`](../docs/tutorials/Particle_Decays_Tutorial.ipynb) | `test_pheno.py`, `test_offshell.py`, `test_loop.py` |
| [`ee_to_ff.py`](ee_to_ff.py) | 2→2 `e⁺e⁻→μ⁺μ⁻` through the Z: `dσ/dcosθ`, forward–backward asymmetry | [`Scattering_Tutorial`](../docs/tutorials/Scattering_Tutorial.ipynb) | `test_scattering.py` |

**Two styles.** The model-building scripts (`sm_scalar_gauge`, `sm_vll`,
`sm_u1x`, `thdm`, `thdm_s3`, `sm_weinberg`, `sm_seesaw`, `sm_decays`) write
their terms with the field-level constructors (`yukawa`, `tilde`, `inner`,
`dirac_mass`, `majorana_mass`, `weinberg`, `L.add_kinetic`), declare
parameters with `P.external`/`P.internal`, and, where a block is supported,
let `model.mass_blocks()`/`model.diagonalize()` find the physical basis.
`fermi_theory`, `sm_ckm`, `ee_to_ff` and `sm_higgs_decays` write their few
terms at component level: a four-fermion operator has no constructor, and
the other three are phenomenology demos (the Higgs-decay model deliberately
keeps single-component, decay-facing Yukawas).

The notebooks without a companion script (`SUN_Groups_Tutorial`,
`DiscreteGroups_Tutorial`, `ModelBuilding_Tutorial`) build their models
inside the notebook.

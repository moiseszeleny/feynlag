"""Standard Model inputs for the research notebooks, in one place [PDG].

Every value is the central value of the PDG 2024 Review of Particle Physics
(S. Navas et al., Phys. Rev. D 110, 030001 (2024)), read off its summary tables
and its "CKM Quark-Mixing Matrix" review, except where a comment says otherwise.
Before this module the same numbers were typed in five places (notebooks 02, 03,
06, ``lfv.py``, ``soft_decays.py``) and had drifted to PDG 2022 values while
citing PDG 2024; import from here instead of retyping.

Masses are in GeV.

``V_EW = 246.0`` is deliberately *not* here: it is the convention every scan
point of notebooks 01 and 05 was generated with (``model.V_EW``), not an input
read from the PDG, and changing it would invalidate those scans.
"""

# --- gauge and Higgs bosons -------------------------------------------------
M_W = 80.3692            # W mass, 80.3692 ± 0.0133
M_Z = 91.1880            # Z mass, 91.1880 ± 0.0020
M_H = 125.20             # Higgs mass, 125.20 ± 0.11 (S = 1.4)

#: Total width of a SM Higgs boson, the denominator of every LFV branching ratio.
#: This is the SM *prediction* (4.1 MeV ± 3.9% at m_H = 125 GeV, LHC Higgs Cross
#: Section Working Group, CERN Yellow Rep. Monogr. 2 (2017), arXiv:1610.07922, as
#: quoted by CMS, arXiv:2601.05168), the convention the LFV searches use. It is not
#: the measured width, which PDG 2024 gives as 3.7 +1.9 -1.4 MeV.
GAMMA_H_SM = 4.1e-3

# --- charged leptons --------------------------------------------------------
M_E = 0.51099895e-3      # 0.51099895000 ± 0.00000000015 MeV
M_MU = 0.1056583755      # 105.6583755 ± 0.0000023 MeV
M_TAU = 1.77693          # 1776.93 ± 0.09 MeV
LEPTON_MASSES = (M_E, M_MU, M_TAU)

# --- quarks -----------------------------------------------------------------
# u, d, s: MS-bar at mu = 2 GeV; c, b: MS-bar m(m); t: direct measurements.
M_U, M_D, M_S = 2.16e-3, 4.70e-3, 93.5e-3
M_C, M_B = 1.2730, 4.183
M_T = 172.57             # 172.57 ± 0.29 (S = 1.5)

# --- CKM magnitudes (the review's equation numbers in the 2024 edition) ------
V_US = 0.22431           # (12.8): K_l3 / K_mu2 average
V_CB = 0.0411            # (12.11): inclusive (42.2) and exclusive (39.8) combined, error scaled
V_UB = 0.00382           # (12.12): inclusive and exclusive average

# --- couplings --------------------------------------------------------------
ALPHA = 1 / 137.035999     # fine-structure constant at q^2 = 0 (Thomson limit), the
                            # coupling of on-shell photons in h -> gamma gamma; the
                            # digits beyond these differ between CODATA editions

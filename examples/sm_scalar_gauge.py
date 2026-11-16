"""SM Higgs + electroweak gauge sector + lepton + quark/QCD sector with
feynlag, end to end.

Run:  python examples/sm_scalar_gauge.py
"""

import sympy as sp

from feynlag import (
    FL_I, FL_J, DiracGamma, Lagrangian, Model, ParameterSet, SU3,
    WeylFermion, conjugate_pair, cubic_couplings, diracPL, diracPR,
    electroweak_scaffold, extract_fermion_vertices, fermion_feynman_rule,
    fermion_mass_matrix, latex_feynman_table, quartic_couplings, tilde,
    yukawa,
)
from feynlag.export.ufo.vvvv import (ADJOINT_VVV_COLOR,
                                     adjoint_vvv, adjoint_vvvv,
                                     assemble_vvvv)


def main():
    # --- symmetries and parameters -------------------------------------
    # electroweak scaffold (SU(2)×U(1), Higgs) from feynlag.models; the QCD
    # colour sector is this example's own addition.
    ew = electroweak_scaffold(lam=0.129)
    SU2L, U1Y, H = ew.SU2L, ew.U1Y, ew.H
    v, mu2 = ew.v, ew.mu2
    P = ParameterSet(*ew.parameters)          # declare once: create + register
    gs = P.external("gs", 1.22, positive=True)
    SU3c = SU3("SU3c", coupling=gs)

    Ll = WeylFermion("Ll", reps={SU2L: 2, U1Y: -sp.Rational(1, 2)},
                     chirality="L", nflavors=3, component_names=["nuL", "eL"])
    eR = WeylFermion("eR", reps={U1Y: -1}, chirality="R", nflavors=3,
                     component_names=["eR"])

    # quark sector: SU(2) doublet Q_L + color-triplet singlets u_R, d_R.
    # flavor structure mirrors the lepton sector exactly — 3 generic,
    # flavor-indexed generations, undiagonalized Yukawas (Yu[i,j]/Yd[i,j]),
    # no CKM here: CKM mixing is physically orthogonal to SU(3) vertex dynamics
    # (gluon vertices are flavor-diagonal/color-universal regardless of it),
    # and a fully generic 3-generation complex-Yukawa SVD doesn't resolve in
    # closed symbolic form via diagonalize_svd. The mass-basis + CKM-insertion
    # route (the FeynRules-SM way) is demonstrated in examples/sm_ckm.py.
    QL = WeylFermion("QL", reps={SU2L: 2, U1Y: sp.Rational(1, 6), SU3c: 3},
                     chirality="L", nflavors=3,
                     component_names=["uL_1", "uL_2", "uL_3",
                                      "dL_1", "dL_2", "dL_3"])
    uR = WeylFermion("uR", reps={U1Y: sp.Rational(2, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["uR_1", "uR_2", "uR_3"])
    dR = WeylFermion("dR", reps={U1Y: -sp.Rational(1, 3), SU3c: 3},
                     chirality="R", nflavors=3,
                     component_names=["dR_1", "dR_2", "dR_3"])

    W, B = ew.W, ew.B
    G = SU3c.bosons("G")

    # --- Lagrangian (field level) -------------------------------------------
    # Each constructor does the gauge contraction, the flavour indices and
    # the + h.c.; the colour index of the quark terms is a spectator, summed
    # diagonally.  A flavour matrix (IndexedBase) gives Y[fl_i, fl_j] with
    # the canonical indices FL_I (barred doublet leg) and FL_J (singlet leg).
    Ye, Yu, Yd = (sp.IndexedBase(n) for n in ("Ye", "Yu", "Yd"))
    LYuk = yukawa(Ye, Ll, H, eR)                     # −Ye L̄ H e_R + h.c.
    LYuk_d = yukawa(Yd, QL, H, dR)                   # −Yd Q̄ H d_R + h.c.
    LYuk_u = yukawa(Yu, QL, tilde(H), uR)            # −Yu Q̄ H̃ u_R, H̃ = iσ₂H*

    L = Lagrangian()
    ew.add_higgs(L)                                    # kinetic + potential
    L.add(LYuk + LYuk_d + LYuk_u, sector="yukawa")
    L.add_kinetic(Ll, eR, QL, uR, dR)                  # the gauge currents

    model = Model("SM-EW+QCD", gauge_groups=[SU2L, U1Y, SU3c],
                  fields=[H, Ll, eR, QL, uR, dR, W, B, G],
                  parameters=P, lagrangian=L)
    print(model.summary())
    print()

    i, j = FL_I, FL_J
    Gp = H.Gp
    eLbar, eRc = Ll.eLbar, eR.eR

    # --- pipeline -----------------------------------------------------------
    # validate() is the umbrella: symmetry/hermiticity/dimension + gauge-anomaly
    # cancellation in one report (the SM fermion content below is anomaly-free).
    print("validation:")
    print(model.validate().summary())
    print("tadpole:   ", model.solve_tadpoles([mu2]))

    h, G0 = H.H0_r, H.H0_i
    print("m_h² =", model.mass_matrix([h])[0, 0])

    # --- gauge sector: the model finds the blocks and their rotations ------
    # (W¹,W² → W± from the unbroken charge; W³,B → Z,γ with the photon
    # fixed as the massless direction of Q = T3 + Y)
    charged, neutral = model.mass_blocks(kind="gauge")
    print("gauge blocks:")
    print("  ", charged)
    print("  ", neutral)
    print("gauge M² =")
    sp.pprint(model.gauge_mass_matrix([*W.components, *B.components]))

    # names: massive states first, the photon last.  No angle= here, so the
    # rotation is explicit in g, g' (with angle="thetaW" the couplings would
    # stay in a free θ_W until tan θ_W = g'/g is substituted)
    Z, A = model.diagonalize(neutral, names=["Z", "A"]).new_fields
    # names: the positive-charge states; their conjugates are named too
    Wp, Wm = model.diagonalize(charged, names=["Wp"],
                               conjugate_names=["Wm"]).rotation.new_fields

    # --- scalar Feynman rules -----------------------------------------------
    Gm, cmap = conjugate_pair(Gp, "Gm")
    boson_fields = [h, G0, Gp, Gm, Z, A, Wp, Wm]

    # charge-based validation now that the physical basis exists: every vertex
    # conserves electric charge, the declared charges agree with the operator
    # derived from the vacuum (Q ∝ T3+Y, W± = ±1 derived automatically), and
    # every vertex pairs with its hermitian conjugate.
    charge_report = model.validate(
        invariance=False, anomalies=False,
        charges={h: 0, G0: 0, Gp: 1, Gm: -1, Z: 0, A: 0, Wp: 1, Wm: -1},
        fields=boson_fields, conjugate_map=cmap,
        conjugates={Gp: Gm, Gm: Gp, Wp: Wm, Wm: Wp})
    print("\ncharge / hermiticity validation:")
    print(charge_report.summary())

    rules = model.feynman_rules(boson_fields, conjugate_map=cmap,
                                simplifier=sp.simplify)
    print(f"\n{len(rules)} scalar/gauge vertices; a few highlights:")
    for key in list(rules)[:6]:
        print(" ", key, "->", rules[key])

    # --- fermion mass matrix -------------------------------------------------
    M_e = fermion_mass_matrix(LYuk, eLbar, eRc, model.vacuum, 3, (i, j),
                              gamma=diracPR)
    print("\nlepton mass matrix M_e =")
    sp.pprint(M_e)

    # quark mass matrices: color-diagonal, so any single color slot's
    # (bar_base, field_base) pair recovers the full flavor structure — no
    # diagonalization/CKM here (see the quark-sector comment above).
    M_u = fermion_mass_matrix(LYuk_u, QL.uL_1bar, uR.uR_1, model.vacuum, 3,
                              (i, j), gamma=diracPR)
    M_d = fermion_mass_matrix(LYuk_d, QL.dL_1bar, dR.dR_1, model.vacuum, 3,
                              (i, j), gamma=diracPR)
    print("\nup-quark mass matrix M_u =")
    sp.pprint(M_u)
    print("\ndown-quark mass matrix M_d =")
    sp.pprint(M_d)

    # --- electroweak quartic gauge couplings (physical basis) -------------
    # The weak->physical rotation U is derived from the Rotation objects
    # registered above, so this needs no hand-typed matrix. A VVVV coupling is
    # three numbers (one per UFO Lorentz structure) tied to the leg ordering,
    # so they live in Vertex.meta['structures'] rather than .coupling.
    # These are in feynlag's own convention; the field->particle leg sign a
    # UFO needs (which flips the cubics but not ggg) is applied at export by
    # the writer — see feynlag.export.ufo.legs.
    print("\nelectroweak gauge self-couplings (physical basis):")
    for vtx in model.gauge_vertices(groups=[SU2L]):
        legs = " ".join(str(x) for x in vtx.particles)
        if vtx.structures is None:
            body = str(sp.simplify(vtx.coupling))
        else:
            body = ", ".join(f"{n} = {sp.simplify(c)}"
                             for n, c in sorted(vtx.structures.items()))
        print(f"  [{vtx.vertex_type:4s}] {legs:16s} {body}")

    # --- QCD self-coupling: internal verification only, not UFO input ------
    # (gluons are ONE UFO particle "g" with the color-adjoint index summed
    # via a color-tensor string, not 8 separate weak-basis components — see
    # export/ufo/writer.py for the color-export side of this).
    G1, G2, G3 = G.components[0], G.components[1], G.components[2]
    ggg = cubic_couplings(SU3c)[(G1, G2, G3)]
    print("\nggg raw coefficient (g_1 g_2 g_3) ->", ggg,
          " == -gs ?", sp.simplify(ggg + gs.s) == 0)
    # ...but that value CONTAINS the colour factor f^abc; it coincides with
    # the exported coupling only because f^123 = 1. The UFO vertex is ONE
    # gluon repeated with a colour tensor, so the exported coupling is the
    # colour-stripped one — see export/ufo/vvvv.py::adjoint_vvv.
    print("ggg UFO coupling (color-stripped) ->", adjoint_vvv(SU3c),
          f"  with color {ADJOINT_VVV_COLOR}")

    G4, G5 = G.components[3], G.components[4]
    gggg_raw = assemble_vvvv(quartic_couplings(SU3c), (G1, G2, G4, G5))
    print("gggg raw coefficients (g_1 g_2 g_4 g_5, VVVV1/2/3) ->", gggg_raw)
    # ...but those still contain the color contraction sum_e f_{ije} f_{kle}
    # for that specific quadruple. The UFO vertex is ONE gluon repeated four
    # times with the color tensor carrying the adjoint index, so the exported
    # coupling is the COLOR-STRIPPED one: i g_s^2 on each structure (= MG's
    # stock sm GC_12). See export/ufo/vvvv.py::adjoint_vvvv.
    print("gggg UFO couplings (color-stripped) ->", adjoint_vvvv(SU3c))

    # --- fermion Feynman rules (Yukawa + gauge currents, physical basis) ---
    # physical_lagrangian applies the vacuum shift (H0 -> (v+h+iG0)/√2), the
    # tadpole solution, and the registered Z/A/W± rotations to EVERY sector,
    # including "yukawa" — the h/G0-lepton couplings and the gauge currents
    # come out of the same extraction pass.
    LYuk_phys = model.physical_lagrangian(sector="yukawa").xreplace(cmap)
    table = extract_fermion_vertices(LYuk_phys, boson_fields)

    mu = sp.Symbol("mu", integer=True)
    gammaL = DiracGamma(mu) * diracPL
    gammaR = DiracGamma(mu) * diracPR
    eL, nuLbar, eRbar = Ll.eL, Ll.nuLbar, eR.eRbar

    def show(name, bar, gamma, field, boson):
        key = (bar, gamma, field)
        coeff = table[key][1][(boson,)]
        rule = sp.simplify(fermion_feynman_rule(coeff, gamma, (boson,)))
        print(f"  {name:10s} ->", rule)

    # note: the Yukawa term carries TWO flavor indices (Ye[i,j]) since it
    # connects the doublet's flavor i to eR's flavor j; the gauge currents
    # above are flavor-diagonal (same index on both legs) instead.
    print("\nh/G0-lepton Yukawa vertices:")
    show("h eL eR", eLbar[i], diracPR, eRc[j], h)

    m_ell = sp.Symbol("m_ell", positive=True)
    hll_coeff = table[(eLbar[i], diracPR, eRc[j])][1][(h,)]
    hll_rule = fermion_feynman_rule(hll_coeff, diracPR, (h,))
    hll_pinned = hll_rule.subs(Ye[i, j], sp.sqrt(2) * m_ell / v.s)
    print("  single flavor, m_ell = Ye v/√2 ->", sp.simplify(hll_pinned),
         " (+ h.c.)")

    print("\ngauge-current vertices:")
    show("W+ nu e", nuLbar[i], gammaL, eL[i], Wp)
    show("Z eL eL", eLbar[i], gammaL, eL[i], Z)
    show("Z eR eR", eRbar[i], gammaR, eRc[i], Z)
    show("A eL eL", eLbar[i], gammaL, eL[i], A)
    show("A eR eR", eRbar[i], gammaR, eRc[i], A)

    # --- LaTeX preview --------------------------------------------------------
    table_preview = {k: v for k, v in list(rules.items())[:5]}
    latex = latex_feynman_table(table_preview)
    print("\nLaTeX preview:\n", latex[:400], "...")


if __name__ == "__main__":
    main()

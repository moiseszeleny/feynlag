"""Export a pure-QCD (SU(3) + one quark flavour) UFO from feynlag.

Built so ``u ubar -> g g`` can be run in MadGraph: that process is the only
cheap one whose amplitude is **linear** in the triple-gluon coupling (t/u
channel quark exchange, carrying no ggg, interfering with the s-channel gluon,
carrying one), so its sign is observable.  ``g g -> g g`` goes as ggg^2 and is
sign-blind.

feynlag's QCD sector has never been exercised in any process — the MadGraph
round-trip covers e+e- -> mu+mu- and e+e- -> W+W-, both QCD-free — while three
separate colour/normalisation defects were fixed in it recently, each caught by
reading MadGraph's shipped model files rather than by running anything.

Everything here comes through the library's own pipeline
(``fermion_gauge_current`` -> ``extract_fermion_vertices``), never hand-typed,
so the run validates feynlag and not a fixture.  ``tests/test_ufo_qcd.py``'s
QCD UFO is deliberately NOT reused: it is a colour-string smoke test whose qqg
is chiral-left only and whose coupling is the component-specific ``gs/2``.

Run:  python scripts/export_qcd_ufo.py [output_dir]
"""

import sys
from pathlib import Path

import sympy as sp

from feynlag import (
    DiracGamma, ExternalParameter, Lagrangian, Model, ParameterSet, SU3,
    WeylFermion, diracPL, diracPR, extract_fermion_vertices,
    fermion_gauge_current, verify_ufo_numeric,
)
from feynlag.export.ufo import (
    ADJOINT_VVV_COLOR, ADJOINT_VVVV_COLORS, UFOParticle, adjoint_vvv,
    adjoint_vvvv, write_ufo,
)

#: alpha_s = 0.118 at m_Z, the point MadGraph's stock sm uses by default
GS_VALUE = 1.2177157847767197          # sqrt(4 pi alpha_s), alpha_s = 0.118

FLAVOR = sp.Symbol("fl_i", integer=True)
MU = sp.Symbol("mu", integer=True)


def build_model():
    """SU(3)_c with one vector-like quark flavour (u_L + u_R triplets)."""
    gs = ExternalParameter("gs", GS_VALUE, positive=True)
    SU3c = SU3("SU3c", coupling=gs)
    uL = WeylFermion("uL", reps={SU3c: 3}, chirality="L", nflavors=1,
                     component_names=["uL_1", "uL_2", "uL_3"])
    uR = WeylFermion("uR", reps={SU3c: 3}, chirality="R", nflavors=1,
                     component_names=["uR_1", "uR_2", "uR_3"])
    G = SU3c.bosons("G")

    L = Lagrangian()
    L.add(fermion_gauge_current(uL, FLAVOR)
          + fermion_gauge_current(uR, FLAVOR), sector="gauge")

    model = Model("QCD_1flavour", gauge_groups=[SU3c], fields=[uL, uR, G],
                  parameters=[gs], lagrangian=L)
    return model, dict(SU3c=SU3c, uL=uL, uR=uR, G=G, gs=gs)


def colour_stripped_qqg(s, chirality):
    """The qqg coupling with the colour matrix element divided out.

    ``fermion_gauge_current`` gives ``g_s * T^a_{ij}`` per colour component —
    it CONTAINS the colour factor, exactly as ``cubic_couplings`` contains
    ``f^{abc}``.  A UFO qqg vertex is one quark and one gluon particle with
    the colour indices carried by a ``T(...)`` tensor, so the coupling that
    pairs with it must be colour-stripped, and the stripped value is plain
    ``g_s``.

    Derived here rather than asserted: every generator and every colour pair
    is checked to give the same ratio, so a wrong colour normalisation shows
    up as a raised exception, not a silently wrong number.
    """
    SU3c, G, gs = s["SU3c"], s["G"], s["gs"]
    field = s["uL"] if chirality == "L" else s["uR"]
    gamma = DiracGamma(MU) * (diracPL if chirality == "L" else diracPR)
    T = SU3c.generators(3)
    bar, comp = field.bar_components, field.components

    current = fermion_gauge_current(field, FLAVOR)
    ratios = set()
    for a in range(SU3c.n_generators):
        Ga = G.components[a]
        table = extract_fermion_vertices(current, [Ga])
        for r in range(3):
            for c in range(3):
                tij = T[a][r, c]
                if tij == 0:
                    continue
                key = (bar[r][FLAVOR], gamma, comp[c][FLAVOR])
                coeff = table.get(key, {}).get(1, {}).get((Ga,), 0)
                ratios.add(sp.nsimplify(sp.simplify(coeff / tij)))
    if len(ratios) != 1:
        raise RuntimeError(
            f"qqg coupling is not a uniform multiple of T^a_ij: {ratios}")
    return ratios.pop()


def export(path, qqg_color="T(3,2,1)"):
    model, s = build_model()
    gs = s["gs"]

    left = colour_stripped_qqg(s, "L")
    right = colour_stripped_qqg(s, "R")
    if sp.simplify(left - right) != 0:
        raise RuntimeError(f"QCD is vector-like; got L={left} R={right}")

    u, ubar, g = sp.symbols("u ubar g")
    particles = [
        UFOParticle(u, 2, "u", antiname="u~", spin=2, color=3,
                    charge=sp.Rational(2, 3), antisymbol=ubar),
        UFOParticle(g, 21, "g", spin=3, color=8),
    ]
    # the writer supplies the Feynman-rule i, so these are coefficients
    fermion_vertices = [
        dict(bar=ubar, field=u, bosons=(g,), left=left, right=right,
             color=qqg_color),
    ]
    SU3c = s["SU3c"]
    vvv = {(g, g, g): adjoint_vvv(SU3c)}
    vvv_colors = {(g, g, g): ADJOINT_VVV_COLOR}
    vvvv = {(g, g, g, g): adjoint_vvvv(SU3c)}
    vvvv_colors = {(g, g, g, g): dict(ADJOINT_VVVV_COLORS)}

    write_ufo(path, "FEYNLAG_QCD", ParameterSet(gs), particles,
              fermion_vertices=fermion_vertices,
              vvv=vvv, vvv_colors=vvv_colors,
              vvvv=vvvv, vvvv_colors=vvvv_colors)
    return path, model, s


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/FEYNLAG_QCD_UFO"
    path, model, s = export(Path(out))
    report = verify_ufo_numeric(path)
    status = "PASS" if report.ok else "FAIL"
    print(f"exported {path}")
    print(f"UFO round-trip: {status} "
          f"({len(report.parameters)} params, {len(report.couplings)} couplings)")
    for name, value in sorted(report.couplings.items()):
        print(f"  {name} = {value}")
    if not report.ok:
        for f in report.failures:
            print(f"  FAILURE: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

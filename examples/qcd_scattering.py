"""Parton-level QCD 2→2 from the Lagrangian — Tier 5, stage 5a.

Quark–quark and quark–antiquark scattering through gluon (and photon)
exchange, with the couplings extracted from a QCD × QED model and the colour
factors computed from the SU(3) generators — nothing like ``C_F = 4/3`` is
typed in.

A coloured species is declared the way the decay calculator already does it:
a ``DiracParticle`` on **one** colour component, with ``color=3``.  The
calculator turns the eight colour-component gluon vertices into one ``q̄ q g``
vertex with the colour-stripped coupling ``i g_s`` and builds each diagram's
colour tensor (``T^a ⊗ T^a`` for a gluon, ``δ ⊗ δ`` for a photon); the
colour-summed ``|M|²`` is ``Σ C_dd' L_d L̄_d'``.

Checked against the four-quark entries of the parton table of Ellis, Stirling
and Webber, *QCD and Collider Physics* (Cambridge University Press, 1996),
Table 7.1.  Parton level only: no PDFs, fixed ``α_s``.

Run:  python examples/qcd_scattering.py
"""

import math

import sympy as sp

from feynlag import (
    ExternalParameter, Lagrangian, Model, SU3, U1, WeylFermion,
    fermion_gauge_current,
)
from feynlag.pheno import DiracParticle, ScatteringCalculator

ALPHA_S, ALPHA = 0.118, 1 / 128.0
GS, EE = math.sqrt(4 * math.pi * ALPHA_S), math.sqrt(4 * math.pi * ALPHA)
GEV2_TO_PB = 0.3893794e9


def build():
    gs = ExternalParameter("gs", GS, positive=True)
    ee = ExternalParameter("ee", EE, positive=True)
    SU3c, U1em = SU3("SU3c", coupling=gs), U1("U1em", coupling=ee)

    def quark(name, charge):
        return [WeylFermion(f"{name}{chi}", reps={SU3c: 3, U1em: charge},
                            chirality=chi, nflavors=1,
                            component_names=[f"{name}{chi}_{k}" for k in (1, 2, 3)])
                for chi in ("L", "R")]

    uL, uR = quark("u", sp.Rational(2, 3))
    dL, dR = quark("d", -sp.Rational(1, 3))
    eL = WeylFermion("eL", reps={U1em: -1}, chirality="L", nflavors=1,
                     component_names=["eL"])
    eR = WeylFermion("eR", reps={U1em: -1}, chirality="R", nflavors=1,
                     component_names=["eR"])
    G, A = SU3c.bosons("G"), U1em.bosons("A")

    i = sp.Symbol("i", integer=True)
    L = Lagrangian()
    for f in (uL, uR, dL, dR, eL, eR):
        L.add(fermion_gauge_current(f, i), sector="gauge")
    model = Model("QCDxQED", gauge_groups=[SU3c, U1em],
                  fields=[uL, uR, dL, dR, eL, eR, G, A],
                  parameters=[gs, ee], lagrangian=L)
    model.check_invariance().raise_on_failure()

    # one colour component each, colour multiplicity declared once
    u = DiracParticle("u", uL.components[0], uR.components[0], 0, color=3)
    d = DiracParticle("d", dL.components[0], dR.components[0], 0, color=3)
    e = DiracParticle("e", eL.components[0], eR.components[0], 0)
    photon = A.components[0]
    calc = ScatteringCalculator(
        model, [u, d, e], masses={photon: 0}, boson_fields=[photon],
        colour_group=SU3c, gluon=G)
    return calc, u, d, e, gs.s, ee.s


def main():
    calc, u, d, e, gs, ee = build()

    print("=" * 70)
    print("Quark vertices (the gluon is one mediator, colour stripped)")
    print("=" * 70)
    for vertex in calc.vertices():
        if vertex.particles[0] in (u.antiparticle, d.antiparticle):
            print("  ", vertex)

    print("\n" + "=" * 70)
    print("Averaged |M|^2 / g_s^4, photon off  (ESW Table 7.1)")
    print("=" * 70)
    processes = [
        ("q q'    -> q q'   ", (u.particle, d.particle), (u.particle, d.particle)),
        ("q qbar  -> q' qbar'", (u.particle, u.antiparticle), (d.particle, d.antiparticle)),
        ("q q     -> q q    ", (u.particle, u.particle), (u.particle, u.particle)),
        ("q qbar  -> q qbar ", (u.particle, u.antiparticle), (u.particle, u.antiparticle)),
    ]
    for label, incoming, outgoing in processes:
        proc = calc.process(incoming, outgoing)
        kin = proc.kinematics
        s, t, U = sp.symbols("s t u")
        averaged = sp.simplify(proc.squared().subs(ee, 0) / 36 / gs**4)
        # show it in terms of s, t, u (u is derived: u = -s - t)
        shown = sp.factor(averaged.subs({kin.s: s, kin.t: t}))
        print(f"  {label}:  {shown}     [u = -s - t]")

    print("\n" + "=" * 70)
    print("Colour matrix of u ubar -> u ubar with photon and gluon")
    print("=" * 70)
    proc = calc.process((u.particle, u.antiparticle), (u.particle, u.antiparticle))
    labels = [f"{info.channel}:{info.mediator}" for info in proc.diagrams]
    print("        " + "".join(f"{name:>8}" for name in labels))
    for name, row in zip(labels, proc.colour):
        print(f"  {name:>6}" + "".join(f"{str(x):>8}" for x in row))
    print("  (singlet x octet in the s channel is 0: Tr T^a = 0)")

    print("\n" + "=" * 70)
    print("Cross sections at sqrt(s) = 100 GeV, |cos theta| < 0.9")
    print("=" * 70)
    values = {gs: GS, ee: EE}
    cut = (-0.9, 0.9)
    for label, incoming, outgoing in processes:
        proc = calc.process(incoming, outgoing, numeric=True, extra=values)
        sigma = proc.numeric_cross_section(100.0**2, cos_range=cut) * GEV2_TO_PB
        print(f"  {label}:  {sigma:12.2f} pb")
    ratio = calc.process((e.particle, e.antiparticle),
                         (u.particle, u.antiparticle))
    s = ratio.kinematics.s
    point = 4 * sp.pi * (ee**2 / (4 * sp.pi))**2 / (3 * s)
    print(f"\n  sigma(e+e- -> u ubar) / (4 pi alpha^2 / 3s) = "
          f"{sp.simplify(ratio.cross_section() / point)}   [N_c Q_u^2 = 4/3]")


if __name__ == "__main__":
    main()

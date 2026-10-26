"""Validate feynlag's QCD sector in MadGraph: ``check u u~ > g g``.

feynlag's QCD sector has never been exercised in any process.  The
cross-section round-trip (``madgraph_roundtrip.py``) covers e+e- -> mu+mu- and
e+e- -> W+W-, both QCD-free, and three separate colour/normalisation defects
were fixed in the QCD vertices recently, each caught by reading MadGraph's
shipped model files rather than by running anything.

``u u~ -> g g`` is the right probe.  Its diagrams are t- and u-channel quark
exchange (no ggg) interfering with the s-channel gluon (one ggg), so the
amplitude is **linear** in the triple-gluon coupling and its sign is
observable.  ``g g -> g g`` would not do: it goes as ggg^2 and is sign-blind.

MadGraph's ``check`` evaluates the matrix element at random phase-space points
and runs model-internal consistency tests (Lorentz invariance, the gauge/BRS
Ward identity, leg-permutation symmetry) with no beams and no Fortran
cross-section run.

A pass alone would only mean "nothing crashed", so this also runs a **negative
control**: the same check on a UFO exported with the ggg sign deliberately
flipped, which must FAIL.  Without that, the result says nothing.  (The same
control is what made the gamma-gamma -> W+W- quartic result decisive; see
docs/benchmark.md.)

Reuses the MadGraph plumbing (download/locate, command runner) from
``madgraph_roundtrip.py``.  Not in CI — each launch is slow.

Run:  python scripts/madgraph_qcd.py
"""

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from madgraph_roundtrip import ensure_mg5, run_mg5        # noqa: E402

PROCESS = "u u~ > g g"

#: each `check` section prints a header line, a column header, one or more
#: rows, then a summary. FAILING rows are followed by INDENTED "JAMP n" lines,
#: so the row block must accept leading whitespace — requiring non-whitespace
#: made every failing section silently vanish from the report, which is how a
#: hard failure first looked like a missing section here.
_SECTION = re.compile(
    r"^(?P<name>[A-Z][^\n:]*results[^\n:]*):\s*\n"
    r"^Process\s+.*\n"
    r"(?P<rows>(?:^[^\n]*\n)+?)"
    r"^Summary:\s*(?P<passed>\d+)/(?P<total>\d+) passed",
    re.M)


def parse_check(log):
    """``{section name: (rows, passed, total)}`` from a ``check`` log."""
    out = {}
    for m in _SECTION.finditer(log):
        rows = [r for r in m.group("rows").splitlines() if r.strip()]
        out[m.group("name").strip()] = (rows, int(m.group("passed")),
                                        int(m.group("total")))
    return out


def run_check(exe, model, work, tag):
    """Run ``check`` on one model; return (sections, log)."""
    commands = f"set automatic_html_opening False\nimport model {model}\ncheck {PROCESS}\n"
    log = run_mg5(exe, commands)
    (work / f"mg_{tag}.log").write_text(log)
    return parse_check(log), log


def report(tag, sections):
    if not sections:
        print(f"  {tag}: no check sections parsed (see log)")
        return None
    all_passed = True
    for name, (rows, passed, total) in sections.items():
        verdict = "passed" if passed == total else "FAILED"
        all_passed &= (passed == total)
        print(f"  {tag:22s} {name:34s} {passed}/{total} {verdict}")
        for row in rows:
            print(f"      {row.strip()}")
    return all_passed


def export_flipped(path):
    """Export the QCD UFO with the ggg sign deliberately wrong."""
    import feynlag.export.ufo.vvvv as V
    original = V.adjoint_vvv
    V.adjoint_vvv = lambda group: -original(group)
    try:
        import export_qcd_ufo
        # the module imported adjoint_vvv by name, so patch it there too
        saved = export_qcd_ufo.adjoint_vvv
        export_qcd_ufo.adjoint_vvv = V.adjoint_vvv
        try:
            export_qcd_ufo.export(path)
        finally:
            export_qcd_ufo.adjoint_vvv = saved
    finally:
        V.adjoint_vvv = original
    return path


def main():
    exe = ensure_mg5()
    work = Path(tempfile.mkdtemp(prefix="mg_qcd_"))
    print(f"work dir: {work}")

    import export_qcd_ufo
    good = work / "FEYNLAG_QCD"
    export_qcd_ufo.export(good)
    bad = work / "FEYNLAG_QCD_FLIPPED"
    export_flipped(bad)

    print(f"\ncheck {PROCESS}\n")
    results = {}
    for tag, model in (("stock_sm", "sm"),
                       ("feynlag", str(good)),
                       ("feynlag_ggg_flipped", str(bad))):
        sections, log = run_check(exe, model, work, tag)
        results[tag] = report(tag, sections)
        print()

    ok = results.get("feynlag") is True
    control_fails = results.get("feynlag_ggg_flipped") is False
    print(f"feynlag passes:        {results.get('feynlag')}")
    print(f"flipped-ggg control:   {results.get('feynlag_ggg_flipped')} "
          f"(must be False for the check to have teeth)")
    if ok and control_fails:
        print("\nVERDICT: QCD sector validated — the check passes and the "
              "deliberately broken model fails it.")
        return 0
    if ok and not control_fails:
        print("\nVERDICT: INCONCLUSIVE — feynlag passes but so does the "
              "broken control, so this process/check is not sensitive to the "
              "ggg sign as run.")
        return 1
    print("\nVERDICT: FAILED — investigate before changing any sign; the "
          "point of this run is that it is an independent oracle.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

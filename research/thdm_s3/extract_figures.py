"""Extract the stored figures of the research notebooks to figures/*.png.

The notebooks are tracked with ``nbstripout --keep-output``, so their figures are
real committed bytes rather than something a re-run has to reproduce.  This pulls
them out so the ``report_*.tex`` write-ups can \\includegraphics them without anyone
re-executing a scan.

The mapping below is by *cell index*, which is brittle if the notebook is
re-organised — so each entry also carries a marker string that must appear in the
producing cell's source.  A shifted notebook fails loudly here instead of
silently putting the wrong plot in the report.

Run from research/thdm_s3/:  python extract_figures.py
"""

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "figures"

NB01 = "01_scalar_parameter_space.ipynb"     # report_scalar_sector.tex
NB02 = "02_s3_fermion_sector.ipynb"          # report_fermion_sector.tex
NB03 = "03_scalar_decays.ipynb"              # report_scalar_decays.tex
NB05 = "05_soft_scalar_scan.ipynb"           # report_soft_vacuum.tex

#: name -> (notebook, cell index, marker that must occur in that cell's source)
FIGURES = {
    "bfb_ft_curve": (NB01, 62, "Eq. (4g) checks one point of a curve"),
    "cut_flow": (NB01, 75, "Cut flow: S$_3$-3HDM scalar parameter space"),
    "mass_ranges": (NB01, 79, "Theory-allowed mass ranges"),
    "correlations": (NB01, 80, "lightest CP-even scalar"),
    "fermion_vus_vs_r": (NB02, 58, "Soft $S_3$ breaking turns on the Cabibbo angle"),
    "decays_delta_scenarios": (NB03, 14, "between scenarios A and B"),
    "decays_gammagamma": (NB03, 26, "Charged-Higgs loop in"),
    "decays_lfv_limits": (NB03, 33, "LFV rates at the exact-S"),
    "soft_cut_flow": (NB05, 22, "Cut flow: exact vs soft-broken"),
    "soft_masses": (NB05, 24, "Scalar masses: exact $S_3$"),
    "soft_hvv": (NB05, 26, "VV coupling of the non-SM-like CP-even states"),
    "soft_breaking": (NB05, 28, "free-soft prior edge"),
    "soft_mD1_edge": (NB05, 30, "The tadpole-fixed $m_{D1}^2$ across the domain"),
}


def main():
    OUTDIR.mkdir(exist_ok=True)
    notebooks = {}

    for name, (notebook, idx, marker) in FIGURES.items():
        if notebook not in notebooks:
            notebooks[notebook] = json.loads((HERE / notebook).read_text())["cells"]
        cell = notebooks[notebook][idx]
        src = "".join(cell["source"])
        if marker not in src:
            raise SystemExit(
                f"{notebook} cell {idx} does not contain {marker!r} — the notebook has been "
                f"re-organised and FIGURES needs updating (refusing to write "
                f"{name}.png from the wrong cell)")
        pngs = [o["data"]["image/png"] for o in cell.get("outputs", [])
                if o["output_type"] == "display_data" and "image/png" in o.get("data", {})]
        if len(pngs) != 1:
            raise SystemExit(f"cell {idx} has {len(pngs)} PNG outputs, expected 1")
        raw = base64.b64decode(pngs[0])
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise SystemExit(f"cell {idx} output is not a PNG")
        (OUTDIR / f"{name}.png").write_bytes(raw)
        print("  %-28s <- %s cell %-3d  %6.1f KB"
              % (name + ".png", notebook[:2], idx, len(raw) / 1024))

    print("wrote %d figures to %s/" % (len(FIGURES), OUTDIR.name))


if __name__ == "__main__":
    main()

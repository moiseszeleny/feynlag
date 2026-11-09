"""Shared set-up for the 3HDM-S₃ research notebooks.

Every notebook used to open with the same 40-60 lines: imports, a silenced
matplotlib logger, plot style, SymPy printing, colours and a `check_quoted`
helper.  They live here once, so a notebook's first cell can be

    from nbsetup import *

followed only by the imports that notebook actually needs.  Nothing here
computes physics; it only fixes how things are imported, printed and plotted.

Run from research/thdm_s3/ (the notebooks' working directory).
"""

import json, logging, math, sys, time               # noqa: F401  (re-exported)
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib

# A broken personal matplotlib style file is noisy on import; silence it.
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt                     # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import derive as dv                                 # noqa: E402  the by-hand derivation vocabulary
import store                                        # noqa: E402  data in cache/, summaries in results/

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
sp.init_printing(use_latex="mathjax")               # typeset maths, not ASCII art

ok = dv.ok                                          # the "✓ ..." pass marker
C_BLUE, C_RED, C_GREEN = "#4C72B0", "#C44E52", "#55A868"   # plot colours


def check_quoted(section, rows):
    """Assert that the numbers the markdown and the reports quote still match this run.

    rows: (what, computed, quoted, tol), tol being half a unit of the quoted last digit.
    """
    stale = [(what, got, want) for what, got, want, tol in rows
             if abs(float(got) - float(want)) > tol]
    assert not stale, "§%s prose is stale (what, computed, quoted): %s" % (section, stale)
    ok("the %d numbers quoted in §%s match this run" % (len(rows), section))


def table(headers, rows, fmt=None):
    """Show `rows` as a rendered Markdown table.

    fmt: one format spec per column (e.g. ``".4f"``), or None for ``str``.
    Cells that are already strings are shown as they are.  Outside a notebook
    it prints a plain aligned table instead.
    """
    fmt = fmt or [None] * len(headers)

    def cell(x, f):
        if isinstance(x, str) or f is None:
            return str(x)
        return format(x, f)

    body = [[cell(x, f) for x, f in zip(r, fmt)] for r in rows]
    if dv._in_notebook():
        from IPython.display import Markdown, display
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        display(Markdown("\n".join(lines)))
    else:
        widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
                  for i, h in enumerate(headers)]
        for r in [list(headers)] + body:
            print("  ".join(s.ljust(w) for s, w in zip(r, widths)))


__all__ = ["json", "logging", "math", "sys", "time", "Path", "np", "sp", "matplotlib",
           "plt", "dv", "store", "ok", "C_BLUE", "C_RED", "C_GREEN",
           "check_quoted", "table"]

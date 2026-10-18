"""The project's data flow: which notebook or script reads and writes which file.

`provenance.py` uses this map to decide what goes stale when something changes,
and `claims.py` uses it to mark what is downstream of an assumption. Paths are
relative to this folder.

What does **not** need listing here, because `provenance.py` finds it itself:

- the local modules a node imports (``model.py``, ``lfv.py``, ...), read from
  its import statements, transitively;
- the feynlag library, fingerprinted as a whole.

Update this map when a notebook starts reading or writing a results file, and
add a ``REPORTS`` entry for every new report.  `run_pipeline.py` runs the nodes
in the order listed here, so keep a node after everything it consumes.
"""

#: node -> files it reads ("consumes") and writes ("produces"), in run order.
#: A node is a notebook (its two-digit prefix) or a script (its file name).
#:
#: Per-point data lives in the gitignored ``cache/`` (see `store.py`); a node
#: that reads it lists the committed **summary** ``results/<name>.json``, whose
#: ``points_sha256`` pins the cache file, so a fingerprint is computable on a
#: fresh clone.  The ``cache/*.pkl`` files are the compute scripts' raw loop
#: outputs, listed directly: provenance reports them as "cache missing" rather
#: than stale when they are absent.
PIPELINE = {
    "scan.py": {"consumes": [], "produces": ["results/viable_points.json"]},
    "01": {"consumes": ["results/viable_points.json"],
           "produces": ["derivations_01.tex"]},
    "02": {"consumes": [],
           "produces": ["results/quark_soft_fit.json", "derivations_02.tex"]},
    "03": {"consumes": ["results/viable_points.json"],
           "produces": ["results/decay_benchmarks.json", "derivations_03.tex"]},
    "04": {"consumes": [], "produces": ["derivations_04.tex"]},
    "scan_soft.py": {"consumes": [], "produces": ["results/viable_points_soft.json"]},
    "05": {"consumes": ["results/viable_points.json", "results/viable_points_soft.json",
                        "results/quark_soft_fit.json"],
           "produces": ["derivations_05.tex"]},
    "compute_soft_decays.py": {"consumes": ["results/viable_points_soft.json"],
                               "produces": ["cache/soft_decays_scan.pkl"]},
    "06": {"consumes": ["results/viable_points.json", "results/viable_points_soft.json",
                        "results/decay_benchmarks.json", "cache/soft_decays_scan.pkl"],
           "produces": ["results/soft_decay_benchmarks.json", "derivations_06.tex"]},
    "compute_mu_e_gamma.py": {"consumes": ["results/viable_points_soft.json"],
                              "produces": ["cache/mu_e_gamma_scan.pkl"]},
    "07": {"consumes": ["results/viable_points.json", "results/viable_points_soft.json",
                        "cache/mu_e_gamma_scan.pkl"],
           "produces": ["results/mu_e_gamma.json", "derivations_07.tex"]},
    "compute_tau_l_gamma.py": {"consumes": ["results/viable_points.json",
                                            "results/viable_points_soft.json"],
                               "produces": ["cache/tau_l_gamma_scan.pkl"]},
    "08": {"consumes": ["results/viable_points.json", "results/viable_points_soft.json",
                        "results/mu_e_gamma.json", "cache/tau_l_gamma_scan.pkl"],
           "produces": ["results/tau_l_gamma.json", "derivations_08.tex"]},
    "benchmarks.py": {"consumes": ["results/viable_points_soft.json",
                                   "results/soft_decay_benchmarks.json",
                                   "results/mu_e_gamma.json", "results/tau_l_gamma.json"],
                      "produces": ["results/benchmarks.json"]},
}

#: report source -> the notebooks whose stored outputs (numbers, figures) it quotes.
REPORTS = {
    "report_scalar_sector.tex": ["01"],
    "report_fermion_sector.tex": ["02"],
    "report_scalar_decays.tex": ["03"],
    "report_soft_vacuum.tex": ["05"],
    "report_soft_decays.tex": ["06"],
    "report_mu_e_gamma.tex": ["07"],
    "report_tau_l_gamma.tex": ["08"],
}

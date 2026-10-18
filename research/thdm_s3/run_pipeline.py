"""Regenerate ``cache/`` and every out-of-date result, in dependency order.

The project's per-point data is not committed (see `store.py`): a fresh clone
has the summaries in ``results/`` but an empty ``cache/``, and a notebook that
reads per-point data cannot run until the cache is rebuilt.  This script walks
`pipeline.PIPELINE` in order and runs a node when

- `provenance.py` does not report it ``ok`` (its code, a module or an input
  changed since it was stamped, or it was never stamped);
- a cache file it produces is missing, including the ``cache/<name>.json`` that
  one of its ``results/<name>.json`` summaries points to;
- a node it consumes from is being run in this invocation.

Scripts run with this folder's Python; notebooks with ``jupyter nbconvert
--execute --inplace`` under the ``--kernel`` kernel.  Each node is stamped in
``results/MANIFEST.json`` after it succeeds (reports are not: re-reading them
is a person's job, ``provenance.py stamp`` with the notebook does it).

At the end every regenerated cache file is compared with the ``points_sha256``
its committed summary records: "reproduced" means the per-point data is the one
the committed results were computed from, byte for byte.

Usage (from anywhere)::

    python research/thdm_s3/run_pipeline.py              # what is missing or stale
    python research/thdm_s3/run_pipeline.py --dry-run    # only list it
    python research/thdm_s3/run_pipeline.py --all        # everything, from scratch
    python research/thdm_s3/run_pipeline.py --only 07 08 # these nodes, nothing else

A full run from an empty ``cache/`` takes about an hour (``README.md`` records
the last measured time); the scans and the compute scripts are most of it.

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import provenance as prov  # noqa: E402
import store  # noqa: E402

PIPELINE = prov.PIPELINE


def _cache_targets(node):
    """Every cache file a node is responsible for (pickles and summaries' points)."""
    out = []
    for f in PIPELINE[node]["produces"]:
        if f.startswith("cache/"):
            out.append(f)
        elif f.startswith("results/") and f.endswith(".json") and (HERE / f).exists():
            ref = json.loads((HERE / f).read_text()).get("points_cache")
            if ref:
                out.append(ref)
    return out


def _status():
    """`provenance.check`'s per-node verdict, without its printout."""
    man = prov.load_manifest()
    status = {}
    for node in PIPELINE:
        rec = man["nodes"].get(node)
        if rec is None:
            status[node] = "unstamped"
            continue
        changed = prov._diff(rec["fingerprint"], prov.fingerprint(node))
        status[node] = "stale" if changed else "ok"
    return status


def plan(force_all=False, only=None):
    """The nodes to run, in pipeline order, each with the reason."""
    status = _status()
    producer = prov._producer()
    todo = {}
    for node, io_ in PIPELINE.items():
        if only is not None:
            if node in only:
                todo[node] = "requested"
            continue
        if force_all:
            todo[node] = "--all"
        elif status[node] != "ok":
            todo[node] = status[node]
        elif any(not (HERE / f).exists() for f in _cache_targets(node)):
            todo[node] = "cache missing"
        else:
            up = [producer[f] for f in io_["consumes"] if producer.get(f) in todo]
            if up:
                todo[node] = "upstream re-run: " + ", ".join(sorted(set(up)))
    return todo


def run_node(node, kernel):
    path = prov.node_file(node)
    if path.suffix == ".py":
        cmd = [sys.executable, path.name]
    else:
        cmd = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               f"--ExecutePreprocessor.kernel_name={kernel}",
               "--ExecutePreprocessor.timeout=-1", path.name]
    t0 = time.time()
    subprocess.run(cmd, cwd=HERE, check=True)
    return time.time() - t0


def compare_hashes():
    """Each summary's recorded ``points_sha256`` against its cache file now."""
    rows = []
    for f in sorted((HERE / "results").glob("*.json")):
        summ = json.loads(f.read_text())
        if "points_sha256" not in summ:
            continue
        cache = HERE / summ["points_cache"]
        committed = subprocess.run(
            ["git", "show", f"HEAD:{f.relative_to(prov.REPO).as_posix()}"],
            cwd=HERE, capture_output=True, text=True).stdout
        want = json.loads(committed).get("points_sha256") if committed else None
        got = store.sha256(cache) if cache.exists() else "missing"
        verdict = ("reproduced" if got == want else
                   "no committed hash" if want is None else
                   "cache missing" if got == "missing" else "DIFFERS from the committed hash")
        rows.append((summ["points_cache"], verdict))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="run every node")
    ap.add_argument("--only", nargs="+", metavar="NODE", help="run just these nodes")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, and why")
    ap.add_argument("--kernel", default="lagrangian",
                    help="Jupyter kernel for the notebooks (default: lagrangian)")
    args = ap.parse_args()
    if args.only:
        unknown = [n for n in args.only if n not in PIPELINE]
        if unknown:
            sys.exit(f"unknown node(s) {unknown}; choose from {list(PIPELINE)}")

    todo = plan(args.all, set(args.only) if args.only else None)
    if not todo:
        print("nothing to run: every node is current and cache/ is complete")
    for node, why in todo.items():
        print(f"{'would run' if args.dry_run else 'run'} {node:<24} ({why})")
    if args.dry_run or not todo:
        return

    t_all = time.time()
    timings = {}
    for node in todo:
        print(f"\n=== {node}", flush=True)
        timings[node] = run_node(node, args.kernel)
        with contextlib.redirect_stdout(io.StringIO()):
            prov.stamp([node], note="run_pipeline.py", reports=False)
        print(f"=== {node} done in {timings[node] / 60:.1f} min", flush=True)

    print(f"\nran {len(todo)} node(s) in {(time.time() - t_all) / 60:.1f} min:")
    for node, t in timings.items():
        print(f"  {node:<24} {t / 60:5.1f} min")
    print("\ncache files against the committed summaries' hashes:")
    for path, verdict in compare_hashes():
        print(f"  {path:<34} {verdict}")


if __name__ == "__main__":
    main()

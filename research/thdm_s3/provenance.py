"""Which results, notebooks and reports are out of date?

Each node of `pipeline.PIPELINE` (a notebook or a scan script) gets a
**fingerprint** with four parts:

- ``code``: its own source. For a notebook this is the code cells only, so
  re-executing it without editing it does not change the fingerprint;
- ``modules``: every local module it imports, transitively (``model.py``,
  ``lfv.py``, ``sm_inputs.py`` ...);
- ``inputs``: the results files it consumes;
- ``library``: the feynlag sources under ``src/feynlag``.

``results/MANIFEST.json`` records the fingerprint, plus the hash of each output,
at the moment a node was last run. ``check`` compares that record against now:

- **stale**: the code, a module or an input changed since the node last ran.
  Re-execute it, then ``stamp`` it.
- **upstream stale**: something it consumes is produced by a stale node.
- **output edited**: an output no longer matches what the node wrote.
- **library changed**: feynlag changed since the node ran. This is a warning,
  not staleness: most library changes do not touch these results.

Reports (`pipeline.REPORTS`) quote stored notebook outputs, so a report is
flagged when a notebook it quotes has stale results, or has been re-run since
the report was last checked against it.

Per-point data lives in the gitignored ``cache/`` (see `store.py`).  A missing
cache file is not staleness: a fresh clone has none.  ``check`` lists it as
"cache missing" with the hint to run ``run_pipeline.py``, and does not count it
as needing attention.  ``check`` also flags any ``results/*.json`` over
`MAX_RESULT_KB`, the size `research/README.md` asks committed results to stay
under.

Usage (from anywhere)::

    python research/thdm_s3/provenance.py check            # the default
    python research/thdm_s3/provenance.py stamp 06 07      # after re-running them
    python research/thdm_s3/provenance.py stamp --all      # declare everything current
    python research/thdm_s3/provenance.py audit            # what git history suggests

``check`` exits with status 1 when anything is stale, so it can gate a script.

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "results" / "MANIFEST.json"
#: committed results files should stay under this (research/README.md)
MAX_RESULT_KB = 100


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("_pipeline", HERE / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PIPELINE, mod.REPORTS


PIPELINE, REPORTS = _load_pipeline()


# ---------------------------------------------------------------- hashing

def _sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _file_hash(path):
    path = HERE / path
    return _sha(path.read_bytes()) if path.exists() else "missing"


def node_file(node):
    """The source file of a node: ``"06"`` → ``06_soft_decays.ipynb``."""
    if node.endswith(".py"):
        return HERE / node
    matches = sorted(HERE.glob(f"{node}_*.ipynb"))
    if len(matches) != 1:
        raise FileNotFoundError(f"node {node!r}: expected one {node}_*.ipynb, "
                                f"found {len(matches)}")
    return matches[0]


def _source(path):
    """The code a node runs: a notebook's code cells, or a script's text."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        return "\n".join("".join(c["source"]) for c in cells
                         if c["cell_type"] == "code")
    return path.read_text()


_IMPORT = re.compile(r"^\s*(?:from\s+(\w+)\s+import|import\s+([\w ,]+))", re.M)


def local_imports(path, seen=None):
    """Local modules ``path`` imports, followed transitively."""
    local = {p.stem for p in HERE.glob("*.py")}
    seen = set() if seen is None else seen
    for m in _IMPORT.finditer(_source(path)):
        names = [m.group(1)] if m.group(1) else [
            n.strip().split()[0] for n in m.group(2).split(",") if n.strip()]
        for name in names:
            if name in local and name not in seen and name != path.stem:
                seen.add(name)
                local_imports(HERE / f"{name}.py", seen)
    return seen


def _library_hash():
    files = sorted((REPO / "src" / "feynlag").rglob("*.py"))
    return _sha(b"".join(_sha(f.read_bytes()).encode() for f in files))


def fingerprint(node):
    path = node_file(node)
    io = PIPELINE[node]
    return {
        "code": _sha(_source(path)),
        "modules": {m: _file_hash(f"{m}.py") for m in sorted(local_imports(path))},
        "inputs": {f: _file_hash(f) for f in io["consumes"]},
        "library": _library_hash(),
    }


def outputs(node):
    return {f: _file_hash(f) for f in PIPELINE[node]["produces"]}


# ---------------------------------------------------------------- manifest

def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "nodes": {}, "reports": {}}


def stamp(nodes, note="", reports=True):
    """Record ``nodes`` as current.  With ``reports``, also mark every report that
    quotes one of them as checked against it (`run_pipeline.py` passes False:
    re-reading a report is a person's job)."""
    man = load_manifest()
    when = _dt.date.today().isoformat()
    for node in nodes:
        man["nodes"][node] = {"fingerprint": fingerprint(node), "outputs": outputs(node),
                              "stamped": when, "commit": _git_head(), "note": note}
    # a report is "checked" against the notebooks it quotes as they stand now
    for rep, nbs in REPORTS.items():
        if reports and any(nb in nodes for nb in nbs):
            man["reports"][rep] = {"tex": _file_hash(rep),
                                   "notebooks": {nb: _sha(node_file(nb).read_bytes())
                                                 for nb in nbs},
                                   "stamped": when}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"stamped {', '.join(nodes)} in {MANIFEST.relative_to(REPO)}")


# ---------------------------------------------------------------- check

def _is_cache(path):
    return path.startswith("cache/")


def _missing_cache(node):
    """The cache files a node consumes or produces that are not on disk."""
    io = PIPELINE[node]
    return [f for f in io["consumes"] + io["produces"]
            if _is_cache(f) and not (HERE / f).exists()]


def _diff(old, new):
    """Human-readable list of what changed between two fingerprints.

    An input that is a missing cache file is not a change: see `_missing_cache`.
    """
    changes = []
    if old["code"] != new["code"]:
        changes.append("its code")
    for kind in ("modules", "inputs"):
        for k in sorted(set(old[kind]) | set(new[kind])):
            if kind == "inputs" and _is_cache(k) and new[kind].get(k) == "missing":
                continue
            if old[kind].get(k) != new[kind].get(k):
                changes.append(f"{k}{'.py' if kind == 'modules' else ''}")
    return changes


def oversized():
    """Committed-size violations: ``results/*.json`` over `MAX_RESULT_KB`."""
    return [(f.relative_to(HERE).as_posix(), f.stat().st_size / 1024)
            for f in sorted((HERE / "results").glob("*.json"))
            if f.stat().st_size > MAX_RESULT_KB * 1024]


def _producer():
    return {f: n for n, io in PIPELINE.items() for f in io["produces"]}


def check():
    man = load_manifest()
    status, notes = {}, {}
    for node in PIPELINE:
        rec = man["nodes"].get(node)
        if rec is None:
            status[node], notes[node] = "unstamped", "never stamped"
            continue
        now = fingerprint(node)
        changed = _diff(rec["fingerprint"], now)
        edited = [f for f, h in outputs(node).items() if rec["outputs"].get(f) != h
                  and not (_is_cache(f) and h == "missing")]
        if changed:
            status[node], notes[node] = "stale", "changed: " + ", ".join(changed)
        elif edited:
            status[node], notes[node] = "output edited", ", ".join(edited)
        else:
            status[node], notes[node] = "ok", ""
        missing = _missing_cache(node)
        if missing:
            notes[node] = (notes[node] + "; " if notes[node] else "") + \
                "cache missing: " + ", ".join(missing)
        if rec["fingerprint"]["library"] != now["library"]:
            notes[node] = (notes[node] + "; " if notes[node] else "") + \
                "warning: feynlag changed since this run"

    # propagate: anything consuming a file whose producer is not ok
    producer = _producer()
    changed_flag = True
    while changed_flag:
        changed_flag = False
        for node, io in PIPELINE.items():
            if status[node] != "ok":
                continue
            bad = [f for f in io["consumes"]
                   if status.get(producer.get(f), "ok") != "ok"]
            if bad:
                status[node] = "upstream stale"
                notes[node] = "via " + ", ".join(
                    f"{f} ({producer[f]})" for f in bad)
                changed_flag = True

    print(f"{'node':<14}{'status':<17}detail")
    for node in PIPELINE:
        print(f"{node:<14}{status[node]:<17}{notes[node]}")

    print(f"\n{'report':<28}{'status':<17}detail")
    rep_bad = 0
    for rep, nbs in REPORTS.items():
        rec = man["reports"].get(rep)
        why = [f"nb{nb} {status[nb]}" for nb in nbs if status[nb] != "ok"]
        if rec is None:
            why.append("never stamped")
        else:
            why += [f"nb{nb} re-run since report was checked" for nb in nbs
                    if rec["notebooks"].get(nb) != _sha(node_file(nb).read_bytes())]
        rep_bad += bool(why)
        print(f"{rep:<28}{'re-check' if why else 'ok':<17}{'; '.join(why)}")

    big = oversized()
    for f, kb in big:
        print(f"\n{f} is {kb:.0f} kB, over the {MAX_RESULT_KB} kB committed-results limit: "
              "move its per-point data to cache/ (store.write)")
    if any(_missing_cache(n) for n in PIPELINE):
        print("\ncache/ is incomplete (normal on a fresh clone): "
              "regenerate it with `python research/thdm_s3/run_pipeline.py`")

    bad = sum(s != "ok" for s in status.values()) + rep_bad + len(big)
    print(f"\n{bad} item(s) need attention" if bad else "\neverything current")
    return 1 if bad else 0


# ---------------------------------------------------------------- audit

def _git(*args):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                          text=True).stdout.strip()


def _code_at(commit, path):
    """A node's code (code cells only, for a notebook) as of ``commit``."""
    rel = path.relative_to(REPO).as_posix()
    blob = _git("show", f"{commit}:{rel}")
    if path.suffix != ".ipynb" or not blob:
        return blob
    return "\n".join("".join(c["source"]) for c in json.loads(blob)["cells"]
                     if c["cell_type"] == "code")


def audit():
    """Git-history check: did a dependency change after the outputs were written?

    For each node this finds the commit that last wrote its outputs and asks
    whether its code (code cells only), its local modules or its inputs differ
    between that commit and the working tree. A changed module is not proof that
    the results moved (the change may be a docstring), so the verdict is
    "suspect", with the diff size, and the person decides. This is meant for
    bootstrapping the manifest and for choosing what to re-run.
    """
    print(f"{'node':<14}{'outputs written':<20}verdict    changed since (lines +/-)")
    for node, io in PIPELINE.items():
        path = node_file(node)
        outs = [f for f in io["produces"] if f.startswith("results/")] or io["produces"]
        commit = _git("log", "-1", "--format=%h", "--", *outs)
        if not commit:
            print(f"{node:<14}{'uncommitted':<20}unknown")
            continue
        when = _git("log", "-1", "--format=%cs", commit)
        changed = []
        if _code_at(commit, path) != _source(path):
            changed.append(f"{path.name} (code cells)")
        for dep in [f"{m}.py" for m in sorted(local_imports(path))] + io["consumes"]:
            stat = _git("diff", "--numstat", commit, "--", dep)
            if stat:
                add, rem = stat.split()[:2]
                changed.append(f"{dep} (+{add}/-{rem})")
        verdict = "suspect" if changed else "ok"
        print(f"{node:<14}{when + ' ' + commit:<20}{verdict:<11}{'; '.join(changed)}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["check"]
    cmd, rest = args[0], args[1:]
    if cmd == "check":
        sys.exit(check())
    elif cmd == "stamp":
        note = ""
        if "--note" in rest:
            i = rest.index("--note")
            note, rest = rest[i + 1], rest[:i] + rest[i + 2:]
        nodes = list(PIPELINE) if rest == ["--all"] else rest
        unknown = [n for n in nodes if n not in PIPELINE]
        if unknown or not nodes:
            sys.exit(f"stamp: give node names from {list(PIPELINE)} or --all")
        stamp(nodes, note)
    elif cmd == "audit":
        audit()
    else:
        sys.exit(__doc__)

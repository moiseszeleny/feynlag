"""Build CLAIMS.md: one review register for every claim the project makes.

Two sources:

1. the notebooks' derivation ledgers, read from the ``derivations_*.tex``
   files that `derive.Ledger.to_latex` emits. Any new notebook that writes one
   is picked up with no change here;
2. the hand-curated load-bearing assumptions in `assumptions.py`, which the
   ledgers do not record.

The register is ordered for manual review: assumptions first, then imported
results (check equation numbers, signs and factors of 2 against the source),
then results derived here (check the listed independent checks).

Review marks live in ``review_status.csv`` (columns ``id, status, note``, plus
two read-only columns for orientation), with status one of ``todo``, ``ok``,
``doubt`` or ``wrong``. Edit that file, never CLAIMS.md. A ledger claim's id
hashes its text, so inserting a step does not shift anyone's mark. When a
reviewed claim's text changes, its mark resets to ``todo`` with a note.

Usage (from anywhere)::

    python research/thdm_s3/claims.py [project_dir]

Research code: `tests/` must never import this (see `research/README.md`).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

STATUSES = ("todo", "ok", "doubt", "wrong")
KINDS = ("assumed", "imported", "derived")
CSV_FIELDS = ("id", "status", "note", "source", "claim")

# ----------------------------------------------------------------- ledger tex

_SUBSECTION = r"\subsection*{"


def _braced(s, start):
    """Return (content, end) of the brace group whose ``{`` is at ``start``."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{" and s[i - 1] != "\\":
            depth += 1
        elif s[i] == "}" and s[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    raise ValueError("unbalanced braces in ledger tex")


#: The reverse of `derive._tex_escape_text`, applied to prose only.
_UNESCAPE = (("\\textbackslash{}", "\\"), ("\\textasciitilde{}", "~"),
             ("\\textasciicircum{}", "^"), ("\\&", "&"), ("\\%", "%"),
             ("\\#", "#"), ("\\_", "_"), ("\\{", "{"), ("\\}", "}"),
             ("\\S{}", "§"), ("\\S", "§"), ("---", "—"), ("--", "–"))
_MATH_SPAN = re.compile(r"(\$[^$]*\$)")


def _md(tex):
    """Ledger tex → Markdown: prose unescaped, ``$...$`` maths kept verbatim."""
    pieces = _MATH_SPAN.split(tex)
    out = []
    for i, p in enumerate(pieces):
        if i % 2 == 0:
            for a, b in _UNESCAPE:
                p = p.replace(a, b)
        out.append(p)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _claim_id(notebook, claim):
    norm = re.sub(r"\s+", " ", claim).strip()
    return f"nb{notebook}-{hashlib.sha1(norm.encode()).hexdigest()[:8]}"


def parse_ledger_tex(path):
    """Read one ``derivations_NN.tex`` back into a list of step records."""
    path = Path(path)
    notebook = re.search(r"derivations_(\w+)\.tex$", path.name).group(1)
    tex = path.read_text()
    steps = []
    pos = tex.find(_SUBSECTION)
    while pos != -1:
        heading, end = _braced(tex, pos + len(_SUBSECTION) - 1)
        nxt = tex.find(_SUBSECTION, end)
        block = tex[end:nxt if nxt != -1 else len(tex)]
        m = re.match(r"\s*(\d+)\.\\quad\s*(.*)", heading, re.S)
        if not m:
            raise ValueError(f"{path.name}: unexpected heading {heading[:60]!r}")
        n, claim = int(m.group(1)), m.group(2)
        kind = re.search(r"\\emph\{(derived here|imported|assumed)\}", block)
        if not kind:
            raise ValueError(f"{path.name} step {n}: no \\emph{{kind}} tag")
        sec = re.search(r"\\emph\{[^}]*\}\s*---\s*\\S(.*?)\\par", block)
        statement = re.search(r"\\textbf\{Claim\}.*?\\begin\{equation\*\}\\small\s*"
                              r"(.*?)\s*\\end\{equation\*\}", block, re.S)
        checks = []
        items = re.search(r"\\begin\{itemize\}\\itemsep0pt(.*?)\\end\{itemize\}",
                          block, re.S)
        if items:
            checks = [_md(c) for c in items.group(1).split("\\item")[1:]]
        fals = re.search(r"\\textbf\{Would be falsified by:\}\s*(.*?)\\par", block, re.S)
        steps.append({
            "id": _claim_id(notebook, claim),
            "notebook": notebook,
            "n": n,
            "kind": kind.group(1).split()[0],
            "claim": _md(claim),
            "section": _md(sec.group(1)).replace("––", "–") if sec else "",
            "statement": statement.group(1).strip() if statement else "",
            "checks": checks,
            "falsified_by": _md(fals.group(1)) if fals else "",
        })
        pos = nxt
    count = tex.count(_SUBSECTION)
    if len(steps) != count:          # format drift in derive.py must fail loudly
        raise AssertionError(f"{path.name}: parsed {len(steps)} of {count} steps")
    return steps


# ------------------------------------------------------------------- pipeline

def downstream(pipeline, origins):
    """Every node reachable from ``origins`` through consumed results files."""
    origins = [origins] if isinstance(origins, str) else list(origins)
    seen, stack = set(origins), list(origins)
    while stack:
        produced = set(pipeline.get(stack.pop(), {}).get("produces", []))
        for node, io in pipeline.items():
            if node not in seen and produced & set(io.get("consumes", [])):
                seen.add(node)
                stack.append(node)
    return sorted(seen - set(origins), key=_node_key)


def _node_key(node):
    return (0, node) if node[:2].isdigit() else (1, node)


def _node_label(node):
    return f"nb{node}" if node[:2].isdigit() else node


# --------------------------------------------------------------- review marks

def load_marks(path):
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    for r in rows.values():
        if r["status"] not in STATUSES:
            raise ValueError(f"review_status.csv: {r['id']} has status "
                             f"{r['status']!r}, expected one of {STATUSES}")
    return rows


def reconcile(entries, marks):
    """Attach a mark to every entry; return (rows to write, orphaned marks)."""
    live = {e["id"] for e in entries}
    # an unreviewed mark carries no information, so only real marks are kept
    orphans = {k: m for k, m in marks.items() if k not in live
               and (m["status"] != "todo" or m["note"])}
    rows = []
    for e in entries:
        mark = marks.get(e["id"])
        if mark is None:
            mark = {"status": "todo", "note": ""}
            # same notebook step, new text: a reviewed claim was edited
            for oid, old in list(orphans.items()):
                if old.get("source") == e["source"] and old["status"] != "todo":
                    mark["note"] = (f"claim changed since review (was {old['status']}"
                                    + (f": {old['note']}" if old["note"] else "") + ")")
                    del orphans[oid]
                    break
        e["status"], e["note"] = mark["status"], mark["note"]
        rows.append({"id": e["id"], "status": e["status"], "note": e["note"],
                     "source": e["source"], "claim": e["claim"][:90]})
    return rows, orphans


# -------------------------------------------------------------------- render

_BADGE = {"todo": "☐ todo", "ok": "✅ ok", "doubt": "❓ doubt", "wrong": "❌ wrong"}


_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)")
_NESTED_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _gh_math(text):
    r"""Make prose render on GitHub: inline ``$x$`` becomes ``$`x`$``.

    GitHub only recognizes a bare ``$...$`` when nothing but whitespace touches
    its outer side, so ``exact-$S_3$``, ``S$_3$`` or ``$\mathbb{Z}_2$-preserving``
    stay raw text (and their ``_`` turn into italics).  The ``$`...`$`` form is
    parsed as a code span first and survives any neighbour.  A ``**x**`` inside a
    claim that `render` bolds as a whole becomes ``*x*``.
    """
    text = _NESTED_BOLD.sub(r"*\1*", str(text))
    return _INLINE_MATH.sub(lambda m: f"$`{m.group(1)}`$", text)


def _cell(s):
    return _gh_math(s).replace("|", "\\|").replace("\n", " ")


def render(entries, notebooks, orphans, ledgerless):
    out = ["<!-- Generated by claims.py: edit review_status.csv, not this file. -->",
           "# 3HDM-S₃: claims register", "",
           "Every claim the project makes, ordered for manual review. Regenerate with "
           "`python research/thdm_s3/claims.py`. Record review marks in "
           "[`review_status.csv`](review_status.csv) (`todo` / `ok` / `doubt` / `wrong`).",
           "", "**How to review:**",
           "1. **Assumptions** (§1). For each one, ask whether the conclusions in "
           "\"affects\" survive if it changes.",
           "2. **Imported** (§2). Check the equation number, sign conventions and factors "
           "of 2 against the cited source.",
           "3. **Derived here** (§3). Read the independent checks, and check that the "
           "falsifier could really have failed.", "",
           "## Summary", "",
           "| notebook | assumed | imported | derived | reviewed (ok) | doubt / wrong |",
           "|---|---|---|---|---|---|"]
    groups = [("curated", "curated assumptions")] + [(nb, f"nb{nb}") for nb in notebooks]
    for key, label in groups:
        es = [e for e in entries if e["group"] == key]
        c = {k: sum(e["kind"] == k for e in es) for k in KINDS}
        ok = sum(e["status"] == "ok" for e in es)
        bad = sum(e["status"] in ("doubt", "wrong") for e in es)
        out.append(f"| {label} | {c['assumed']} | {c['imported']} | {c['derived']} | "
                   f"{ok}/{len(es)} | {bad} |")
    if ledgerless:
        out += ["", "Notebooks with no derivation ledger yet: "
                + ", ".join(f"`{n}`" for n in ledgerless) + "."]

    out += ["", "## 1. Load-bearing assumptions", "",
            "| id | status | assumption | where | affects | robustness |",
            "|---|---|---|---|---|---|"]
    for e in (e for e in entries if e["kind"] == "assumed"):
        out.append(f"| `{e['id']}` | {_BADGE[e['status']]} | {_cell(e['claim'])} | "
                   f"{_cell(e['where'])} | {_cell(e['affects'])} | "
                   f"{_cell(e['robustness'])} |")

    for num, kind, title in ((2, "imported", "Imported from the literature"),
                             (3, "derived", "Derived here")):
        out += ["", f"## {num}. {title}", ""]
        for e in (e for e in entries if e["kind"] == kind):
            out.append(f"#### `{e['id']}` · {_BADGE[e['status']]} · {_gh_math(e['where'])}")
            out.append(f"**{_gh_math(e['claim'])}**")
            if e["statement"]:
                out += ["", f"$${e['statement']}$$"]
            out.append("")
            out += [f"- check: {_gh_math(c)}" for c in e["checks"]] or ["- check: *none recorded*"]
            if e["falsified_by"]:
                out.append(f"- falsified by: {_gh_math(e['falsified_by'])}")
            if e["downstream"]:
                out.append(f"- notebook feeds: {e['downstream']}")
            if e["note"]:
                out.append(f"- review note: {_gh_math(e['note'])}")
            out.append("")

    if orphans:
        out += ["## Orphaned review marks", "",
                "These marks match no current claim; the claim was removed or rewritten. "
                "Delete them from `review_status.csv` once you have checked them.", "",
                "| id | status | note | was |", "|---|---|---|---|"]
        for k, m in orphans.items():
            out.append(f"| `{k}` | {m['status']} | {_cell(m['note'])} | "
                       f"{_cell(m.get('claim', ''))} |")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------- main

def _load(project, name):
    """Import ``project/name.py`` by path (it is not an installed module)."""
    path = project / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"_project_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_assumptions(project):
    pipe, assum = _load(project, "pipeline"), _load(project, "assumptions")
    return (getattr(pipe, "PIPELINE", {}) if pipe else {},
            getattr(assum, "ASSUMPTIONS", []) if assum else [])


def build(project):
    project = Path(project).resolve()
    pipeline, curated = _load_assumptions(project)

    entries = []
    for a in curated:
        origins = a["introduced_in"]
        origins = [origins] if isinstance(origins, str) else list(origins)
        down = downstream(pipeline, origins)
        affects = a["affects"]
        if down:
            affects += " · downstream: " + ", ".join(_node_label(n) for n in down)
        entries.append({"id": a["id"], "group": "curated", "kind": "assumed",
                        "claim": a["statement"],
                        "where": a["where"] + " · from " + ", ".join(
                            _node_label(o) for o in origins),
                        "affects": affects, "robustness": a["robustness"],
                        "source": a["id"]})

    texs = sorted(project.glob("derivations_*.tex"))
    notebooks = []
    for tex in texs:
        for s in parse_ledger_tex(tex):
            nb = s["notebook"]
            if nb not in notebooks:
                notebooks.append(nb)
            down = downstream(pipeline, nb)
            where = f"nb{nb}" + (f" §{s['section']}" if s["section"] else "")
            entries.append({**s, "group": nb, "where": where,
                            "source": f"nb{nb} step {s['n']}",
                            "affects": "only this notebook" if not down else
                            "nb" + nb + " and downstream: "
                            + ", ".join(_node_label(n) for n in down),
                            "robustness": "; ".join(s["checks"]) or "no check recorded",
                            "downstream": ", ".join(_node_label(n) for n in down)})

    present = {p.name[:2] for p in project.glob("[0-9][0-9]_*.ipynb")}
    ledgerless = sorted(present - set(notebooks))

    marks_path = project / "review_status.csv"
    rows, orphans = reconcile(entries, load_marks(marks_path))
    with marks_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
        for k, m in orphans.items():      # kept, never silently dropped
            w.writerow({f: m.get(f, "") for f in CSV_FIELDS} | {"id": k})

    (project / "CLAIMS.md").write_text(render(entries, notebooks, orphans, ledgerless))
    counts = {k: sum(e["kind"] == k for e in entries) for k in KINDS}
    print(f"CLAIMS.md: {len(entries)} claims ({counts['assumed']} assumed, "
          f"{counts['imported']} imported, {counts['derived']} derived) from "
          f"{len(texs)} ledgers + {len(curated)} curated assumptions; "
          f"{sum(e['status'] == 'ok' for e in entries)} reviewed ok, "
          f"{len(orphans)} orphaned marks")
    return entries


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)

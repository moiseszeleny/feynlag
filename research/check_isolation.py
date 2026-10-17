"""Check rule 6 of research/README.md: nothing outside research/ refers to it.

Searches every tracked file outside ``research/`` (plus files that are new but
not ignored) for a path into this directory, or for prose pointing at
research notebooks, code or scans. Prints each hit and exits with status 1 if
there is any; prints nothing else and exits 0 when the repo is clean.

Usage (from anywhere)::

    python research/check_isolation.py

It lives inside research/ so that checking the rule does not itself break it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: What counts as a reference: a path into this directory (also written with a
#: Windows separator, or starting at a project folder; `main` adds every
#: project folder name), or prose pointing at the research material.
#: "research" on its own is fine: "research domain", or the "Science/Research"
#: classifier, names a topic, not a file.
PATTERN = (r"research[/\\]"
           r"|research (notebook|code|scan|project|folder|director)")


def main():
    projects = [p.name for p in Path(__file__).resolve().parent.iterdir() if p.is_dir()]
    pattern = PATTERN + "".join(rf"|\b{name}/" for name in projects
                                if name not in ("__pycache__",))
    out = subprocess.run(
        ["git", "grep", "-n", "-I", "-i", "-E", "--untracked", pattern, "--",
         ".", ":(exclude)research/"],
        cwd=REPO, capture_output=True, text=True)
    if out.returncode not in (0, 1):            # 1 = no match
        sys.exit(f"git grep failed: {out.stderr.strip()}")
    hits = out.stdout.strip().splitlines()
    for line in hits:
        print(line[:200])
    if hits:
        print(f"\n{len(hits)} reference(s) to research/ outside it (research/README.md rule 6)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

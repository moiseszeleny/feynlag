# Notebook style: written for a student

The research notebooks are read by people new to the project: a student, a collaborator
starting on the model, a referee. Write for that reader. They know quantum field theory but
not this repository's history. Notebook [`03_scalar_decays.ipynb`](03_scalar_decays.ipynb)
is the reference implementation of this guide.

**These changes are presentation only.** Restyling a notebook must not change a computed
number. See [Verifying a restyle](#verifying-a-restyle).

## 1. Opening cells

The first markdown cell has, in this order:

1. **Title**, and one line: *Research notebook NN of 09. Its conclusions are findings, and
   may be revised.*
2. **The question**: the physics the notebook answers, in a paragraph a newcomer can follow.
3. **Where this fits**: the 01–09 table, with this notebook in bold.
4. **Results at a glance**: the numbered findings, one or two sentences each, with § links.
5. **What you need first**: the physics background, the tutorial, and the earlier sections
   it relies on.
6. **Notation**: a table of *symbol · meaning · where defined* for every symbol used before
   it is introduced, and a line saying that `[Tags]` are references.
7. **How to read this notebook**: the rhythm (idea → compute → check), what a ✓ means,
   the *Think first* boxes, and the hidden `ledger` cells.

## 2. Set-up

```python
from nbsetup import *        # np, sp, plt, dv, store, ok, check_quoted, table, colours

import decays as D           # one comment per import: what it provides
...
ledger = dv.Ledger("NN — title")
```

- [`nbsetup.py`](nbsetup.py) holds what every notebook repeats: the logging silencer, plot
  style, `sp.init_printing`, colours, `ok`, `check_quoted` and `table`. Never copy them
  back into a notebook.
- Import only what the notebook uses.

## 3. Narration is physics, not history

| instead of | write |
|---|---|
| "`NOTES.md` recorded this as a blocker; §1 fixes it" | "To read off couplings between mass eigenstates the model must know the rotation that defines them." |
| "the engine used to refuse VSS" | "These come from `VSS` vertices, a vector and two scalars with a derivative coupling." |
| "the sign error this replaced was ~0.3 rad" | (move to `NOTES.md`) |
| "this notebook first built $M_\ell$ with √3 typed in" | "**The trap.** A tempting shortcut types the √3 in… The check below must catch it." |

- The history (which bug was fixed, which reading was superseded) belongs in `NOTES.md`.
  Check that it is recorded there before deleting it from a notebook.
- A wrong approach worth knowing about stays in the notebook as a **trap** or a **negative
  control**, described in the present tense, not as a confession.
- Keep section numbers and titles stable: `CLAIMS.md`, `ROADMAP_LFV.md`, `BENCHMARKS.md` and
  the reports cite them as `nbNN §n`. If a title must change (03 §5 dropped "the channels
  the engine used to refuse"), keep the number.
- Explain each piece of jargon at first use: gauge-phobic, Källén function, decoupling
  limit, rate ratio vs branching ratio.

## 4. One idea per cell

- A code cell does one thing: compute, show, check, or plot. Aim for under ~40 lines.
  The `store.write` deliverable cell may be longer.
- Each code cell gets a short markdown cell before it: what it does and what to look for.
- A section's checks go in its last code cell, starting with `# checks`.
- Comments that explain *why* become markdown. Comments that explain a line stay as
  comments.

## 5. Outputs a person can read

- **No ASCII art.** Never `print()` a SymPy object. Use `dv.show(label, expr)` or
  `dv.rows([...])`, which typeset under the kernel. For an expression too big to read,
  show its structure (`collect`, a term count) and point to `derivations_NN.pdf`.
- **Tables** go through `table(headers, rows, fmt=[...])`, not printf columns. Use the
  same format specs the old printf used, so the numbers stay identical. Never name a
  variable `table` (notebook 09 did, which shadowed the helper), and escape a literal `|`
  in a cell as `\|`, or write it in LaTeX as `\lvert…\rvert`.
- **No stray `Out[]`.** End a cell with `;`, or assign `_ = ...`, when its last call returns
  a value (`ledger.step`, `dv.show`, `dv.falsify`, `ledger.to_latex`).
- **Figures:** axis labels with units, and a one-to-three-sentence *Figure:* caption cell
  under the plot saying what to see. Plotting cells hold only the plot. Keep each figure's
  `set_title` text, because `extract_figures.py` finds the cell by that marker.

## 6. Checks

- Keep every `assert`: they are the verification.
- An `ok()` message says in words what was verified.
- `ledger.step(...)` cells get the metadata `{"jupyter": {"source_hidden": true},
  "tags": ["ledger"]}`. Keep their order: it fixes the order of `derivations_NN.tex`.

## 7. Think first

```markdown
> **Think first.** <the question, to predict before running>
```

(Older notebooks used "Before running the next cell", "Predict" and "Work this one out
before reading on" for the same thing; they now all read **Think first**.) If the text
right after the box gives the answer away, move that text into the folded answer.

After the cell that settles it, add the answer folded:

```markdown
<details><summary><b>Answer to the think-first question</b></summary>
...
</details>
```

## Restyling by script

Rebuild a notebook from its git-HEAD version with a short script, rather than by editing
the JSON by hand. The 2026-11 rollout used three operations:
- `sub`: change text inside one cell (a printf block into a `table()` call);
- `split`: cut a code cell before the lines that start with given markers, inserting a
  markdown cell before each piece;
- `insert_before`: add a markdown cell.

Code is cut, never retyped. Apply a cell's `sub`s **before** its `split`: a split
captures the text at the moment it is made, so a later `sub` on the same cell is
silently lost (the rollout's script refused that order).

## Verifying a restyle

Run all of this from `research/thdm_s3/`, after saving the old notebook:

1. Re-execute it with the miniconda nbconvert (`NOTES.md`, *Re-execution*). Every assert must
   pass and there must be no `error` output.
2. The files the node produces (`pipeline.py`) must not change: `derivations_NN.tex` byte
   for byte, and `results/*.json` up to round-off. **Check round-off against a re-run of the
   old notebook in the same environment, not against git.** For 03, the environment had
   drifted since the commit, so the old and new notebooks agreed byte for byte while both
   differed from git at the 10⁻¹⁵ level (and in quantities that are zero by symmetry). In
   that case keep the committed results (`git checkout results/<file>`), and delete the
   matching `cache/<file>`, so downstream nodes do not go stale for nothing.
   **Check who reads the cache first.** A node that loads the per-point file through
   `store.full`/`store.points` (notebook 08 reads `cache/mu_e_gamma.json`; `benchmarks.py`
   reads the soft-decay and τ caches) needs the summary's `points_sha256` to match the
   local cache. Keeping a committed summary then means that consumer cannot run in this
   environment. In the 2026-11 rollout, 07's summary was therefore kept as regenerated
   (only its hash line changed), while `benchmarks.py` was left stale.
   A difference that is not round-off but that the notebook itself declares
   environment-dependent (notebook 02: the quark fit finds 4 exact solutions with today's
   numpy/scipy where the commit has 5, and every claim says "at least four") is kept as
   regenerated, so that the stored outputs and the committed results agree. Re-run the
   nodes downstream of it.
3. Every number in the old outputs appears in the new ones, and the other way round,
   timings excepted. The ✓ count is unchanged too.
4. Update the cell indices of `extract_figures.py` and run it.
5. `provenance.py stamp NN`, then `provenance.py check`: the node and its report are `ok`.

"""Guided physical basis: the model finds its own mass blocks.

Going from a Lagrangian to the mass eigenstates by hand means deciding which
symbols form each block, building every mass matrix, knowing which rotation
each block needs, creating the new symbols and wiring the conjugate maps.
This module does the bookkeeping and leaves the physics choices (the names)
to the user:

- :func:`mass_blocks` — partition the scalar fluctuations and the Dirac
  fermion legs into the blocks that actually mix (connected components of the
  mass matrix), each tagged with its electric charge (derived from the
  vacuum, :mod:`feynlag.charges`), its CP content, and its Goldstone
  directions (derived from the broken generators, not assumed).
- :func:`diagonalize_block` — pick the diagonalization route for one block,
  verify it, register the rotation(s) and record the new states.
- :class:`PhysicalRegistry` — the accumulated physical fields, conjugate maps,
  charges, Goldstones and masses (``model.physical``).

Nothing is assumed about CP or alignment: with a CP-violating potential the
CP-even and CP-odd fluctuations simply come out as one larger block.  Where a
closed form does not exist the functions refuse with the available options
rather than return an unusable expression.
"""

from dataclasses import dataclass, field as dc_field

import sympy as sp

from ..conventions import SQRT2
from ..texsymbol import tex_symbol
from .diagonalize import (
    Rotation, diagonalize_svd, diagonalize_svd_2x2, rotation_2x2,
    solve_mixing_angle_2x2,
)
from .masses import charged_mass_matrix, scalar_mass_matrix

__all__ = ["MassBlock", "Diagonalization", "PhysicalRegistry", "mass_blocks",
           "diagonalize_block", "goldstone_directions", "charge_operator"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _is_zero(expr):
    if expr == 0:
        return True
    return sp.simplify(expr) == 0


def _components(n, connected):
    """Connected components of ``range(n)`` under ``connected(a, b)``."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if connected(a, b):
                parent[find(a)] = find(b)
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: g[0])


def _unit_phase(vec):
    """Rescale a vector so its first non-zero entry is real and positive."""
    for entry in vec:
        if not _is_zero(entry):
            phase = sp.simplify(entry / sp.Abs(entry))
            return (vec / phase).applyfunc(sp.simplify)
    return vec


def _normalized(vec):
    norm2 = sp.simplify(sum(sp.Abs(e) ** 2 for e in vec))
    return _unit_phase((vec / sp.sqrt(norm2)).applyfunc(sp.simplify))


def _independent(vectors):
    """Orthonormal basis (complex inner product) of the span of ``vectors``."""
    basis = []
    for v in vectors:
        w = sp.Matrix(v)
        for b in basis:
            w = w - (b.H * w)[0] * b
        w = w.applyfunc(sp.simplify)
        if all(_is_zero(e) for e in w):
            continue
        basis.append(_normalized(w))
    return basis


# --------------------------------------------------------------------------
# electric charge and Goldstone directions
# --------------------------------------------------------------------------

def charge_operator(model):
    """The unbroken charge operator, normalised, or ``None``.

    :func:`~feynlag.charges.derive_charge_operator` fixes the operator only up
    to a scale; here it is normalised so the first non-abelian Cartan
    coefficient is ``+1`` (``Q = T³ + …``), or the first non-zero coefficient
    when there is none.  Returns ``None`` when no gauge group acts on a VEV
    (nothing is broken, every scalar is neutral by convention).
    """
    from ..charges import _vevd_components, derive_charge_operator

    vevd = _vevd_components(model)
    if not any(g in f.reps for g in model.gauge_groups for f, _ in vevd):
        return None
    coeffs = derive_charge_operator(model)
    pivot = next((c for (g, idx), c in coeffs.items()
                  if idx is not None and c != 0), None)
    if pivot is None:
        pivot = next(c for c in coeffs.values() if c != 0)
    return {k: sp.simplify(c / pivot) for k, c in coeffs.items()}


def _component_charges(model, fields, coeffs):
    """``{component: charge}`` for the given fields (``None`` = not an
    eigenstate; every charge is 0 when ``coeffs`` is ``None``)."""
    from ..charges import _eigen_charge, _field_charge_matrix

    out = {}
    for f in fields:
        if coeffs is None:
            for comp in f.components:
                out[comp] = sp.S.Zero
            continue
        Q = _field_charge_matrix(f, coeffs)
        for k, comp in enumerate(f.components):
            out[comp] = _eigen_charge(Q, k)
    return out


def goldstone_directions(model):
    """Goldstone directions ``i T^a ⟨φ⟩`` in fluctuation coordinates.

    One dict ``{coordinate symbol: value}`` per gauge generator that does not
    annihilate the vacuum.  The coordinates are the real fluctuations of the
    VEV'd components (``φ = (v + φ_r + iφ_i)/√2`` ⟹ ``δφ_r = √2 Re δφ``,
    ``δφ_i = √2 Im δφ``) and the remaining complex components themselves.
    VEVs are taken real, as :meth:`~feynlag.fields.Scalar.expand_vev` assumes.
    """
    directions = []
    for group in model.gauge_groups:
        n_gen = 1 if group.abelian else group.n_generators
        for a in range(n_gen):
            coords = {}
            for f in model.scalars:
                expansions = getattr(f, "vev_expansions", {})
                if not expansions or group not in f.reps:
                    continue
                # (a component expanded around a zero VEV contributes 0)
                phi0 = sp.Matrix([
                    (expansions[c][0] if expansions[c][2] is None
                     else expansions[c][0] / SQRT2) if c in expansions else 0
                    for c in f.components])
                T = (f.charge(group) * sp.eye(f.dim) if group.abelian
                     else sp.Matrix(f.generators(group)[a]))
                delta = (sp.I * T * phi0).applyfunc(sp.expand)
                for comp, d in zip(f.components, delta):
                    if d == 0:
                        continue
                    if comp in expansions:
                        _, re, im = expansions[comp]
                        if im is None:
                            coords[re] = d
                        else:
                            coords[re] = sp.simplify(SQRT2 * sp.re(d))
                            coords[im] = sp.simplify(SQRT2 * sp.im(d))
                    else:
                        coords[comp] = d
            coords = {k: v for k, v in coords.items() if v != 0}
            if coords:
                directions.append(coords)
    return directions


# --------------------------------------------------------------------------
# MassBlock
# --------------------------------------------------------------------------

@dataclass
class MassBlock:
    """One set of fields that mix among themselves in the mass matrix.

    Attributes:
        kind: ``"scalar"`` or ``"fermion"``.
        fields: the basis.  Scalars: real fluctuation symbols, or complex
            components for a charged block.  Fermions: the left-handed
            component ``IndexedBase``\\ s (``right`` holds the right-handed).
        matrix: ``M²`` (scalars; symmetric, or hermitian with rows ``∂/∂φ̄``
            for a complex block) or the Dirac mass matrix ``M`` (fermions;
            rows bar-left, columns right).
        charge: electric charge of the members (``None`` if unknown).
        cp: ``"even"``, ``"odd"``, ``"mixed"`` for a real neutral block,
            else ``None``.
        complex: the basis is made of complex fields.
        goldstones: orthonormal Goldstone vectors in this basis.
        right: fermion blocks — the right-handed components.
        nflavors: fermion blocks — flavours per component.
        diagonalizable: ``False`` when :func:`diagonalize_block` has no route
            for the block; ``note`` then says what to use instead.
    """

    kind: str
    fields: list
    matrix: sp.Matrix
    charge: object = None
    cp: object = None
    complex: bool = False
    goldstones: list = dc_field(default_factory=list)
    right: list = dc_field(default_factory=list)
    nflavors: int = 1
    diagonalizable: bool = True
    note: str = ""

    @property
    def size(self):
        return len(self.fields)

    def __repr__(self):
        parts = [self.kind,
                 "Q=?" if self.charge is None else f"Q={self.charge}"]
        if self.cp:
            parts.append(f"cp-{self.cp}")
        names = ", ".join(str(getattr(f, "label", f)) for f in self.fields)
        parts.append(f"[{names}]")
        if self.kind == "fermion":
            rnames = ", ".join(str(r.label) for r in self.right)
            parts.append(f"x [{rnames}]")
            if self.nflavors != 1:
                parts.append(f"flavours: {self.nflavors}")
        else:
            parts.append(f"goldstones: {len(self.goldstones)}")
        if not self.diagonalizable:
            parts.append(f"({self.note})")
        return "  ".join(parts)


# --------------------------------------------------------------------------
# scalar blocks
# --------------------------------------------------------------------------

def _zero_scalars(model):
    """Substitution setting every scalar component without a VEV expansion,
    and its conjugate, to zero (``Vacuum.zero_fluctuations`` only knows the
    VEV'd fields)."""
    sub = {}
    for f in model.scalars:
        expansions = getattr(f, "vev_expansions", {})
        for comp in f.components:
            if comp not in expansions:
                sub[sp.conjugate(comp)] = 0
                sub[comp] = 0
    return sub


def _clean(M, zero):
    return M.applyfunc(lambda e: sp.factor(sp.expand(e.subs(zero)))
                       if e != 0 else sp.S.Zero)


def _goldstones_in(basis, directions):
    vectors = []
    for coords in directions:
        vec = sp.Matrix([coords.get(s, 0) for s in basis])
        if any(e != 0 for e in vec):
            vectors.append(vec)
    return _independent(vectors)


def _check_goldstones(block):
    for g in block.goldstones:
        residual = (block.matrix * g).applyfunc(sp.simplify)
        if any(not _is_zero(e) for e in residual):
            names = ", ".join(str(f) for f in block.fields)
            raise ValueError(
                f"the Goldstone direction of the block [{names}] is not a "
                f"zero mode of its mass matrix, so the vacuum is not a "
                f"stationary point of the potential as given: solve the "
                f"tadpoles first (model.solve_tadpoles([...])).")


def _scalar_blocks(model, coeffs):
    potential = model.potential
    vacuum = model.vacuum
    tadpoles = model._tadpole_solutions
    zero = _zero_scalars(model)
    charges = _component_charges(model, model.scalars, coeffs)
    directions = goldstone_directions(model)
    blocks = []

    # ---- real neutral sector: VEV fluctuations + real VEV-less components
    cp_of = {}
    reals = []
    complex_comps = []
    for f in model.scalars:
        expansions = getattr(f, "vev_expansions", {})
        for comp in f.components:
            if comp in expansions:
                _, re, im = expansions[comp]
                reals.append(re)
                cp_of[re] = "even"
                if im is not None:
                    reals.append(im)
                    cp_of[im] = "odd"
            elif f.real:
                reals.append(comp)
                cp_of[comp] = "even"
            else:
                complex_comps.append(comp)

    if reals:
        M = _clean(scalar_mass_matrix(potential, vacuum, reals,
                                      tadpole_subs=tadpoles), zero)
        for idx in _components(len(reals), lambda a, b: M[a, b] != 0
                               or M[b, a] != 0):
            basis = [reals[k] for k in idx]
            tags = {cp_of[s] for s in basis}
            blocks.append(MassBlock(
                "scalar", basis, M.extract(idx, idx), charge=sp.S.Zero,
                cp=tags.pop() if len(tags) == 1 else "mixed",
                goldstones=_goldstones_in(basis, directions)))

    # ---- complex components (charged, or neutral without a VEV expansion)
    if complex_comps:
        for comp in complex_comps:
            if charges[comp] is None:
                raise NotImplementedError(
                    f"the component {comp} is not an electric-charge "
                    f"eigenstate in the declared basis; rotate it to charge "
                    f"eigenstates before calling mass_blocks")
        Mc = _clean(charged_mass_matrix(potential, vacuum, complex_comps,
                                        tadpole_subs=tadpoles), zero)
        # holomorphic second derivatives ∂²V/∂a∂b: non-zero means a field
        # mixes with a *conjugate* field, which this basis cannot express
        # (conjugates are held fixed through Dummies, as in
        # charged_mass_matrix: φ and φ* are independent variables)
        dummies = {sp.conjugate(c): sp.Dummy() for c in complex_comps}
        shifted = vacuum.shift(potential).xreplace(dummies)
        zero_all = {**zero, **{d: 0 for d in dummies.values()}}
        for a, ca in enumerate(complex_comps):
            for cb in complex_comps[a:]:
                d2 = sp.diff(shifted, ca, cb)
                d2 = sp.expand(vacuum.zero_fluctuations(d2).subs(zero_all))
                if tadpoles:
                    d2 = sp.expand(d2.subs(tadpoles))
                if _is_zero(d2):
                    continue
                if ca == cb or charges[ca] == 0:
                    raise NotImplementedError(
                        f"the neutral complex component {ca} has a φφ mass "
                        f"term, so its real and imaginary parts are split. "
                        f"Declare them with expand_vev({{{ca}: 0}}) on its "
                        f"field (a zero VEV), then call mass_blocks again.")
                raise NotImplementedError(
                    f"the components {ca} and {cb} carry opposite charge and "
                    f"mix ({ca}·{cb} mass term): the block needs {ca} "
                    f"together with conjugate({cb}), which mass_blocks does "
                    f"not build yet. Use model.mass_matrix by hand, or "
                    f"redeclare one field in the conjugate representation.")
        for idx in _components(
                len(complex_comps),
                lambda a, b: Mc[a, b] != 0 or Mc[b, a] != 0):
            basis = [complex_comps[k] for k in idx]
            qs = {charges[c] for c in basis}
            if len(qs) != 1:
                raise ValueError(
                    f"components of different electric charge mix: "
                    f"{dict((str(c), charges[c]) for c in basis)}. The vacuum "
                    f"or the potential breaks electric charge.")
            blocks.append(MassBlock(
                "scalar", basis, Mc.extract(idx, idx), charge=qs.pop(),
                complex=True,
                goldstones=_goldstones_in(basis, directions)))

    for block in blocks:
        _check_goldstones(block)
    return blocks


# --------------------------------------------------------------------------
# fermion (Dirac) blocks
# --------------------------------------------------------------------------

def _fermion_blocks(model, coeffs):
    from ..dirac import diracPR
    from ..fields import Fermion
    from ..vertices.bilinear import (
        Bilinear, MajoranaBilinear, expand_bilinear, fermion_mass_matrix,
    )

    fermions = [f for f in model.fields if isinstance(f, Fermion)]
    if not fermions:
        return []
    sectors = [t.expr for t in model.lagrangian.terms
               if t.sector in ("yukawa", "other")]
    L0 = sp.expand(expand_bilinear(
        model.vacuum.at_vacuum(sp.expand(sp.Add(*sectors)))))
    if model._tadpole_solutions:
        L0 = sp.expand(L0.subs(model._tadpole_solutions))
    terms = L0.as_ordered_terms() if L0.is_Add else ([L0] if L0 != 0 else [])
    mass_terms, majorana_bases = [], set()
    for term in terms:
        for mb in term.atoms(MajoranaBilinear):
            for leg in (mb.field1, mb.field2):
                majorana_bases.add(getattr(leg, "base", None))
        bils = list(term.atoms(Bilinear))
        if len(bils) == 1 and bils[0].gamma == diracPR:
            mass_terms.append(term)
    L_mass = sp.Add(*mass_terms)

    left, right, nfl = [], [], {}
    for f in fermions:
        for comp in f.components:
            nfl[comp] = f.nflavors
            (left if f.chirality == "L" else right).append(comp)
    bar_of = {c: f.bar(c) for f in fermions for c in f.components}
    charges = _component_charges(model, fermions, coeffs)

    sub = {}
    for a, lc in enumerate(left):
        for b, rc in enumerate(right):
            if not L_mass.has(bar_of[lc]) or not L_mass.has(rc):
                continue
            block = fermion_mass_matrix(L_mass, bar_of[lc], rc, model.vacuum,
                                        nfl[lc], None, gamma=diracPR) \
                if nfl[lc] == nfl[rc] else None
            if block is None:
                raise NotImplementedError(
                    f"{lc.label} and {rc.label} have different numbers of "
                    f"flavours; build this mass matrix by hand")
            if any(e != 0 for e in block):
                sub[(a, b)] = block.applyfunc(
                    lambda e: sp.factor(e) if e != 0 else sp.S.Zero)

    # bipartite connected components over left ∪ right
    nL = len(left)
    nodes = list(range(nL + len(right)))

    def connected(x, y):
        if x < nL <= y:
            return (x, y - nL) in sub
        return False

    blocks = []
    for comp_idx in _components(len(nodes), connected):
        ls = [k for k in comp_idx if k < nL]
        rs = [k - nL for k in comp_idx if k >= nL]
        if not ls or not rs:
            lone = left[ls[0]] if ls else right[rs[0]]
            if lone in majorana_bases:
                continue          # a purely Majorana state: not a Dirac block
            # a massless chiral fermion
            blocks.append(MassBlock(
                "fermion", [left[k] for k in ls], sp.zeros(len(ls), len(rs)),
                charge=charges[lone], right=[right[k] for k in rs],
                nflavors=nfl[lone], diagonalizable=False,
                note="massless: no Dirac partner"))
            continue
        n = nfl[left[ls[0]]]
        rows = []
        for a in ls:
            rows.append([sub.get((a, b), sp.zeros(n, n)) for b in rs])
        M = sp.Matrix(sp.BlockMatrix(rows))
        block = MassBlock("fermion", [left[k] for k in ls], M,
                          charge=charges[left[ls[0]]],
                          right=[right[k] for k in rs], nflavors=n)
        members = set(block.fields) | set(block.right)
        if members & majorana_bases:
            block.diagonalizable = False
            block.note = ("has a Majorana mass term: use seesaw_mass_matrix "
                          "+ diagonalize_takagi + MajoranaRotation")
        elif n != 1:
            block.diagonalizable = False
            block.note = ("flavoured: diagonalize the matrix with "
                          "diagonalize_svd(method='numeric')")
        elif len(ls) != len(rs):
            block.diagonalizable = False
            block.note = "rectangular: unequal numbers of left and right legs"
        blocks.append(block)
    return blocks


def mass_blocks(model, kind=None):
    """The mass blocks of ``model`` at its vacuum.

    Args:
        kind: ``"scalar"``, ``"fermion"`` or ``None`` (both).

    Returns:
        list of :class:`MassBlock`, scalars first (real neutral blocks, then
        complex ones), in field-declaration order.

    Raises:
        ValueError: a Goldstone direction is not a zero mode (unsolved
            tadpoles), or the vacuum breaks electric charge.
        NotImplementedError: a block needs a basis this function does not
            build (a field mixing with a conjugate field); the message names
            the components and the way out.
    """
    if kind not in (None, "scalar", "fermion"):
        raise ValueError(f"unknown block kind {kind!r}")
    coeffs = charge_operator(model)
    out = []
    if kind in (None, "scalar") and model.scalars:
        out += _scalar_blocks(model, coeffs)
    if kind in (None, "fermion"):
        out += _fermion_blocks(model, coeffs)
    return out


# --------------------------------------------------------------------------
# registry of physical states
# --------------------------------------------------------------------------

class PhysicalRegistry:
    """The physical states of a model, accumulated as rotations are made.

    Attributes:
        bosons: physical boson symbols, conjugate partners included — the
            ``fields`` argument of ``Model.vertices`` / ``feynman_rules``.
        cmap: ``{conjugate(field): partner}`` for the vertex extractor.
        conjugates: the antiparticle pairing, both directions.
        goldstones: the would-be Goldstone symbols (with partners).
        masses: ``{field: m²}`` for scalars, ``{name: m}`` for fermions.
        fermions: new fermion ``IndexedBase``\\ s (left, right and bars).
    """

    def __init__(self, model):
        self._model = model
        self.bosons = []
        self.cmap = {}
        self.conjugates = {}
        self.goldstones = []
        self.masses = {}
        self.fermions = []

    def add_bosons(self, symbols, goldstones=()):
        for s in symbols:
            if s not in self.bosons:
                self.bosons.append(s)
        for s in goldstones:
            if s not in self.goldstones:
                self.goldstones.append(s)

    def add_conjugate_pair(self, field, partner, cmap=None):
        self.conjugates[field] = partner
        self.conjugates[partner] = field
        if cmap:
            self.cmap.update(cmap)

    @property
    def charges(self):
        """``{physical boson: electric charge}``, derived from the vacuum and
        propagated through the registered rotations
        (:func:`~feynlag.charges.physical_charges`); a conjugate partner
        carries minus its field's charge."""
        from ..charges import physical_charges

        coeffs = charge_operator(self._model)
        if coeffs is None:
            return {b: sp.S.Zero for b in self.bosons}
        derived = physical_charges(self._model, coeffs)
        out = {}
        for b in self.bosons:
            if b in derived:
                out[b] = derived[b]
        for b in self.bosons:
            if b not in out and self.conjugates.get(b) in out:
                out[b] = -out[self.conjugates[b]]
        return out

    def __repr__(self):
        return (f"PhysicalRegistry(bosons={self.bosons}, "
                f"goldstones={self.goldstones})")


# --------------------------------------------------------------------------
# diagonalization
# --------------------------------------------------------------------------

@dataclass
class Diagonalization:
    """Result of :func:`diagonalize_block`.

    Attributes:
        block: the :class:`MassBlock`.
        rotations: the :class:`Rotation`\\ s (registered on the model unless
            ``register=False``); empty for a 1×1 block kept as is.
        new_fields: the physical symbols, in ``names`` order (fermions: the
            left-handed ``IndexedBase``\\ s; ``new_right`` has the others).
        masses_squared: scalars — ``m²`` per new field.
        masses: fermions — ``m`` per new field (may be signed for a symbolic
            2×2; the physical mass is its absolute value).
        goldstones: the new symbols that are would-be Goldstones.
        cmap: ``{conjugate(field): partner}`` for the complex new fields.
        angle_relation: the defining relation of the mixing angle, if any.
        branch_valid_for: with ``values=`` on a symbolic 2×2 — the signs of
            ``(M11 − M22, M12)`` at that point; the mass ordering of the two
            names holds where those signs do.
    """

    block: MassBlock
    rotations: list
    new_fields: list
    masses_squared: list = dc_field(default_factory=list)
    masses: list = dc_field(default_factory=list)
    goldstones: list = dc_field(default_factory=list)
    cmap: dict = dc_field(default_factory=dict)
    new_right: list = dc_field(default_factory=list)
    angle_relation: object = None
    angle_solution: object = None
    branch_valid_for: object = None

    @property
    def rotation(self):
        return self.rotations[0] if self.rotations else None


def _angle_symbol(angle):
    if angle is None:
        return None
    if isinstance(angle, str):
        return sp.Symbol(angle, real=True)
    return sp.sympify(getattr(angle, "symbol", angle))


def _register_angle(model, angle, solution):
    """A string ``angle`` becomes an InternalParameter of the model; an
    undefined InternalParameter passed in gets defined."""
    from ..parameters import InternalParameter

    if isinstance(angle, str) and angle not in model.parameters:
        model.parameters.add(InternalParameter(angle, solution))
    elif isinstance(angle, InternalParameter) and angle.expr is None:
        angle.define(solution)


def _numeric_matrix(M, values, what):
    Mn = sp.Matrix(M).subs(_value_map(values))
    if Mn.free_symbols:
        raise ValueError(
            f"{what}: values= leaves free symbols "
            f"{sorted(Mn.free_symbols, key=str)} in the mass matrix")
    return Mn


def _value_map(values):
    """``values`` keyed by symbol, floats read as exact decimals (``0.3`` →
    ``3/10``) so a numeric rotation is computed at full working precision and
    a Goldstone comes out as an exact zero mode, not as 1e-13."""
    return {sp.sympify(getattr(k, "symbol", k)):
            sp.nsimplify(v, rational=True) for k, v in (values or {}).items()}


def _numeric_hermitian(Mn, dps=50):
    """``(R, eigenvalues)`` with rows of ``R`` the eigenvectors, ascending."""
    import mpmath

    n = Mn.rows
    with mpmath.workdps(dps + 10):
        A = mpmath.matrix(n, n)
        for a in range(n):
            for b in range(n):
                re, im = sp.N(Mn[a, b], dps + 10).as_real_imag()
                A[a, b] = mpmath.mpc(mpmath.mpf(str(re)), mpmath.mpf(str(im)))
        real = all(mpmath.im(A[a, b]) == 0 for a in range(n) for b in range(n))
        if real:
            A = A.apply(mpmath.re)
            E, Q = mpmath.eigsy(A)
        else:
            E, Q = mpmath.eighe(A)
        scale = max((abs(e) for e in E), default=0)
        order = sorted(range(n), key=lambda k: E[k])

        def num(z):
            z = mpmath.mpc(z)
            out = sp.Float(str(z.real), dps)
            if z.imag != 0:
                out += sp.I * sp.Float(str(z.imag), dps)
            return out

        R = sp.Matrix(n, n, lambda i, a: num(mpmath.conj(Q[a, order[i]])))
        eig = [sp.S.Zero if abs(E[k]) <= scale * mpmath.mpf(10) ** (-dps + 5)
               else sp.Float(str(E[k]), dps) for k in order]
    return R, eig, real


def _new_symbols(names, tex, real):
    tex = tex or {}
    return [tex_symbol(n, tex.get(n), real=True) if real
            else tex_symbol(n, tex.get(n)) for n in names]


def _verify(rot, M, what, symbolic=True):
    """Check that ``rot`` diagonalizes ``M``: symbolically, and at random
    numeric points for whatever the symbolic pass cannot reduce to zero.

    ``symbolic=False`` skips the symbolic pass — for the ``tan 2θ`` route,
    whose explicit ``½·atan(…)`` angle makes ``simplify`` take minutes on an
    identity that holds by construction; the numeric pass still runs.
    """
    from ..verify import numeric_equal

    if symbolic:
        ok, residuals = rot.check(M)
        if ok:
            return
    else:
        D = rot.apply(M)
        n = D.shape[0]
        residuals = [((i, j), D[i, j]) for i in range(n) for j in range(n)
                     if i != j and D[i, j] != 0]
    scale = sum((sp.Abs(e) for e in sp.Matrix(M)), sp.S.Zero)
    for (i, j), r in residuals:
        syms = sorted((r + scale).free_symbols, key=str)
        if syms:
            good, _ = numeric_equal(r / scale, 0, syms, seed=0, tol=1e-9)
        else:
            good = abs(complex(sp.N(r / scale))) < 1e-9
        if not good:
            raise ValueError(
                f"{what}: the rotation does not diagonalize the mass matrix "
                f"(entry {(i, j)} is not zero)")


def _scalar_route(model, block, names, tex, angle, values, rotation, order):
    n = block.size
    M = block.matrix
    real = not block.complex
    what = f"diagonalize({block.fields})"
    info = {}
    k = len(block.goldstones)

    if rotation is not None:
        R = sp.Matrix(rotation)
        kind = "orthogonal" if real and all(e.is_real is not False
                                            for e in R) else "unitary"
        return R, kind, info, 0

    if n == 1:
        return sp.eye(1), "orthogonal", info, k

    if k >= n - 1:
        # the Goldstone vectors fix the rotation: rows are the projections
        # G_a = g_a† · φ, then the orthogonal complement
        rows = [g.H for g in block.goldstones[:n]]
        if k == n - 1:
            if n == 2:
                g = block.goldstones[0]
                comp = sp.Matrix([[-sp.conjugate(g[1]), sp.conjugate(g[0])]])
                comp = comp.applyfunc(sp.simplify)
                if real:
                    comp = sp.Matrix([[-g[1], g[0]]])
            else:
                G = sp.Matrix.vstack(*rows)
                null = G.nullspace()
                comp = _normalized(null[0]).H
            rows.append(comp)
        R = sp.Matrix.vstack(*rows).applyfunc(sp.simplify)
        is_real = all(sp.im(e) == 0 for e in R)
        sym = _angle_symbol(angle)
        if sym is not None:
            if not (n == 2 and is_real):
                raise ValueError(f"{what}: angle= needs a real 2×2 block")
            tan = sp.simplify(R[0, 1] / R[0, 0])
            info["angle_relation"] = sp.Eq(sp.tan(sym), tan)
            info["angle_solution"] = sp.atan(tan)
            R = rotation_2x2(sym)
        return R, "orthogonal" if is_real else "unitary", info, k

    if n == 2 and real and not (values and not M.free_symbols):
        sym = _angle_symbol(angle)
        solution, tan2theta = solve_mixing_angle_2x2(M)
        if values is not None:
            Mn = _numeric_matrix(M, values, what)
            m11, m22, m12 = (float(Mn[a, b]) for a, b in
                             ((0, 0), (1, 1), (0, 1)))
            if abs(m11 - m22) <= 1e-12 * max(abs(m11), abs(m22), 1e-300):
                raise ValueError(
                    f"{what}: M11 = M22 at values=, so the two states are "
                    f"not ordered by the tan(2θ) branch at this point")
            # principal ½·atan puts the heavier state first iff M11 > M22;
            # otherwise shift by ±π/2 with the sign of M12
            heavy_first = order == "heavy_first"
            shift = 0
            if (m11 > m22) != heavy_first:
                shift = 1 if m12 > 0 else -1
            solution = solution + shift * sp.pi / 2
            th = float(solution.subs(_value_map(values)))
            D = (rotation_2x2(th) * Mn * rotation_2x2(th).T)
            first_heavier = float(D[0, 0]) >= float(D[1, 1])
            if first_heavier != heavy_first:
                raise ValueError(f"{what}: branch selection failed at values=")
            info["branch_valid_for"] = (
                1 if m11 > m22 else -1,
                0 if m12 == 0 else (1 if m12 > 0 else -1))
        theta = sym if sym is not None else solution
        # without an angle symbol the relation is stated for a placeholder θ
        # (tan(2·atan(…)/2) would evaluate the equation away)
        info["angle_relation"] = sp.Eq(
            sp.tan(2 * (sym if sym is not None
                        else sp.Symbol("theta", real=True))), tan2theta)
        info["angle_solution"] = solution
        info["route"] = "tan2theta"
        return rotation_2x2(theta), "orthogonal", info, 0

    if values is not None:
        Mn = _numeric_matrix(M, values, what)
        R, eig, is_real = _numeric_hermitian(Mn)
        if order == "heavy_first":
            R = sp.Matrix(list(reversed([R.row(i) for i in range(n)])))
            eig = list(reversed(eig))
        info["numeric_masses"] = eig
        info["numeric_matrix"] = Mn
        return R, "orthogonal" if is_real else "unitary", info, 0

    raise NotImplementedError(
        f"{what}: a symbolic {n}×{n} block with {k} Goldstone(s) has no "
        f"closed-form rotation here. Options: pass values={{...}} (a "
        f"benchmark point) for a numeric rotation; impose a limit or symmetry "
        f"that splits the block and call mass_blocks again; or pass "
        f"rotation=<matrix> (it is checked against the mass matrix).")


def _diagonalize_scalar(model, block, names, tex, angle, values, rotation,
                        order, conjugate_names, register):
    n = block.size
    real = not block.complex
    what = f"diagonalize({block.fields})"
    if names is not None and len(names) != n:
        raise ValueError(f"{what}: need {n} names, got {len(names)}")
    if order not in ("heavy_first", "light_first"):
        raise ValueError("order must be 'heavy_first' or 'light_first'")

    R, kind, info, n_gold = _scalar_route(model, block, names, tex, angle,
                                          values, rotation, order)
    if n == 1 and names is None and rotation is None:
        new = list(block.fields)
        rotations = []
        m2 = [sp.S.Zero if block.goldstones else sp.simplify(block.matrix[0, 0])]
    else:
        if names is None:
            raise ValueError(f"{what}: names=[...] is required")
        new = _new_symbols(names, tex, real)
        rot = Rotation(block.fields, new, R, kind=kind)
        M_check = info.get("numeric_matrix", block.matrix)
        if "numeric_masses" in info:
            m2 = info["numeric_masses"]
        else:
            tan2 = info.get("route") == "tan2theta"
            explicit = rot
            if angle is not None and "angle_solution" in info:
                # verify with the explicit angle, then keep the symbol
                explicit = Rotation(block.fields, new, R.subs(
                    _angle_symbol(angle), info["angle_solution"]), kind=kind)
            _verify(explicit, M_check, what, symbolic=not tan2)
            if tan2:
                # kept in terms of the angle: the explicit ½·atan forms are
                # correct but unreadable (and slow to simplify)
                m2 = rot.masses_squared(block.matrix, simplifier=sp.expand)
            else:
                # a Goldstone-fixed rotation has a simple closed form
                m2 = explicit.masses_squared(block.matrix)
            for a in range(n_gold):
                m2[a] = sp.S.Zero
        if "angle_solution" in info:
            rot.angle_relation = info["angle_relation"]
            rot.angle_solution = info["angle_solution"]
            _register_angle(model, angle, info["angle_solution"])
        rotations = [rot]
        if register:
            model.rotate(rot)

    goldstones = list(new[:n_gold])
    if rotation is not None and block.goldstones:
        # a user-supplied rotation: the Goldstones are its massless states
        goldstones = [s for s, m in zip(new, m2)
                      if _is_zero(m)][:len(block.goldstones)]

    result = Diagonalization(
        block, rotations, new, masses_squared=m2, goldstones=goldstones,
        angle_relation=info.get("angle_relation"),
        angle_solution=info.get("angle_solution"),
        branch_valid_for=info.get("branch_valid_for"))

    if register:
        from ..fields import conjugate_pair

        reg = model.physical
        tex = tex or {}
        all_new, all_gold = [], []
        for k, s in enumerate(new):
            all_new.append(s)
            if s in goldstones:
                all_gold.append(s)
            if not real:
                cname = conjugate_names[k] if conjugate_names else None
                partner = reg.conjugates.get(s)
                if partner is None:
                    partner, cmap = conjugate_pair(
                        s, cname, tex=tex.get(cname) if cname else None)
                    reg.add_conjugate_pair(s, partner, cmap)
                    result.cmap.update(cmap)
                all_new.append(partner)
                if s in goldstones:
                    all_gold.append(partner)
        reg.add_bosons(all_new, goldstones=all_gold)
        for s, m in zip(new, m2):
            reg.masses[s] = m
    return result


def _flavour_index(model, bases):
    """The single symbolic flavour index the Lagrangian uses on ``bases``."""
    found = set()
    for atom in model.lagrangian.total.atoms(sp.Indexed):
        if atom.base in bases:
            found.update(atom.indices)
    symbolic = {i for i in found if not i.is_number}
    if len(symbolic) != 1 or len(found) != 1:
        raise NotImplementedError(
            f"the fermion legs {[str(b) for b in bases]} must all carry the "
            f"same symbolic flavour index for a mass-basis rotation (found "
            f"{sorted(found, key=str)}); write every term with the canonical "
            f"index feynlag.FL_I")
    return symbolic.pop()


def _diagonalize_fermion(model, block, names, tex, angle, values, register):
    from ..fields import _BAR_PARTNER, bar_partner

    what = f"diagonalize({[str(f) for f in block.fields]})"
    if not block.diagonalizable:
        raise NotImplementedError(f"{what}: {block.note}")
    n = block.size
    M = block.matrix
    if n == 1 and names is None:
        return Diagonalization(block, [], list(block.fields),
                               masses=[M[0, 0]], new_right=list(block.right))
    if names is None or len(names) != n:
        raise ValueError(f"{what}: need {n} names")

    tex = tex or {}

    def base(name):
        return sp.IndexedBase(tex_symbol(name, tex.get(name)))

    newL = [base(f"{nm}L") for nm in names]
    newR = [base(f"{nm}R") for nm in names]
    newLbar = [base(f"{nm}Lbar") for nm in names]
    newRbar = [base(f"{nm}Rbar") for nm in names]
    for comp, bar in list(zip(newL, newLbar)) + list(zip(newR, newRbar)):
        _BAR_PARTNER[comp] = bar
        _BAR_PARTNER[bar] = comp

    oldLbar = [bar_partner(c) for c in block.fields]
    oldRbar = [bar_partner(c) for c in block.right]
    i = _flavour_index(model, set(block.fields) | set(block.right)
                       | set(oldLbar) | set(oldRbar))

    def legs(bases):
        return [b[i] for b in bases]

    if n == 2 and M.free_symbols and values is None:
        if isinstance(angle, (tuple, list)):
            aL, aR = (_angle_symbol(a) for a in angle)
        else:
            aL = aR = None
        rotL, rotR = diagonalize_svd_2x2(M, legs(block.fields),
                                         legs(block.right), legs(newL),
                                         legs(newR), angle_left=aL,
                                         angle_right=aR)
        thL, thR = rotL.angle_solution, rotR.angle_solution
        # masses in terms of the angle symbols when given (the explicit
        # atan forms are correct but unreadable)
        D = rotL.matrix * M * rotR.matrix.T
        masses = [sp.expand(D[0, 0]), sp.expand(D[1, 1])]
        if isinstance(angle, (tuple, list)):
            for a, sol in zip(angle, (thL, thR)):
                _register_angle(model, a, sol)
    elif n == 1:
        rotL = Rotation(legs(block.fields), legs(newL), sp.eye(1))
        rotR = Rotation(legs(block.right), legs(newR), sp.eye(1))
        masses = [M[0, 0]]
    else:
        Mn = _numeric_matrix(M, values, what) if values is not None else M
        if Mn.free_symbols:
            raise NotImplementedError(
                f"{what}: a symbolic {n}×{n} Dirac block has no closed form; "
                f"pass values={{...}} for a numeric biunitary rotation")
        rotL, rotR = diagonalize_svd(Mn, legs(block.fields), legs(block.right),
                                     legs(newL), legs(newR), method="numeric")
        D = rotL.matrix * Mn * rotR.matrix.H
        masses = [sp.re(D[a, a]) for a in range(n)]

    rotations = [rotL, rotL.bar(legs(oldLbar), legs(newLbar)),
                 rotR, rotR.bar(legs(oldRbar), legs(newRbar))]
    if register:
        for r in rotations:
            model.rotate(r)
        reg = model.physical
        for b in newL + newR + newLbar + newRbar:
            if b not in reg.fermions:
                reg.fermions.append(b)
        for nm, m in zip(names, masses):
            reg.masses[nm] = m
    return Diagonalization(block, rotations, newL, masses=masses,
                           new_right=newR)


def diagonalize_block(model, block, names=None, tex=None, angle=None,
                      values=None, rotation=None, order="heavy_first",
                      conjugate_names=None, register=True):
    """Rotate one :class:`MassBlock` to its mass eigenstates.

    The route follows from the block:

    - **1×1**: nothing to rotate; ``names`` optionally renames the state.
    - **n×n with n−1 Goldstones** (or n): the rotation is fixed exactly by
      the Goldstone vectors — Goldstones first, then the physical state.  For
      a real 2×2, ``angle=`` writes it as ``rotation_2x2(angle)`` with
      ``tan(angle)`` attached (the 2HDM ``β``).
    - **real symmetric 2×2 without a Goldstone**: the ``tan 2θ`` route
      (:func:`~feynlag.vacuum.diagonalize.diagonalize_orthogonal_2x2`);
      ``names`` follow the rows of ``rotation_2x2(θ)``.  With ``values=`` the
      numbers choose the branch so the first name is the heavier state
      (``order="light_first"`` for the opposite), and
      ``Diagonalization.branch_valid_for`` records where that holds.
    - **anything else with ``values=``**: a numeric rotation at that point
      (eigenvalues in ``order``).
    - **fermion 2×2** (one flavour per component): the analytic biunitary
      route, registering field- and bar-side rotations for both chiralities;
      ``names=["e1", "e2"]`` creates ``e1L, e1R, e1Lbar, …``;
      ``angle=("thL", "thR")`` names the two angles.

    Every rotation is verified against the mass matrix before it is
    registered.  A symbolic block with no closed form raises
    :class:`NotImplementedError` listing the options.

    Args:
        names: names of the physical states (see the order rules above).
        tex: ``{name: LaTeX}`` for the new symbols.
        angle: name, symbol or Parameter for the mixing angle (a pair for a
            fermion block); a string is registered as an InternalParameter.
        values: ``{parameter: number}`` benchmark point.
        rotation: an explicit matrix ``R`` (``new = R·old``), still checked.
        conjugate_names: names of the antiparticle symbols of a complex
            block (default ``<name>_c``).
        register: register the rotation on the model and record the states
            in ``model.physical`` (default).

    Returns:
        :class:`Diagonalization`.
    """
    if block.kind == "fermion":
        return _diagonalize_fermion(model, block, names, tex, angle, values,
                                    register)
    return _diagonalize_scalar(model, block, names, tex, angle, values,
                               rotation, order, conjugate_names, register)

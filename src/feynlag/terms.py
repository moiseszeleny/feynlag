"""Field-level term constructors: write ``−y L̄ H e_R + h.c.`` as one call.

The component-level route (hand-typed :class:`~feynlag.vertices.bilinear.Bilinear`
atoms, one per gauge component, plus a hand-typed ``h.c.``) stays fully
supported; these helpers are an additive layer that *returns the same SymPy
expression*, so everything downstream (invariance checks, mass matrices,
rotations, vertex extraction) is unchanged.

Public surface:

- :func:`inner`, :func:`eps` — the scalar invariants ``A†B`` and ``Aᵀ(iσ₂)B``.
- :func:`tilde` — marks the conjugate doublet ``H̃ = iσ₂H*`` for :func:`yukawa`.
- :func:`yukawa` — ``−(y F̄_L S F_R + h.c.)`` with the gauge contraction,
  flavour indices and hermitian conjugate done for you.
- :func:`dirac_mass` — ``−(M F̄_L F_R + h.c.)`` (vector-like / bare mass).
- :func:`majorana_mass` — ``−½ M FᵀC F + h.c.``.
- :func:`weinberg` — the dim-5 operator ``c (F ε S)(F ε S) + h.c.``.
- :func:`kinetic` — the canonical kinetic/gauge terms of a list of fields.

The gauge contractions are the ones :mod:`feynlag.suggest` enumerates (that
module imports them from here, so there is a single implementation).  A
contraction outside that set (a triplet scalar, two different SU(2) factors,
mismatched colour) raises :class:`NotImplementedError` rather than guessing —
write such a term at component level.

Only the U(1) charge balance is checked at construction, to give an immediate
and specific error; :meth:`feynlag.lagrangian.Model.check_invariance` remains
the authority on invariance (non-abelian, discrete and global symmetries).
"""

import itertools

import sympy as sp

from .dirac import diracC, diracPL, diracPR
from .fields import Fermion, Scalar, dag
from .groups.gauge import SU2
from .vertices.bilinear import Bilinear, MajoranaBilinear, expand_bilinear

__all__ = ["FL_I", "FL_J", "Tilde", "tilde", "inner", "eps", "yukawa",
           "dirac_mass", "majorana_mass", "weinberg", "kinetic"]

#: canonical flavour-index symbols, the defaults of every constructor.  Using
#: the same symbols in every term is what makes the ``xreplace`` keys of a
#: fermion mass-basis :class:`~feynlag.vacuum.diagonalize.Rotation` match.
FL_I = sp.Symbol("fl_i", integer=True)
FL_J = sp.Symbol("fl_j", integer=True)


# --------------------------------------------------------------------------
# Representation bookkeeping (shared with feynlag.suggest)
# --------------------------------------------------------------------------

def _u1_groups(gauge_groups):
    return [g for g in gauge_groups if getattr(g, "abelian", False)]


def _nonabelian_signature(field, gauge_groups):
    """Rep dimension of ``field`` under each non-abelian group (1 = singlet)."""
    sig = []
    for g in gauge_groups:
        if getattr(g, "abelian", False):
            continue
        rep = field.reps.get(g)
        sig.append((g, g.rep_dim(rep) if rep is not None else 1))
    return tuple(sig)


def _is_su2_doublet(field, gauge_groups):
    """True iff ``field`` is a 2 of exactly one SU(2) and a singlet of every
    other non-abelian group."""
    su2s = [g for g in gauge_groups if isinstance(g, SU2)]
    doublet = [g for g in su2s if field.reps.get(g) == 2]
    if len(doublet) != 1:
        return False
    sig = dict(_nonabelian_signature(field, gauge_groups))
    return all(d == 1 for g, d in sig.items() if not isinstance(g, SU2))


def _charge_is_zero(charges, u1s):
    """Every U(1) charge in ``charges`` vanishes, group by group (a sum over
    the groups would accept charges (+1, −1) under two U(1)s)."""
    return all(sp.simplify(charges.get(g, sp.S.Zero)) == 0 for g in u1s)


def _nonabelian_reps(field, gauge_groups):
    """Ordered ``[(group, dim), ...]`` of the non-abelian reps a field
    actually carries (singlets omitted), in the field's own declaration
    order — the order :meth:`~feynlag.fields.Field.generators` uses to build
    the Kronecker product, hence the order of the flat component index."""
    out = []
    for g in field.reps:
        if getattr(g, "abelian", False):
            continue
        d = g.rep_dim(field.reps[g])
        if d > 1:
            out.append((g, d))
    return out


def _flat_index(field, local, gauge_groups):
    """Flat component index of ``field`` from per-group local indices.

    ``local`` maps a non-abelian group to its index within that group's rep;
    groups absent from ``local`` (singlets) contribute index 0.  Mixed-radix
    in the field's declaration order — matching the Kronecker product in
    :meth:`~feynlag.fields.Field.generators` / the flat ``components`` list.
    """
    idx = 0
    for g, d in _nonabelian_reps(field, gauge_groups):
        idx = idx * d + local.get(g, 0)
    return idx


def _spectator_iter(reps):
    """Cartesian product over a list of ``(group, dim)`` index ranges,
    yielding ``{group: local_index}`` dicts (one for the empty list)."""
    groups = [g for g, _ in reps]
    ranges = [range(d) for _, d in reps]
    for combo in itertools.product(*ranges):
        yield dict(zip(groups, combo))


def _doublet_setup(FL, FR, gauge_groups):
    """``(su2, spectators)`` when ``F̄_L · (doublet scalar) · F_R`` can be
    contracted, else ``None``.

    ``F_L`` must be a 2 of exactly one SU(2) that ``F_R`` is a singlet of, and
    every other non-abelian rep (the *spectators*, e.g. colour) must match
    between the two so it can be δ-summed.
    """
    su2s = [g for g in gauge_groups if isinstance(g, SU2)]
    fl_su2 = [g for g in su2s if FL.reps.get(g) == 2]
    if len(fl_su2) != 1 or any(FR.reps.get(g) == 2 for g in su2s):
        return None
    su2 = fl_su2[0]
    spec = [(g, d) for g, d in _nonabelian_reps(FL, gauge_groups)
            if g is not su2]
    if spec != _nonabelian_reps(FR, gauge_groups):
        return None
    return su2, spec


# --------------------------------------------------------------------------
# Coupling-free gauge contractions
# --------------------------------------------------------------------------

def contract_bare(FL, FR, i, j):
    """``F̄_L P_R F_R`` with every gauge index δ-contracted (matched reps)."""
    return sum((Bilinear(FL.bar_components[k][i], diracPR, FR.components[k][j])
                for k in range(FL.dim)), sp.S.Zero)


def contract_direct(FL, S, FR, i, j, su2, spec, gauge_groups):
    """``F̄_L · S F_R``: the doublet scalar saturates ``F_L``'s SU(2) index;
    the spectator reps (colour, …) are summed diagonally."""
    total = sp.S.Zero
    for sp_idx in _spectator_iter(spec):
        fr = FR.components[_flat_index(FR, sp_idx, gauge_groups)]
        for a in (0, 1):
            fl = FL.bar_components[_flat_index(
                FL, {su2: a, **sp_idx}, gauge_groups)]
            total += S.components[a] * Bilinear(fl[i], diracPR, fr[j])
    return total


def contract_tilde(FL, S, FR, i, j, su2, spec, gauge_groups):
    """``F̄_L · S̃ F_R`` with ``S̃ = iσ₂S*`` (``S̃ = (S₂*, −S₁*)``)."""
    eps_ab = {(0, 1): 1, (1, 0): -1}
    total = sp.S.Zero
    for sp_idx in _spectator_iter(spec):
        fr = FR.components[_flat_index(FR, sp_idx, gauge_groups)]
        for a, b in ((0, 1), (1, 0)):
            fl = FL.bar_components[_flat_index(
                FL, {su2: a, **sp_idx}, gauge_groups)]
            total += (eps_ab[(a, b)] * sp.conjugate(S.components[b])
                      * Bilinear(fl[i], diracPR, fr[j]))
    return total


def contract_weinberg(F, S, i, j):
    """``(ε_ab F^a S^b)(ε_cd F^c S^d)`` with ``C P_L`` between the fermions."""
    CPL = diracC * diracPL
    # (εX)_a = ε_ab X^b with ε_12 = 1, ε_21 = −1  →  (εX) = [X^2, −X^1]
    epsS = [S.components[1], -S.components[0]]
    total = sp.S.Zero
    for a in (0, 1):
        for c in (0, 1):
            total += epsS[a] * epsS[c] * MajoranaBilinear(
                F.components[a][i], CPL, F.components[c][j])
    return total


# --------------------------------------------------------------------------
# Scalar invariants
# --------------------------------------------------------------------------

class Tilde:
    """Marker for the conjugate doublet ``S̃ = iσ₂S*`` (or ``S*`` for a gauge
    singlet), built by :func:`tilde` and read by :func:`yukawa`.

    It is not a field: never list it in ``Model(fields=...)``.
    """

    def __init__(self, scalar):
        if not isinstance(scalar, Scalar):
            raise TypeError(f"tilde() takes a Scalar, got {scalar!r}")
        self.scalar = scalar

    def __repr__(self):
        return f"tilde({self.scalar.name})"


def tilde(scalar):
    """The conjugate doublet ``H̃ = iσ₂H*``, for ``yukawa(y, Q, tilde(H), u)``."""
    return Tilde(scalar)


def _groups_of(*fields):
    """Ordered union of the gauge groups the given fields are charged under."""
    out = []
    for f in fields:
        for g in f.reps:
            if g not in out:
                out.append(g)
    return out


def inner(A, B):
    """The invariant ``A†B`` of two scalars in the same representation.

    ``inner(H, H)`` is ``H†H``; ``inner(H1, H2)`` is ``H1†H2`` (add
    ``hc(...)`` yourself where the potential needs it).
    """
    for f in (A, B):
        if not isinstance(f, Scalar):
            raise TypeError(f"inner() takes Scalars, got {f!r}")
    groups = _groups_of(A, B)
    if _nonabelian_signature(A, groups) != _nonabelian_signature(B, groups):
        raise ValueError(
            f"inner({A.name}, {B.name}): the two fields are in different "
            f"non-abelian representations, so {A.name}†{B.name} is not "
            f"gauge invariant")
    return sp.expand((dag(A) * B.mat)[0])


def eps(A, B):
    """The SU(2) invariant ``Aᵀ(iσ₂)B = A₁B₂ − A₂B₁`` of two doublets.

    It carries U(1) charge ``q_A + q_B`` and vanishes for ``A is B``.
    """
    for f in (A, B):
        if not isinstance(f, Scalar):
            raise TypeError(f"eps() takes Scalars, got {f!r}")
    groups = _groups_of(A, B)
    if not (_is_su2_doublet(A, groups) and _is_su2_doublet(B, groups)):
        raise ValueError(
            f"eps({A.name}, {B.name}): both fields must be SU(2) doublets "
            f"(of the same SU(2), singlets of everything else non-abelian)")
    a0, a1 = A.components
    b0, b1 = B.components
    return sp.expand(a0 * b1 - a1 * b0)


# --------------------------------------------------------------------------
# Fermion terms
# --------------------------------------------------------------------------

def _symbol(coupling):
    """A Parameter → its symbol; anything else is sympified as is."""
    return sp.sympify(getattr(coupling, "symbol", coupling))


def _coupling_and_indices(coupling, indices):
    """``(coefficient, i, j)`` for a scalar or flavour-matrix coupling.

    An :class:`~sympy.IndexedBase` coupling becomes ``Y[i, j]`` with two
    leg indices; any other coupling is flavour-universal and both legs share
    one index.  ``indices`` overrides the canonical :data:`FL_I`/:data:`FL_J`
    (one index, or a pair; symbols or integers).

    Raises ``ValueError`` for an explicit pair of *different* indices on a
    flavour-universal coupling (it would be silently put on ``(i, i)``:
    FG-9), and for a flavour matrix with one *symbolic* index on both legs
    (that is the diagonal term ``Y[i,i] F̄_i F_i``, not the matrix).  An
    integer pair ``(a, a)`` is the ordinary diagonal element ``Y[a, a]``
    (FG-11).
    """
    explicit_pair = isinstance(indices, (tuple, list))
    if indices is None:
        i, j = FL_I, FL_J
    elif explicit_pair:
        i, j = (sp.sympify(k) for k in indices)
    else:
        i = j = sp.sympify(indices)
    if isinstance(coupling, sp.IndexedBase):
        if i == j and not i.is_Integer:
            raise ValueError(
                f"the flavour matrix {coupling} needs two distinct leg "
                f"indices, got {i} twice (an integer pair (a, a) is the "
                f"diagonal element {coupling}[a, a])")
        return coupling[i, j], i, j
    if explicit_pair and i != j:
        raise ValueError(
            f"the coupling {coupling} is flavour-universal (δ_ij), so the leg "
            f"indices ({i}, {j}) cannot differ: pass one index, or an "
            f"IndexedBase flavour matrix for a term with two leg indices")
    return _symbol(coupling), i, i


def _check_chiral_pair(what, FL, FR):
    for f in (FL, FR):
        if not isinstance(f, Fermion):
            raise TypeError(f"{what}: expected a fermion field, got {f!r}")
    if FL.chirality != "L" or FR.chirality != "R":
        raise ValueError(
            f"{what}: the first fermion is the barred left-handed field and "
            f"the second the right-handed one (the term is F̄_L … F_R + h.c.); "
            f"got {FL.name} ({FL.chirality}) and {FR.name} ({FR.chirality})")


def _charge_sums(charges, u1s):
    return {g: sp.simplify(charges.get(g, sp.S.Zero)) for g in u1s}


def _with_hc(expr, hc):
    if hc:
        expr = expr + sp.conjugate(sp.expand(expr))
    return expand_bilinear(sp.expand(expr))


def yukawa(coupling, FL, S, FR, indices=None, hc=True):
    """The Yukawa term ``−(y F̄_L S F_R + h.c.)``.

    The gauge contraction is chosen from the representations:

    - ``F_L`` an SU(2) doublet, ``F_R`` an SU(2) singlet, ``S`` a doublet:
      ``F̄_L · S F_R`` (``S``) or ``F̄_L · S̃ F_R`` (``tilde(S)``), any other
      non-abelian index (colour) summed diagonally;
    - ``F_L`` and ``F_R`` in the same representation, ``S`` a gauge singlet:
      ``S F̄_L F_R`` (``tilde(S)`` gives ``S* F̄_L F_R``).

    Args:
        coupling: a Parameter/symbol (flavour-universal, both legs share one
            flavour index) or an :class:`~sympy.IndexedBase` flavour matrix
            ``Y`` (the term is ``Y[i,j] F̄_{L,i} S F_{R,j}``).
        FL, FR: the left- and right-handed :class:`~feynlag.fields.WeylFermion`.
        S: a :class:`~feynlag.fields.Scalar` or ``tilde(scalar)``.
        indices: flavour index or ``(bar, field)`` pair, symbols or
            integers; default :data:`FL_I`, :data:`FL_J`.  A
            flavour-universal coupling takes one index (a pair must repeat
            it); a flavour matrix takes two distinct symbols, or integers
            (``(a, a)`` is the diagonal element ``Y[a, a]``).
        hc: add the hermitian conjugate (default).  Pass ``False`` only to
            inspect the bare operator.

    Raises:
        ValueError: the U(1) charges do not sum to zero (the message gives
            each sum and says when the other of ``S``/``tilde(S)`` balances);
            or the ``indices`` contradict the coupling (see above).
        NotImplementedError: no supported gauge contraction for these reps.
    """
    conj = isinstance(S, Tilde)
    scalar = S.scalar if conj else S
    if not isinstance(scalar, Scalar):
        raise TypeError(f"yukawa: expected a Scalar or tilde(Scalar), got {S!r}")
    what = f"yukawa({FL.name}̄ {S!r} {FR.name})" if conj else \
        f"yukawa({FL.name}̄ {scalar.name} {FR.name})"
    _check_chiral_pair(what, FL, FR)
    c, i, j = _coupling_and_indices(coupling, indices)

    groups = _groups_of(FL, scalar, FR)
    u1s = _u1_groups(groups)
    sign = -1 if conj else 1

    def charges(s):
        return {g: -FL.charge(g) + FR.charge(g) + s * scalar.charge(g)
                for g in u1s}

    scalar_sig = _nonabelian_signature(scalar, groups)
    setup = _doublet_setup(FL, FR, groups)
    if setup is not None and _is_su2_doublet(scalar, groups) \
            and scalar.reps.get(setup[0]) == 2:
        su2, spec = setup
        build = contract_tilde if conj else contract_direct
        op = build(FL, scalar, FR, i, j, su2, spec, groups)
    elif (all(d == 1 for _, d in scalar_sig)
          and _nonabelian_signature(FL, groups)
          == _nonabelian_signature(FR, groups)):
        s0 = scalar.components[0]
        op = (sp.conjugate(s0) if conj else s0) * contract_bare(FL, FR, i, j)
    else:
        raise NotImplementedError(
            f"{what}: no supported gauge contraction for these "
            f"representations. Supported: doublet F_L × doublet scalar × "
            f"SU(2)-singlet F_R (other non-abelian reps equal), or F_L and "
            f"F_R in the same representation with a gauge-singlet scalar. "
            f"Write this term at component level with Bilinear.")

    q = _charge_sums(charges(sign), u1s)
    if any(v != 0 for v in q.values()):
        other = _charge_sums(charges(-sign), u1s)
        bad = ", ".join(f"{g.name} sums to {v}" for g, v in q.items() if v != 0)
        hint = ""
        if all(v == 0 for v in other.values()) and not scalar.real:
            fix = scalar.name if conj else f"tilde({scalar.name})"
            hint = f" The charges balance with {fix} instead."
        raise ValueError(
            f"{what} is not U(1) invariant: {bad} (−q[{FL.name}] + "
            f"q[{'conj ' if conj else ''}{scalar.name}] + q[{FR.name}] must "
            f"vanish).{hint}")

    return _with_hc(-c * op, hc)


def dirac_mass(mass, FL, FR, indices=None, hc=True):
    """The bare Dirac mass ``−(M F̄_L F_R + h.c.)``.

    Needs ``F_L`` and ``F_R`` in the same gauge representation (a vector-like
    fermion, modelled as two :class:`~feynlag.fields.WeylFermion`\\ s).
    ``mass`` is a Parameter/symbol or an :class:`~sympy.IndexedBase` matrix,
    as in :func:`yukawa`.
    """
    what = f"dirac_mass({FL.name}̄ {FR.name})"
    _check_chiral_pair(what, FL, FR)
    c, i, j = _coupling_and_indices(mass, indices)
    groups = _groups_of(FL, FR)
    if _nonabelian_signature(FL, groups) != _nonabelian_signature(FR, groups):
        raise ValueError(
            f"{what}: {FL.name} and {FR.name} are in different non-abelian "
            f"representations, so a bare mass is not gauge invariant (a "
            f"chiral fermion gets its mass from a Yukawa term)")
    u1s = _u1_groups(groups)
    q = _charge_sums({g: FR.charge(g) - FL.charge(g) for g in u1s}, u1s)
    if any(v != 0 for v in q.values()):
        bad = ", ".join(f"{g.name}: q[{FR.name}] − q[{FL.name}] = {v}"
                        for g, v in q.items() if v != 0)
        raise ValueError(f"{what} is not U(1) invariant: {bad}")
    return _with_hc(-c * contract_bare(FL, FR, i, j), hc)


def majorana_form(F, gauge_groups=None):
    """The invariant symmetric form ``B`` of a bare Majorana mass ``FᵀBF``.

    ``Σ_ab B_ab F_aᵀ C F_b`` is invariant iff ``Tᵀ B + B T = 0`` for every
    generator ``T`` of every non-abelian group acting on ``F``
    (:meth:`~feynlag.fields.Field.generators`; ``δF = iαTF``).  It must be
    symmetric, since ``F_aᵀ C P F_b`` is symmetric in ``a, b``.  Such a ``B``
    exists iff the representation is **real**: a gauge singlet (``[[1]]``),
    the SU(2) triplet in the adjoint basis (``δ_ab``), … — not a pseudo-real
    doublet (its ``ε`` is antisymmetric) or a complex SU(3) triplet.

    ``B`` is normalised to ``B†B = 1``, so the mass eigenvalue is the
    coupling, with the phase making the first non-zero diagonal entry (else
    the first non-zero entry) ``+1``.  U(1) charges are not looked at here.

    Args:
        F: the fermion.
        gauge_groups: the groups to impose (default: those ``F`` carries).

    Returns:
        the ``dim × dim`` Matrix ``B``, or ``None`` if the representation is
        not real.
    """
    groups = list(F.reps) if gauge_groups is None else list(gauge_groups)
    n = F.dim
    gens = [T for g in groups if not getattr(g, "abelian", False)
            and g in F.reps for T in F.generators(g)]
    if not gens:
        return sp.eye(n)
    unknowns = {}
    B = sp.zeros(n, n)
    for a in range(n):
        for b in range(a, n):
            x = sp.Dummy(f"B{a}{b}")
            unknowns[x] = (a, b)
            B[a, b] = B[b, a] = x
    eqs = []
    for T in gens:
        eqs.extend(e for e in (T.T * B + B * T) if e != 0)
    xs = list(unknowns)
    A, _ = sp.linear_eq_to_matrix(eqs, xs) if eqs else (sp.zeros(0, len(xs)),
                                                         None)
    null = A.nullspace()
    if not null:
        return None
    if len(null) > 1:
        raise NotImplementedError(
            f"majorana_form({F.name}): the invariant form is not unique "
            f"({len(null)} solutions); a reducible field is not supported")
    sol = B.xreplace(dict(zip(xs, null[0])))
    norm = sp.sqrt(sp.simplify((sol.H * sol)[0, 0]))
    sol = (sol / norm).applyfunc(sp.simplify)
    ref = next((sol[k, k] for k in range(n) if sol[k, k] != 0), None)
    if ref is None:
        ref = next(e for e in sol if e != 0)
    sol = (sol * (sp.Abs(ref) / ref)).applyfunc(sp.simplify)
    return sol


def contract_majorana(F, B, i, j, gamma):
    """``Σ_ab B_ab F_a[i]ᵀ Γ F_b[j]`` — the bare Majorana sandwich."""
    comps = F.components
    return sp.Add(*[B[a, b] * MajoranaBilinear(comps[a][i], gamma, comps[b][j])
                    for a in range(len(comps)) for b in range(len(comps))
                    if B[a, b] != 0])


def majorana_mass(mass, F, indices=None, gamma=None, hc=True):
    """The Majorana mass ``−½ M FᵀB C F + h.c.`` of a fermion in a real rep.

    For a gauge singlet ``B = 1`` (``−½ M FᵀCF``); for a real non-singlet
    representation (an SU(2) triplet with ``Y = 0``) ``B`` is its invariant
    symmetric form, :func:`majorana_form` (``δ_ab`` in the adjoint basis:
    ``−½ M Σ^aᵀ C Σ^a``).

    Args:
        mass: a Parameter/symbol (flavour-diagonal) or an
            :class:`~sympy.IndexedBase` symmetric matrix ``M[i,j]``.
        F: the fermion; U(1)-neutral, in a real representation of every
            non-abelian group.
        gamma: the middle structure; default ``diracC*diracPL``, the
            convention :func:`~feynlag.vertices.bilinear.majorana_mass_matrix`
            and the seesaw examples use.

    Raises:
        ValueError: ``F`` carries a U(1) charge, or its representation is not
            real (a doublet's Majorana mass comes from :func:`weinberg`).
    """
    if not isinstance(F, Fermion):
        raise TypeError(f"majorana_mass: expected a fermion, got {F!r}")
    charged = [g.name for g, r in F.reps.items()
               if getattr(g, "abelian", False) and sp.sympify(r) != 0]
    if charged:
        raise ValueError(
            f"majorana_mass({F.name}): the field is charged under "
            f"{', '.join(charged)}, so FᵀCF is not gauge invariant (a "
            f"Majorana mass for a doublet comes from the Weinberg operator)")
    B = majorana_form(F)
    if B is None:
        nonsinglet = [g.name for g, r in F.reps.items()
                      if not getattr(g, "abelian", False) and g.rep_dim(r) > 1]
        raise ValueError(
            f"majorana_mass({F.name}): its representation of "
            f"{', '.join(nonsinglet)} is not real, so no FᵀCF contraction is "
            f"gauge invariant (a Majorana mass for a doublet comes from the "
            f"Weinberg operator)")
    c, i, j = _coupling_and_indices(mass, indices)
    if gamma is None:
        gamma = diracC * diracPL
    op = -sp.Rational(1, 2) * c * contract_majorana(F, B, i, j, gamma)
    return _with_hc(op, hc)


def weinberg(coupling, F, S, indices=None, hc=True):
    """The dim-5 Weinberg operator ``c (F ε S)(F ε S) + h.c.``.

    ``coupling`` is the full Wilson coefficient (``c/Λ``, mass dimension −1);
    after EWSB the Majorana mass is ``m_ν = −c v²/Λ``.  Note the sign: the
    operator enters with ``+c`` (the convention of ``examples/sm_weinberg.py``),
    not with the ``−`` of a Yukawa term.
    """
    if not isinstance(F, Fermion) or not isinstance(S, Scalar):
        raise TypeError("weinberg(coupling, fermion_doublet, scalar_doublet)")
    groups = _groups_of(F, S)
    if not (_is_su2_doublet(F, groups) and _is_su2_doublet(S, groups)):
        raise NotImplementedError(
            f"weinberg({F.name}, {S.name}): both must be doublets of the same "
            f"SU(2) and singlets of every other non-abelian group")
    u1s = _u1_groups(groups)
    q = _charge_sums({g: 2 * F.charge(g) + 2 * S.charge(g) for g in u1s}, u1s)
    if any(v != 0 for v in q.values()):
        bad = ", ".join(f"{g.name}: 2q[{F.name}] + 2q[{S.name}] = {v}"
                        for g, v in q.items() if v != 0)
        raise ValueError(f"weinberg({F.name}, {S.name}) is not U(1) "
                         f"invariant: {bad}")
    c, i, j = _coupling_and_indices(coupling, indices)
    return _with_hc(c * contract_weinberg(F, S, i, j), hc)


def kinetic(*fields, index=None):
    """Canonical kinetic/gauge terms of ``fields`` as ``[(expr, sector)]``.

    A complex scalar gives ``(D_μφ)†(D^μφ)`` and a real one the canonical
    ``½(D_μφ)(D^μφ)`` (sector ``kinetic``); a charged fermion
    gives its gauge current (sector ``yukawa``, where the examples keep
    currents).  The flavour index defaults to :data:`FL_I`.  See
    :meth:`feynlag.lagrangian.Lagrangian.add_kinetic`.
    """
    from .suggest import suggest_kinetic
    terms = suggest_kinetic(fields, flavor_index=FL_I if index is None else index)
    return [(t.expr, t.sector) for t in terms]

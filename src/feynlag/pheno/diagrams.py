"""Amplitude-level diagram objects: open spinor chains, propagators, ``|M|²``.

Tier 1 of ``docs/manual/scattering_roadmap.md``.  The 1→2 decay engine in
:mod:`~feynlag.pheno.amplitudes` never constructs an amplitude — each
squared-amplitude function writes down an already-squared closed form, with
the 1→2 mass-interference terms and the initial-state spin average baked in
as literals for that one topology.  There is nothing there to extend to a
diagram with a *propagator* in the middle, or to a sum over more than one
diagram.  This module builds the amplitude explicitly instead, as a sum of
:class:`Diagram`\\ s, each a product of open fermion :class:`SpinorChain`\\ s
and :class:`BosonPropagator`\\ s — the covariant engine
(:mod:`~feynlag.pheno.lorentz`) is reused unchanged; what is new is
*combining* its output across a propagator and (eventually) across diagrams.

**Interference is Tier 3, shipped.**  :meth:`Amplitude.squared` performs
the full ``Σ_{d,d'} M_d M̄_{d'}`` double sum.  A diagonal ``d = d'`` term
keeps the exact Tier-1/2 code path (:meth:`SpinorChain.trace` /
:meth:`SpinorChain.epsilon_structure`), so every single-diagram result is
unchanged.  An off-diagonal pair is classified by how its fermion lines pair
the external legs:

- **same pairing** (e.g. γ and Z both in the s channel) — the cross term is
  still a product of two open traces, one per line, each now carrying ``Γ``
  from ``d`` and ``Γ̄`` from ``d'`` (:meth:`SpinorChain.cross_trace`); the
  ε·ε piece is assembled exactly as in Tier 2;
- **exchanged pairing** (Bhabha's s × t, Møller's t × u) — the two lines
  fuse into **one** trace through all four legs (:func:`_exchange_term`).
  Every Lorentz index of that trace is internal (contracted with one of the
  two propagator numerators), so its γ₅ part reduces to ε tensors of the
  diagram's own momenta, and a vanishing Gram determinant over every 4-subset
  (:func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes`) proves it
  zero before it is dropped.

Relative signs are *data*, not inferred here: the fermion-permutation sign
of each diagram rides in :attr:`Diagram.coefficient` (the topology
enumerator :mod:`~feynlag.pheno.topology` derives it), and the propagator's
own ``±i`` comes from :meth:`BosonPropagator.phase` — both cancel in a
diagonal term, so Tier 1–2 never needed them.

**The ε (γ₅) term is Tier 2, shipped.**  Each chain's own trace never needs
it — the diagonal-trace split in :meth:`SpinorChain.trace` proves it drops
for a chain's own two legs, the same fact
:func:`~feynlag.pheno.lorentz.reduce_projectors` proves for the 1→2 engine.
But when *two* chiral chains meet through a propagator, the product of
their ε coefficients (:meth:`ChainVertex.epsilon_coefficient`) is a
genuine, non-zero contribution to ``|M|²`` — the forward–backward-asymmetry
term — which :meth:`SpinorChain.epsilon_structure` exposes and
:meth:`Amplitude.squared` assembles via
:func:`~feynlag.pheno.epsilon.epsilon_pair_tensor`, after
:func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes` proves the
remaining *single*-ε cross terms (one chain's ε piece times the other's
ordinary trace) vanish for this diagram's own momenta.
"""

import itertools
from dataclasses import dataclass

import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix, LorentzIndex
from sympy.tensor.tensor import TensAdd, TensExpr, TensorHead

from .epsilon import (
    assert_epsilon_single_vanishes, epsilon_pair_tensor, gamma5_trace_coefficient,
)
from .lorentz import contract_to_dots, dirac_trace, index, slashed
from .propagator import (
    breit_wigner, propagator_denominator, vector_propagator_numerator,
)

__all__ = [
    "Amplitude", "BosonPropagator", "ChainVertex", "Diagram", "Leg",
    "SpinorChain",
]


def _chain_indices(tag):
    """Deterministic external ``(Γ, Γ̄)`` Lorentz-index pair for a chain
    tagged ``tag``.

    :func:`~feynlag.pheno.lorentz.contract_to_dots` requires every index name
    to appear exactly twice; two chains meeting at one propagator need four
    distinct external names (this chain's Γ/Γ̄ and the other chain's Γ/Γ̄), so
    callers must tag each chain distinctly (e.g. ``"in"``/``"out"``).
    """
    return index(f"mu_{tag}"), index(f"mup_{tag}")


_fresh_counter = itertools.count()


def _fresh_index(tag):
    """A never-before-used Lorentz index (same reasoning as
    :func:`~feynlag.pheno.epsilon._fresh_dummy`: two traces multiplied into
    one term must not share a dummy name)."""
    return index(f"{tag}_{next(_fresh_counter)}")


def _flip(chirality):
    return {"L": "R", "R": "L", None: None}[chirality]


def _expand_chiral_string(factors):
    """Expand a cyclic Dirac string into projector-free strings.

    ``factors`` is a sequence of

    - ``('S', leg)`` — the spin sum ``p̸ ± m`` of a :class:`Leg`;
    - ``('G', atom, w_L, w_R)`` — a vertex factor ``atom·(w_L P_L + w_R P_R)``
      with ``atom`` ``None`` (a scalar vertex), a Lorentz ``TensorIndex``
      (a ``γ^atom``), or an explicit slot ``('i', TensorIndex)`` /
      ``('m', TensorHead)`` (a ``γ`` with that index / a slash).

    Every spin sum is split into its slash and mass pieces and every vertex
    into its two chiral pieces; each projector is then pushed to the right
    end (``P γ = γ P̄`` past a γ, unchanged past a mass, ``P P' = δ_{PP'} P``)
    and the surviving string ``X·P_χ`` is split with
    ``Tr[X P_{L,R}] = ½Tr[X] ∓ ½Tr[Xγ₅]``.  This is the same algebra
    :meth:`SpinorChain.trace` hand-derives for one ``Γ``/``Γ̄`` pair, done
    mechanically so it covers any string.

    Returns:
        ``{atoms: (c_tr, c_g5)}`` — ``atoms`` the γ string as a tuple of
        ``('m', TensorHead)`` (a slash) / ``('i', TensorIndex)`` (a ``γ^μ``)
        slots, ``c_tr`` the coefficient of ``Tr[atoms]`` and ``c_g5`` that of
        ``Tr[atoms·γ₅]``.  Odd-length strings are dropped (zero trace).
    """
    states = [((), None, sp.S.One)]
    for factor in factors:
        new = []
        if factor[0] == "S":
            leg = factor[1]
            sigma = -1 if leg.anti else 1
            for atoms, chi, c in states:
                new.append((atoms + (("m", leg.momentum),), _flip(chi), c))
                if leg.mass != 0:
                    new.append((atoms, chi, c * sigma * leg.mass))
        elif factor[0] == "G":
            _, atom, w_left, w_right = factor
            if atom is not None and not isinstance(atom, tuple):
                atom = ("i", atom)
            for atoms, chi, c in states:
                if atom is not None:
                    atoms, chi = atoms + (atom,), _flip(chi)
                for proj, w in (("L", w_left), ("R", w_right)):
                    if w == 0 or (chi is not None and chi != proj):
                        continue
                    new.append((atoms, proj, c * w))
        else:
            raise ValueError(f"_expand_chiral_string: unknown factor {factor!r}")
        states = new

    out = {}
    for atoms, chi, c in states:
        if len(atoms) % 2:
            continue
        c_tr, c_g5 = out.get(atoms, (sp.S.Zero, sp.S.Zero))
        if chi is None:
            c_tr += c
        else:
            c_tr += c / 2
            c_g5 += c / 2 if chi == "R" else -c / 2
        out[atoms] = (c_tr, c_g5)
    return out


def _string_tensor(atoms):
    """The γ-matrix product for an ``atoms`` tuple (fresh slash dummies).

    Raises:
        ValueError: a momentum is slashed twice.  SymPy 1.14's ``gamma_trace``
            returns a **wrong** result for such a string
            (``Tr[q̸γ^a p̸p̸γ^b q̸]g_ab`` comes out ``72p²q²``, not ``16p²q²``;
            checked directly), so it is refused here rather than traced.  A
            line's own cross trace never repeats a head; the exchange trace,
            where a massive numerator's ``q̸`` meets a spin sum, goes through
            :func:`_scalar_trace` instead.
    """
    heads = [value for kind, value in atoms if kind == "m"]
    if len(set(heads)) != len(heads):
        raise ValueError(
            f"_string_tensor: repeated momentum in {atoms}; SymPy's "
            f"gamma_trace mis-evaluates such strings")
    expr = sp.S.One
    for kind, value in atoms:
        if kind == "i":
            expr = expr * GammaMatrix(value)
        else:
            expr = expr * slashed(value, _fresh_index("sl"))
    return expr


def _contract_pairs(atoms):
    """Remove every contracted ``γ^ρ … γ_ρ`` pair from a trace string.

    The four-dimensional identities, valid for any γ-vectors ``a_i`` (so
    further index atoms inside the pair are fine — they are removed in turn):

    - ``γ^ρ γ_ρ = 4``;
    - ``γ^ρ a₁⋯a_n γ_ρ = −2 a_n⋯a₁`` for odd ``n``;
    - ``γ^ρ a₁⋯a_n γ_ρ = 2(a_n a₁⋯a_{n−1} + a_{n−1}⋯a₁ a_n)`` for even
      ``n ≥ 2``.

    Returns:
        ``[(coefficient, atoms)]`` with only ``('m', head)`` atoms left.

    Raises:
        ValueError: an index without its contracted partner (a free index —
            this helper is only for fully contracted strings).
    """
    position = next((k for k, atom in enumerate(atoms) if atom[0] == "i"), None)
    if position is None:
        return [(sp.S.One, tuple(atoms))]
    rho = atoms[position][1]
    partner = next((j for j in range(position + 1, len(atoms))
                    if atoms[j][0] == "i" and atoms[j][1] == -rho), None)
    if partner is None:
        raise ValueError(f"_contract_pairs: index {rho} has no contracted partner")
    head = tuple(atoms[:position])
    inner = tuple(atoms[position + 1:partner])
    tail = tuple(atoms[partner + 1:])
    n = len(inner)
    if n == 0:
        pieces = [(sp.Integer(4), head + tail)]
    elif n % 2:
        pieces = [(sp.Integer(-2), head + inner[::-1] + tail)]
    else:
        pieces = [(sp.Integer(2), head + (inner[-1],) + inner[:-1] + tail),
                  (sp.Integer(2), head + inner[:-1][::-1] + (inner[-1],) + tail)]
    out = []
    for coeff, piece in pieces:
        for c2, reduced in _contract_pairs(piece):
            out.append((coeff * c2, reduced))
    return out


def _slash_trace(heads, dot, memo):
    """``Tr[h̸₁⋯h̸_n] = Σ_k (−1)^k (h₁·h_k) Tr[h̸₂⋯ĥ_k⋯h̸_n]`` — straight to
    dot products, memoised on the head tuple."""
    if len(heads) % 2:
        return sp.S.Zero
    if not heads:
        return sp.Integer(4)
    if heads in memo:
        return memo[heads]
    first, rest = heads[0], heads[1:]
    total = sp.S.Zero
    for k, other in enumerate(rest):
        sign = 1 if k % 2 == 0 else -1
        total += sign * dot(first, other) * _slash_trace(rest[:k] + rest[k + 1:],
                                                          dot, memo)
    memo[heads] = sp.expand(total)
    return memo[heads]


def _scalar_trace(factors, dot):
    """γ₅-free ``Tr[factors]`` for a **fully contracted** string, as a scalar.

    :func:`_expand_chiral_string` → :func:`_contract_pairs` →
    :func:`_slash_trace`: no SymPy tensor objects at all, so neither of the
    two SymPy 1.14 defects this route would otherwise meet (a repeated
    slashed momentum, see :func:`_string_tensor`; ``kahane_simplify``'s
    ``Repeated index`` error on interleaved contractions such as
    ``γ^aγ^b k̸₃ γ_a k̸₄ γ_b``) can occur.  The γ₅ coefficients are
    discarded — callers must have proven them zero.
    """
    memo, total = {}, sp.S.Zero
    for atoms, (c_tr, _) in _expand_chiral_string(factors).items():
        if c_tr == 0:
            continue
        for coeff, reduced in _contract_pairs(atoms):
            total += c_tr * coeff * _slash_trace(
                tuple(value for _, value in reduced), dot, memo)
    return sp.expand(total)


def _chiral_trace(factors):
    """``Tr[factors]`` split into its γ₅-free and γ₅ parts (SymPy-traced).

    For one line's open cross trace (:meth:`SpinorChain.cross_trace`) — at
    most four γ's, distinct momenta.  The fully contracted exchange trace
    goes through :func:`_scalar_trace` instead.

    Returns:
        ``(plain, eps4)``: ``plain`` the γ₅-free trace (a tensor open on the
        free ``'i'`` atoms); ``eps4`` a list of ``(prefactor, slots)`` for the
        4-γ strings, with ``prefactor = c_g5·κ`` (``Tr[abcdγ₅] = κ ε^{abcd}``,
        :func:`~feynlag.pheno.epsilon.gamma5_trace_coefficient`) and
        ``slots`` in :func:`~feynlag.pheno.epsilon.epsilon_pair_tensor`
        format.

    Raises:
        NotImplementedError: a ≥6-γ string with a γ₅ part — no ε reduction
            is implemented for it.
    """
    plain, eps4 = sp.S.Zero, []
    for atoms, (c_tr, c_g5) in _expand_chiral_string(factors).items():
        if c_tr != 0:
            plain += c_tr * dirac_trace(_string_tensor(atoms))
        if c_g5 != 0 and len(atoms) == 4:
            eps4.append((c_g5 * gamma5_trace_coefficient(), atoms))
        elif c_g5 != 0 and len(atoms) > 4:
            raise NotImplementedError(
                f"_chiral_trace: a γ₅ trace of {len(atoms)} γ's — no ε "
                f"reduction is implemented beyond four")
    return plain, eps4


def _gamma_factor(vertex, atom):
    """``('G', …)`` factor for ``Γ`` of ``vertex`` (``atom`` its γ index)."""
    return ("G", atom if vertex.structure == "V" else None,
            vertex.g_left, vertex.g_right)


@dataclass(frozen=True)
class Leg:
    """One external fermion leg of an open spinor chain.

    Args:
        momentum: the momentum ``TensorHead`` (e.g. ``kin.k1`` from a
            :class:`~feynlag.pheno.kinematics.TwoToTwoKinematics`).
        mass: the leg's mass.
        anti: ``True`` for an antiparticle leg (``Σ v v̄ = p̸ − m``),
            ``False`` for a particle leg (``Σ u ū = p̸ + m``) — this is a
            property of the *particle*, not of whether the leg is incoming or
            outgoing; the incoming/outgoing distinction is instead encoded by
            which of a chain's two legs the caller assigns to ``out``
            (the ψ̄/bar slot) vs. ``inn`` (the ψ/field slot), following the
            standard external-fermion-line table (incoming particle / outgoing
            antiparticle → field slot; outgoing particle / incoming
            antiparticle → bar slot).
    """

    momentum: object
    mass: sp.Expr
    anti: bool = False

    def __post_init__(self):
        object.__setattr__(self, "mass", sp.sympify(self.mass))


@dataclass(frozen=True)
class ChainVertex:
    """One fermion-line vertex: ``Γ = g_L P_L + g_R P_R`` (structure ``'S'``,
    a scalar/pseudoscalar current) or ``Γ = γ^μ(g_L P_L + g_R P_R)``
    (structure ``'V'``, a vector current) — the same two structures
    :func:`~feynlag.pheno.vertices.classify_gamma` classifies a 1→2 vertex
    into.
    """

    structure: str
    g_left: sp.Expr
    g_right: sp.Expr

    def __post_init__(self):
        if self.structure not in ("S", "V"):
            raise NotImplementedError(
                f"ChainVertex: unsupported structure {self.structure!r}; "
                f"only 'S' and 'V' are implemented")
        object.__setattr__(self, "g_left", sp.sympify(self.g_left))
        object.__setattr__(self, "g_right", sp.sympify(self.g_right))

    def bar(self):
        """``Γ̄ = γ⁰Γ†γ⁰``, matching :func:`feynlag.dirac.dirac_conjugate`.

        A ``'V'`` current is self-conjugate structure-wise (``γ^μP_L =
        P_Rγ^μ``): the two effects cancel, so the chirality labels do *not*
        swap, only the couplings conjugate.  A ``'S'`` current has bare
        projectors, which *do* swap (``P̄_L = P_R``), so the returned
        ``g_left``/``g_right`` are conjugated **and exchanged**.
        """
        if self.structure == "V":
            return ChainVertex("V", sp.conjugate(self.g_left),
                               sp.conjugate(self.g_right))
        return ChainVertex("S", sp.conjugate(self.g_right),
                           sp.conjugate(self.g_left))

    def epsilon_coefficient(self):
        """Coefficient of the ``Tr[…γ₅]`` (ε) piece of this vertex's chain.

        Only four-γ terms survive a γ₅ trace.  A ``'S'`` chain has at most
        two explicit γ's (from the two spin sums) and is always zero.  A
        ``'V'`` chain has four, and the chirality-diagonal split
        ``Tr[X P_{L,R}] = ½Tr[X] ∓ ½Tr[Xγ₅]`` leaves
        ``(|g_R|² − |g_L|²)/2`` as the coefficient that
        :func:`~feynlag.pheno.lorentz.reduce_projectors` drops.
        """
        if self.structure == "S":
            return sp.S.Zero
        return (sp.Abs(self.g_right)**2 - sp.Abs(self.g_left)**2) / 2


@dataclass(frozen=True)
class SpinorChain:
    """One open fermion line: ``(bar leg) Γ (field leg)``, spin-summed.

    ``out`` is the ψ̄ (bar) slot, ``inn`` the ψ (field) slot — see
    :class:`Leg`.  For a ``'V'`` vertex, ``indices``/``bar_indices`` must each
    hold exactly one external ``TensorIndex`` (the Γ-side and Γ̄-side
    Lorentz slots respectively, e.g. from :func:`_chain_indices`); for a
    ``'S'`` vertex both must be empty.
    """

    out: Leg
    inn: Leg
    vertices: tuple
    indices: tuple = ()
    bar_indices: tuple = ()

    def epsilon_coefficient(self):
        """This chain's :meth:`ChainVertex.epsilon_coefficient`.

        Raises:
            NotImplementedError: more than one vertex on the line — a
                derivative-coupling / multi-propagator chain, Tier 4.
        """
        if len(self.vertices) != 1:
            raise NotImplementedError(
                "SpinorChain.epsilon_coefficient: more than one vertex on a "
                "line is a derivative-coupling chain, Tier 4 of the "
                "scattering roadmap")
        return self.vertices[0].epsilon_coefficient()

    def epsilon_structure(self):
        """This chain's ε (γ₅) piece as ``(prefactor, slots)``, or ``None``.

        Only the diagonal 4-γ term of a ``'V'`` chain has an ε piece — a
        ``'S'`` chain's :meth:`epsilon_coefficient` is always zero (at most
        two γ's, no room for ``Tr[Xγ₅]``), and a pure-vector ``'V'`` coupling
        (``|g_L|=|g_R|``) also gives zero — so this returns ``None`` in
        exactly the cases where the chain contributes no ε term at all,
        including every case the pre-Tier-2 engine already handled correctly.

        No mass and no :attr:`Leg.anti` sign enters ``prefactor``: the
        chirality-mixing (mass) term of :meth:`trace` has at most two
        explicit γ's and so has no ``Tr[Xγ₅]`` piece either — only the
        diagonal, momentum-slash-only term does.

        Returns:
            ``(prefactor, slots)`` with ``prefactor = epsilon_coefficient()
            · κ`` (:func:`~feynlag.pheno.epsilon.gamma5_trace_coefficient`)
            and ``slots`` the 4-tuple ``(('m', out.momentum), ('i', mu),
            ('m', inn.momentum), ('i', mu_bar))`` for
            :func:`~feynlag.pheno.epsilon.epsilon_pair_tensor` — or ``None``.
        """
        coeff = self.epsilon_coefficient()
        if coeff == 0:
            return None
        prefactor = coeff * gamma5_trace_coefficient()
        mu, mu_bar = self.indices[0], self.bar_indices[0]
        slots = (("m", self.out.momentum), ("i", mu),
                 ("m", self.inn.momentum), ("i", mu_bar))
        return prefactor, slots

    def vertex(self):
        """The chain's single :class:`ChainVertex`.

        Raises:
            NotImplementedError: more than one vertex on the line (Tier 4).
        """
        if len(self.vertices) != 1:
            raise NotImplementedError(
                "SpinorChain: more than one vertex on a line is a "
                "derivative-coupling / multi-propagator chain, Tier 4 of "
                "the scattering roadmap")
        return self.vertices[0]

    def cross_trace(self, partner, index_a, index_b):
        """``Tr[S_out Γ S_inn Γ̄']`` — this chain's ``Γ`` against the ``Γ̄``
        of ``partner``, the same fermion line in *another* diagram.

        The interference analogue of :meth:`trace` + :meth:`epsilon_structure`
        (with ``partner is self`` it reproduces both, a pinned identity), built
        by the mechanical projector algebra of :func:`_chiral_trace` rather
        than hand-derived — so it also covers mixed scalar × vector pairs.

        Args:
            partner: the other diagram's :class:`SpinorChain` on the same
                legs (only its vertex is used).
            index_a, index_b: the Lorentz indices for ``Γ`` / ``Γ̄'`` (used
                only by a ``'V'`` vertex).

        Returns:
            ``(plain, eps4)`` as in :func:`_chiral_trace`.
        """
        partner_bar = partner.vertex().bar()
        plain, eps4 = _chiral_trace((
            ("S", self.out),
            _gamma_factor(self.vertex(), index_a),
            ("S", self.inn),
            _gamma_factor(partner_bar, index_b),
        ))
        return plain, eps4

    def trace(self):
        """The γ₅-free trace for this chain, open on ``indices``/
        ``bar_indices`` for a ``'V'`` vertex (a plain scalar for ``'S'``).

        ``Tr[spin_sum(out)·Γ·spin_sum(inn)·Γ̄]``, following the same
        diagonal/chirality-mixing split as
        :func:`~feynlag.pheno.amplitudes.ffs_squared`/``ffv_squared`` — the
        mass-interference term is a hand-derived literal there and here for
        the identical reason: :mod:`~feynlag.pheno.lorentz` has no explicit
        γ₅/projector object to build ``Γ``/``Γ̄`` from directly.  The diagonal
        term's γ₅-free half is
        ``Tr[X(|g_L|²P_L+|g_R|²P_R)] = ½(|g_L|²+|g_R|²)Tr[X] ∓ ½(...)Tr[Xγ₅]``
        — the ``½`` below is that split's non-ε half; the ``Tr[Xγ₅]`` half is
        :meth:`epsilon_structure`, assembled separately at the
        :class:`Amplitude` level (unlike
        :func:`~feynlag.pheno.lorentz.reduce_projectors`, which this method
        no longer calls, this chain-level engine computes that ε term
        instead of proving it away).  The result is left with its external
        indices **open** — reduction to on-shell dot products happens once,
        at the :class:`Amplitude` level, after combining with the other
        chain and the propagator.

        Raises:
            NotImplementedError: more than one vertex (Tier 4), or a
                structure outside ``{'S', 'V'}``.
        """
        if len(self.vertices) != 1:
            raise NotImplementedError(
                "SpinorChain.trace: more than one vertex on a line is a "
                "derivative-coupling / multi-propagator chain, Tier 4 of "
                "the scattering roadmap")
        vertex = self.vertices[0]
        gL, gR = vertex.g_left, vertex.g_right
        m_out, m_inn = self.out.mass, self.inn.mass
        tag = f"{id(self.out.momentum)}_{id(self.inn.momentum)}"
        d_out, d_inn = index(f"so_{tag}"), index(f"si_{tag}")
        # The DIAGONAL piece (weighted |g_L|²+|g_R|²) uses the bare momentum
        # slash, never the mass — P_L P_R = 0 kills any mass insertion there.
        # The mass-dependent piece survives only in the chirality-MIXING term
        # below (weighted 2Re(g_L ḡ_R)), exactly as in
        # amplitudes.ffs_squared/ffv_squared.  Folding the mass into a single
        # spin_sum() here and weighting the whole thing by |g_L|²+|g_R|² would
        # double-count it for a non-chiral (g_L=g_R) coupling.
        out_slash = slashed(self.out.momentum, d_out)
        inn_slash = slashed(self.inn.momentum, d_inn)
        sign_out = -1 if self.out.anti else 1
        sign_inn = -1 if self.inn.anti else 1
        mixing_mass = 2 * sign_out * m_out * sign_inn * m_inn

        if vertex.structure == "S":
            chain = out_slash * inn_slash
            total = sp.S.Zero
            if gL != 0 or gR != 0:
                total += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * sp.Rational(1, 2) * dirac_trace(chain)
            if gL != 0 and gR != 0:
                total += 2 * sp.re(gL * sp.conjugate(gR)) * mixing_mass
            return sp.expand(total)

        # 'V'
        if len(self.indices) != 1 or len(self.bar_indices) != 1:
            raise ValueError(
                "SpinorChain.trace: a 'V' vertex needs exactly one external "
                "index in each of `indices`/`bar_indices`")
        mu, mu_bar = self.indices[0], self.bar_indices[0]
        total = sp.S.Zero
        if gL != 0 or gR != 0:
            chain = out_slash * GammaMatrix(mu) * inn_slash * GammaMatrix(mu_bar)
            total += (sp.Abs(gL)**2 + sp.Abs(gR)**2) * sp.Rational(1, 2) * dirac_trace(chain)
        if gL != 0 and gR != 0:
            total += (2 * sp.re(gL * sp.conjugate(gR)) * mixing_mass
                     * LorentzIndex.metric(mu, mu_bar))
        return total


@dataclass(frozen=True)
class BosonPropagator:
    """An internal boson line joining two chain vertices.

    Args:
        spin: ``0`` or ``1``.
        q2: the invariant ``q²`` (e.g. ``kin.s``).
        mass, width: as in
            :func:`~feynlag.pheno.propagator.propagator_denominator`.
        momentum: callable ``q(index) -> TensExpr``; required for a massive
            spin-1 propagator (unused for spin 0 or a massless spin-1).
    """

    spin: int
    q2: sp.Expr
    mass: sp.Expr
    width: sp.Expr = sp.S.Zero
    momentum: object = None

    def __post_init__(self):
        if self.spin not in (0, 1):
            raise NotImplementedError(
                f"BosonPropagator: unsupported spin {self.spin}; only 0 and "
                f"1 are implemented")
        object.__setattr__(self, "q2", sp.sympify(self.q2))
        object.__setattr__(self, "mass", sp.sympify(self.mass))
        object.__setattr__(self, "width", sp.sympify(self.width))
        if self.spin == 1 and self.mass != 0 and self.momentum is None:
            raise ValueError(
                "BosonPropagator: a massive spin-1 propagator needs "
                "`momentum` (a callable q(index) -> TensExpr)")

    def denominator_squared_inverse(self):
        """``1/|q²−m²+imΓ|²``, reusing
        :func:`~feynlag.pheno.propagator.breit_wigner` verbatim."""
        return breit_wigner(self.q2, self.mass, self.width)

    def denominator(self):
        """``q² − m² + i m Γ`` (:func:`~feynlag.pheno.propagator.propagator_denominator`)."""
        return propagator_denominator(self.q2, self.mass, self.width)

    def phase(self):
        """The propagator's constant phase: ``+i`` for spin 0 (``i/D``),
        ``−i`` for spin 1 (``−i N_{ab}/D``) — matching
        :func:`~feynlag.pheno.propagator.scalar_propagator`/
        :func:`~feynlag.pheno.propagator.vector_propagator`.  Unit modulus,
        so it cancels in a diagonal term; it matters for the *relative*
        phase of a scalar- and a vector-mediated diagram."""
        return sp.I if self.spin == 0 else -sp.I

    def numerator(self, a, b):
        """The real propagator numerator, both indices returned lowered:
        ``1`` for spin 0; ``g_ab`` (Feynman gauge) for a massless spin 1;
        :func:`~feynlag.pheno.propagator.vector_propagator_numerator`
        (reused verbatim) for a massive spin 1."""
        if self.spin == 0:
            return sp.S.One
        if self.mass == 0:
            return LorentzIndex.metric(-a, -b)
        return vector_propagator_numerator(self.momentum, self.mass, a, b)


@dataclass(frozen=True)
class Diagram:
    """One Feynman diagram: two spinor chains joined by one propagator.

    ``coefficient`` is an overall complex prefactor not already carried by
    either chain's couplings or by the propagator's
    :meth:`~BosonPropagator.phase` — in practice the **fermion-permutation
    sign** (±1) that relates diagrams with different pairings of the external
    fermions (Bhabha's s and t channels differ by −1).  It cancels in a
    single-diagram ``|M|²`` (Tier 1's assemblers leave it at ``1``); in an
    interference term it is the relative sign, so hand-built multi-diagram
    amplitudes must set it (the :mod:`~feynlag.pheno.topology` enumerator
    derives it).
    """

    chains: tuple
    propagators: tuple
    coefficient: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "coefficient", sp.sympify(self.coefficient))


def _legs(chain):
    """A chain's ``(bar-slot, field-slot)`` momentum heads — its pairing."""
    return (chain.out.momentum, chain.inn.momentum)


def _check_topology(diagram):
    if len(diagram.chains) != 2:
        raise NotImplementedError(
            "Amplitude.squared: only the two-chain, single-propagator 2→2 "
            "topology is assembled")
    if len(diagram.propagators) != 1:
        raise NotImplementedError(
            "Amplitude.squared: multi-propagator diagrams are not yet "
            "assembled")


def _numerator(prop, a, b):
    """``prop.numerator`` with the two γ-slot indices, or ``1`` (spin 0)."""
    return prop.numerator(a, b) if prop.spin == 1 else sp.S.One


def _contract(expr):
    if isinstance(expr, TensExpr):
        return expr.contract_metric(LorentzIndex.metric)
    return expr


def _pair_factor(d1, d2):
    """``c₁c̄₂ · η₁η̄₂ / (D₁ D̄₂)`` — every non-trace factor of ``M₁M̄₂``."""
    p1, p2 = d1.propagators[0], d2.propagators[0]
    return (d1.coefficient * sp.conjugate(d2.coefficient)
            * p1.phase() * sp.conjugate(p2.phase())
            / (p1.denominator() * sp.conjugate(p2.denominator())))


@dataclass(frozen=True)
class Amplitude:
    """A sum of :class:`Diagram`\\ s."""

    diagrams: tuple

    def squared(self, kin):
        """The spin/colour-**summed** ``|M|²`` — no averaging (see
        :func:`~feynlag.pheno.scattering.average_factor`).

        ``Σ_d |M_d|² + Σ_{d<d'} 2Re(M_d M̄_{d'})``.  A diagonal term is
        :meth:`_diagonal` — exactly the Tier-1/2 single-diagram computation,
        so a one-diagram amplitude is unchanged bit-for-bit.  An off-diagonal
        term is :func:`_same_pairing_term` (both diagrams pair the external
        fermions into lines the same way) or :func:`_exchange_term` (the
        pairings differ — the lines fuse into one trace); see the module
        docstring.  Each term reaches the scalar level through **one**
        :func:`~feynlag.pheno.lorentz.contract_to_dots` call on a fully
        contracted expression.

        Raises:
            NotImplementedError: a diagram that is not the two-chain,
                one-propagator topology, a pair of diagrams whose pairings
                are neither equal nor exchanged, or a single-ε term that does
                not provably vanish.
        """
        for diagram in self.diagrams:
            _check_topology(diagram)
        total = sp.S.Zero
        for diagram in self.diagrams:
            total += self._diagonal(diagram, kin)
        for d1, d2 in itertools.combinations(self.diagrams, 2):
            term = _interference_term(d1, d2, kin)
            total += term + sp.conjugate(term)
        return sp.expand(total)

    @staticmethod
    def _diagonal(diagram, kin):
        """``|M_d|²`` for one diagram — the Tier-1/2 path, unchanged.

        For a spin-1 mediator, when either chain has a non-zero
        :meth:`~SpinorChain.epsilon_structure`, this assembles the ε (γ₅)
        contribution: it proves the single-ε cross terms vanish for this
        diagram's own momenta
        (:func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes`) and, if
        both chains have a non-zero ε piece, adds the computed ε·ε term
        (:func:`~feynlag.pheno.epsilon.epsilon_pair_tensor`) to the ordinary
        (non-ε) piece before the one ``contract_to_dots`` call.
        """
        chain_a, chain_b = diagram.chains
        tensor_a = chain_a.trace()
        tensor_b = chain_b.trace()
        prop = diagram.propagators[0]

        if prop.spin == 0:
            combined = tensor_a * tensor_b
        else:
            mu, mu_p = chain_a.indices[0], chain_a.bar_indices[0]
            nu, nu_p = chain_b.indices[0], chain_b.bar_indices[0]
            n1 = prop.numerator(mu, nu)
            n2 = prop.numerator(mu_p, nu_p)
            # _contract, not .contract_metric: a zero-coupling line traces
            # to a plain 0, which has no tensor methods.
            combined = _contract(tensor_a * n1 * tensor_b * n2)

            ea = chain_a.epsilon_structure()
            eb = chain_b.epsilon_structure()
            if (ea is not None and ea[0] != 0) or (eb is not None and eb[0] != 0):
                # single-ε cross terms (this chain's ε piece × the other's
                # ordinary trace) exist whenever EITHER coefficient is
                # non-zero, regardless of the other — prove they vanish.
                assert_epsilon_single_vanishes(_momentum_heads(diagram), kin.dot)
            if ea is not None and eb is not None:
                eps_prod = sp.simplify(ea[0] * eb[0])
                if eps_prod != 0:
                    ee = epsilon_pair_tensor(ea[1], eb[1])
                    # stepwise contract_metric (one numerator at a time) is
                    # ~3.5x faster here than contracting both at once.
                    ee = (ee * n1).contract_metric(LorentzIndex.metric).expand()
                    ee = (ee * n2).contract_metric(LorentzIndex.metric)
                    combined = combined + eps_prod * ee

        result = contract_to_dots(combined, kin.dot)
        result *= prop.denominator_squared_inverse()
        result *= sp.Abs(diagram.coefficient)**2
        return sp.expand(result)


def _interference_term(d1, d2, kin):
    """``M₁ M̄₂`` (spin-summed) for two distinct diagrams."""
    a1, b1 = d1.chains
    pairs1 = {_legs(a1), _legs(b1)}
    pairs2 = {_legs(c) for c in d2.chains}
    if pairs1 == pairs2:
        return _same_pairing_term(d1, d2, kin)
    outs = lambda d: {c.out.momentum for c in d.chains}   # noqa: E731
    inns = lambda d: {c.inn.momentum for c in d.chains}   # noqa: E731
    if outs(d1) == outs(d2) and inns(d1) == inns(d2):
        return _exchange_term(d1, d2, kin)
    raise NotImplementedError(
        "Amplitude.squared: two diagrams whose fermion lines do not share "
        "the same external legs — not the same 2→2 process")


def _same_pairing_term(d1, d2, kin):
    """``M₁M̄₂`` when both diagrams pair the legs into the same two lines.

    Each line gives one open cross trace ``Tr[S Γ₁ S Γ̄₂]``
    (:meth:`SpinorChain.cross_trace`); ``d1``'s numerator joins the two ``Γ₁``
    slots and ``d2``'s the two ``Γ̄₂`` slots.  The γ₅ parts are treated as in
    :meth:`Amplitude._diagonal`: single-ε pieces proven zero, the ε·ε piece
    computed.
    """
    chain_a, chain_b = d1.chains
    partner = {_legs(c): c for c in d2.chains}
    p1, p2 = d1.propagators[0], d2.propagators[0]
    ia, ia_bar = _fresh_index("xa"), _fresh_index("xab")
    ib, ib_bar = _fresh_index("xb"), _fresh_index("xbb")
    plain_a, eps_a = chain_a.cross_trace(partner[_legs(chain_a)], ia, ia_bar)
    plain_b, eps_b = chain_b.cross_trace(partner[_legs(chain_b)], ib, ib_bar)
    n1 = _numerator(p1, ia, ib)
    n2 = _numerator(p2, ia_bar, ib_bar)

    combined = _contract(plain_a * n1 * plain_b * n2)
    if eps_a or eps_b:
        heads = _momentum_heads(d1)
        heads += [h for h in _momentum_heads(d2) if h not in heads]
        assert_epsilon_single_vanishes(heads, kin.dot)
    for pref_a, slots_a in eps_a:
        for pref_b, slots_b in eps_b:
            ee = epsilon_pair_tensor(slots_a, slots_b)
            ee = _contract(ee * n1).expand()
            ee = _contract(ee * n2)
            combined = combined + pref_a * pref_b * ee

    result = contract_to_dots(combined, kin.dot)
    return sp.expand(result * _pair_factor(d1, d2))


def _exchange_term(d1, d2, kin):
    """``M₁M̄₂`` when the two diagrams pair the legs differently.

    With ``d1`` lines ``(o₁,i₁)``, ``(o₂,i₂)`` and ``d2`` lines ``(o₁,i₂)``,
    ``(o₂,i₁)``,

    ``M₁M̄₂ = Tr[S_{o₁} Γ_a S_{i₁} Γ̄'_{(o₂,i₁)} S_{o₂} Γ_b S_{i₂} Γ̄'_{(o₁,i₂)}]``

    contracted with ``d1``'s numerator on the ``Γ_a, Γ_b`` slots and ``d2``'s
    on the two ``Γ̄'`` slots (``S`` the spin sums).  Every index is internal,
    so after contraction the γ₅ part is a sum of ε tensors of the diagram's
    own external momenta — proven zero by
    :func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes` before it
    is dropped (a topology where that proof fails raises instead).
    """
    by_out_1 = {c.out.momentum: c for c in d1.chains}
    by_out_2 = {c.out.momentum: c for c in d2.chains}
    chain_a = d1.chains[0]
    o1, i1 = chain_a.out, chain_a.inn
    chain_b = next(c for c in d1.chains if c is not chain_a)
    o2, i2 = chain_b.out, chain_b.inn
    x2 = by_out_2[o1.momentum]          # (o1, i2) in d2
    y2 = by_out_2[o2.momentum]          # (o2, i1) in d2
    if (by_out_1[o1.momentum] is not chain_a or x2.inn.momentum != i2.momentum
            or y2.inn.momentum != i1.momentum):
        raise ValueError("_exchange_term: the two diagrams do not exchange "
                         "the field slots of their fermion lines")

    p1, p2 = d1.propagators[0], d2.propagators[0]

    heads = _momentum_heads(d1)
    heads += [h for h in _momentum_heads(d2) if h not in heads]
    assert_epsilon_single_vanishes(heads, kin.dot)

    # The numerators go *inside* the one trace — the metric as a contracted
    # γ^ρ…γ_ρ pair, q_aq_b as two slashes — and the string is reduced to dot
    # products directly (_scalar_trace), never through SymPy's tensor
    # engine: this route has no ε·ε piece that would need indices open.
    result = sp.S.Zero
    for c1, a_slot, b_slot in _numerator_slots(p1):
        for c2, g_slot, d_slot in _numerator_slots(p2):
            result += c1 * c2 * _scalar_trace((
                ("S", o1), _gamma_factor(chain_a.vertex(), a_slot),
                ("S", i1), _gamma_factor(y2.vertex().bar(), g_slot),
                ("S", o2), _gamma_factor(chain_b.vertex(), b_slot),
                ("S", i2), _gamma_factor(x2.vertex().bar(), d_slot),
            ), kin.dot)
    return sp.expand(result * _pair_factor(d1, d2))


def _momentum_terms(prop):
    """``q = Σ c_h h`` — the propagator momentum as ``[(c_h, head)]``."""
    expr = prop.momentum(index("_momentum_terms_dummy")).expand()
    terms = expr.args if isinstance(expr, TensAdd) else (expr,)
    out = []
    for term in terms:
        coeff = getattr(term, "coeff", sp.S.One)
        (head,) = term.components
        out.append((coeff, head))
    return out


def _numerator_slots(prop):
    """A propagator numerator ``N_{ab}`` as ``[(c, slot_a, slot_b)]`` with
    ``N_{ab}γ^a…γ^b = Σ c·(slot_a)…(slot_b)``: ``(1, None, None)`` for spin
    0; ``(1, γ^ρ, γ_ρ)`` for the metric; and ``(−c_hc_{h'}/m², h̸, h̸')`` for
    each pair of heads of ``q`` in a massive spin-1 ``−q_aq_b/m²``."""
    if prop.spin == 0:
        return [(sp.S.One, None, None)]
    rho = _fresh_index("nr")
    out = [(sp.S.One, ("i", rho), ("i", -rho))]
    if prop.mass != 0:
        for ca, ha in _momentum_terms(prop):
            for cb, hb in _momentum_terms(prop):
                out.append((-ca * cb / prop.mass**2, ("m", ha), ("m", hb)))
    return out


def _momentum_heads(diagram):
    """Every momentum ``TensorHead`` appearing in ``diagram``.

    The four external legs, plus whatever a propagator's own momentum
    callable is built from (e.g. ``k1+k2`` for the s-channel assemblers in
    :mod:`~feynlag.pheno.scattering` — already a subset of the four leg
    momenta today, but this stays generic rather than assuming that).
    """
    heads = []
    for chain in diagram.chains:
        for leg in (chain.out, chain.inn):
            if leg.momentum not in heads:
                heads.append(leg.momentum)
    for prop in diagram.propagators:
        if prop.momentum is not None:
            for head in prop.momentum(index("_momentum_heads_dummy")).atoms(TensorHead):
                if head not in heads:
                    heads.append(head)
    return heads

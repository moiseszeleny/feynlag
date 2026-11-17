"""Scalar-level reduction of fully contracted Dirac traces.

A spin-summed ``|M|²`` is a Dirac trace whose Lorentz indices are all
contracted — with each other, with a metric, or with a momentum.  This module
reduces such an object straight to dot products, without building a SymPy
tensor expression:

1. :func:`_expand_chiral_string` multiplies out spin sums, propagator
   numerators and chiral vertices, pushing every projector to the right end;
2. :func:`_contract_pairs` removes each contracted ``γ^ρ … γ_ρ`` pair with the
   four-dimensional contraction identities;
3. :func:`_slash_trace` evaluates the remaining trace of slashed momenta.

:func:`contract_trace` adds the step in front that the external-boson
processes of Tier 4 need: a trace multiplied by *loose* tensors — metrics and
momentum vectors carrying the same index names as γ's inside the trace (a
polarization sum ``−g_{μμ'} + k_μk_{μ'}/m²``, a triple-gauge vertex
``g^{μν}(p₁−p₂)^ρ + …``).  They are eliminated by renaming (a metric
identifies its two indices; a vector meeting a ``γ`` becomes a slash; two
vectors become a dot product), after which every index sits on a contracted
γ pair and steps 1–3 apply.

Why not SymPy's tensor engine: ``gamma_trace`` (SymPy 1.14) returns a wrong
result for a string that slashes one momentum twice
(``Tr[q̸γ^a p̸p̸γ^b q̸]g_ab`` comes out ``72p²q²``, not ``16p²q²``), and
``kahane_simplify`` raises on interleaved contractions.  A squared
fermion-exchange diagram has both.  Everything here is plain SymPy scalars.

The γ₅ coefficients are computed by step 1 and then **discarded** by
:func:`_scalar_trace` — the caller must have proven the γ₅ part zero (for a
fully contracted 2→2 trace: three independent momenta cannot fill an ε
tensor, :func:`~feynlag.pheno.epsilon.assert_epsilon_single_vanishes`).

Index conventions.  Steps 1–3 take a contracted pair as two ``('i', x)`` atoms
with ``x`` and ``-x`` (a SymPy ``TensorIndex``, or an :class:`Ix`).
:func:`contract_trace` takes plain hashable **names** (strings), each
appearing exactly twice, and assigns the up/down pair itself.
"""

from dataclasses import dataclass

import sympy as sp

__all__ = ["Ix", "contract_tensors", "contract_trace"]


@dataclass(frozen=True)
class Ix:
    """A Lorentz index for the scalar route: ``-Ix(n)`` is its lowered
    partner, which is all :func:`_contract_pairs` asks of an index."""

    name: object
    up: bool = True

    def __neg__(self):
        return Ix(self.name, not self.up)


def _flip(chirality):
    return {"L": "R", "R": "L", None: None}[chirality]


def _expand_chiral_string(factors):
    """Expand a cyclic Dirac string into projector-free strings.

    ``factors`` is a sequence of

    - ``('S', leg)`` — the spin sum ``p̸ ± m`` of a
      :class:`~feynlag.pheno.diagrams.Leg`;
    - ``('Q', combo, mass)`` — a fermion-propagator numerator ``q̸ + m`` with
      ``q = Σ c_h h`` given as ``combo = ((c_h, head), …)``;
    - ``('G', atom, w_L, w_R)`` — a vertex factor ``atom·(w_L P_L + w_R P_R)``
      with ``atom`` ``None`` (a scalar vertex), a Lorentz index (a
      ``γ^atom``), an explicit slot ``('i', index)`` / ``('m', TensorHead)``
      (a ``γ`` with that index / a slash), or ``('q', combo)`` (a slashed
      linear combination of momenta).

    Every spin sum is split into its slash and mass pieces and every vertex
    into its two chiral pieces; each projector is then pushed to the right
    end (``P γ = γ P̄`` past a γ, unchanged past a mass, ``P P' = δ_{PP'} P``)
    and the surviving string ``X·P_χ`` is split with
    ``Tr[X P_{L,R}] = ½Tr[X] ∓ ½Tr[Xγ₅]``.  This is the same algebra
    :meth:`~feynlag.pheno.diagrams.SpinorChain.trace` hand-derives for one
    ``Γ``/``Γ̄`` pair, done mechanically so it covers any string.

    Returns:
        ``{atoms: (c_tr, c_g5)}`` — ``atoms`` the γ string as a tuple of
        ``('m', TensorHead)`` (a slash) / ``('i', index)`` (a ``γ^μ``)
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
        elif factor[0] == "Q":
            _, combo, mass = factor
            for atoms, chi, c in states:
                for k, head in combo:
                    new.append((atoms + (("m", head),), _flip(chi), c * k))
                if mass != 0:
                    new.append((atoms, chi, c * mass))
        elif factor[0] == "G":
            _, atom, w_left, w_right = factor
            if atom is None:
                options = [(sp.S.One, None)]
            elif isinstance(atom, tuple) and atom[0] == "q":
                options = [(k, ("m", head)) for k, head in atom[1]]
            elif isinstance(atom, tuple):
                options = [(sp.S.One, atom)]
            else:
                options = [(sp.S.One, ("i", atom))]
            for atoms, chi, c in states:
                for k, slot in options:
                    if slot is None:
                        atoms2, chi2 = atoms, chi
                    else:
                        atoms2, chi2 = atoms + (slot,), _flip(chi)
                    for proj, w in (("L", w_left), ("R", w_right)):
                        if w == 0 or (chi2 is not None and chi2 != proj):
                            continue
                        new.append((atoms2, proj, c * k * w))
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


def _scalar_trace(factors, dot, memo=None):
    """γ₅-free ``Tr[factors]`` for a **fully contracted** string, as a scalar.

    :func:`_expand_chiral_string` → :func:`_contract_pairs` →
    :func:`_slash_trace`: no SymPy tensor objects at all, so neither of the
    two SymPy 1.14 defects this route would otherwise meet (a repeated
    slashed momentum, see :func:`~feynlag.pheno.diagrams._string_tensor`;
    ``kahane_simplify``'s ``Repeated index`` error on interleaved
    contractions such as ``γ^aγ^b k̸₃ γ_a k̸₄ γ_b``) can occur.  The γ₅
    coefficients are discarded — callers must have proven them zero.

    Args:
        memo: optional dict shared across calls with the same ``dot`` (the
            slash traces of one ``|M|²`` recur many times).
    """
    memo = {} if memo is None else memo
    total = sp.S.Zero
    for atoms, (c_tr, _) in _expand_chiral_string(factors).items():
        if c_tr == 0:
            continue
        for coeff, reduced in _contract_pairs(atoms):
            total += c_tr * coeff * _slash_trace(
                tuple(value for _, value in reduced), dot, memo)
    return sp.expand(total)


def _factor_index(factor):
    """The index *name* a ``'G'`` factor's γ carries, or ``None``."""
    if factor[0] != "G":
        return None
    atom = factor[1]
    if atom is None or isinstance(atom, tuple):
        return None
    return atom


def _combo_dot(combo_a, combo_b, dot):
    return sum(ca * cb * dot(ha, hb) for ca, ha in combo_a for cb, hb in combo_b)


def _eliminate_loose(factors, tensors, dot, who):
    """Remove every loose tensor: a metric identifies its two indices, two
    vectors on one index become a dot product, a vector meeting a ``γ``
    becomes a slash.  Returns ``(scalar coefficient, factors)``.

    Raises:
        ValueError: an unknown tensor, an index name that does not appear
            exactly twice, or (with no ``γ`` to meet) a vector left alone on
            its index.
    """
    metrics = [t for t in tensors if t[0] == "g"]
    vectors = [t for t in tensors if t[0] == "v"]
    if len(metrics) + len(vectors) != len(tensors):
        raise ValueError(f"{who}: unknown tensor in {tensors!r}")

    counts = {}
    for name in ([_factor_index(f) for f in factors]
                 + [x for _, a, b in metrics for x in (a, b)]
                 + [a for _, _, a in vectors]):
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    bad = {name: n for name, n in counts.items() if n != 2}
    if bad:
        raise ValueError(
            f"{who}: every index must appear exactly twice; got "
            f"{bad}")

    coeff = sp.S.One
    # 1. a metric identifies its two indices
    while metrics:
        _, a, b = metrics.pop()
        if a == b:
            coeff *= 4
            continue
        metrics = [("g", a if x == b else x, a if y == b else y)
                   for _, x, y in metrics]
        vectors = [("v", combo, a if x == b else x) for _, combo, x in vectors]
        factors = [("G", a, f[2], f[3]) if _factor_index(f) == b else f
                   for f in factors]

    # 2. a vector meets another vector (a dot product) or a γ (a slash)
    by_index = {}
    for _, combo, a in vectors:
        by_index.setdefault(a, []).append(combo)
    for a, combos in by_index.items():
        if len(combos) == 2:
            coeff *= _combo_dot(combos[0], combos[1], dot)
        else:
            (combo,) = combos
            factors = [("G", ("q", tuple(combo)), f[2], f[3])
                       if _factor_index(f) == a else f for f in factors]
    return coeff, factors


def contract_tensors(tensors, dot):
    """``∏ tensors``, fully contracted, as a scalar — :func:`contract_trace`
    without a Dirac trace (a pure-boson ``|M|²``, Tier 5 stage 5c).

    Args:
        tensors: ``('g', a, b)`` metrics and ``('v', combo, a)`` vectors on
            plain hashable index names, each name appearing exactly twice.
        dot: callable ``(head, head) -> scalar``.

    Raises:
        ValueError: an index name that does not appear exactly twice.
    """
    coeff, _ = _eliminate_loose([], tensors, dot, "contract_tensors")
    return coeff


def contract_trace(factors, tensors, dot, memo=None):
    """``Tr[factors] × ∏ tensors``, fully contracted, as a scalar (γ₅-free
    part — see the module docstring).

    Args:
        factors: the Dirac string, as for :func:`_expand_chiral_string`,
            with a ``'G'`` factor's free Lorentz index given as a plain
            hashable **name** (not a tuple).
        tensors: loose tensors sharing those names —
            ``('g', a, b)`` (a metric) and ``('v', combo, a)`` (the vector
            ``Σ c_h h^a``, ``combo = ((c_h, head), …)``).
        dot: callable ``(head, head) -> scalar``, e.g.
            :meth:`~feynlag.pheno.kinematics.TwoToTwoKinematics.dot`.
        memo: optional slash-trace cache, see :func:`_scalar_trace`.

    Raises:
        ValueError: an index name that does not appear exactly twice.
    """
    coeff, factors = _eliminate_loose(list(factors), tensors, dot,
                                      "contract_trace")

    # 3. what is left is contracted γ pairs inside the trace
    seen = set()
    paired = []
    for f in factors:
        name = _factor_index(f)
        if name is None:
            paired.append(f)
        elif name in seen:
            paired.append(("G", ("i", Ix(name, up=False)), f[2], f[3]))
        else:
            seen.add(name)
            paired.append(("G", ("i", Ix(name)), f[2], f[3]))
    return sp.expand(coeff * _scalar_trace(paired, dot, memo))

"""A layer of commuting Z rotations as one addition into a phase-gradient register.

The layer is  prod_j exp(-i theta_j Z_j)  on bits x_j (a Z_j is a qubit, or the parity of a Pauli
string computed into one by CNOTs).  Up to a global phase it is the diagonal unitary

    |x> -> exp(2 pi i f(x) / 2^b) |x>,      f(x) = sum_j K_j x_j  (mod 2^b),

with theta_j = pi K_j / 2^b.  Write each K_j in signed digits, K_j = sum_p d_jp 2^p with d_jp in
{-1, 0, 1}, and a digit -1 on x as the bit NOT x with the constant -1: then

    f(x) + C = sum_p 2^p (number of ones among the bits in column p),

where column p holds one bit (x_j or NOT x_j) for every nonzero digit d_jp and C is the sum of
2^p over the digits -1.  The circuit has three parts.

compute    copy each digit's bit into column p (CNOTs, an X for a digit -1) and reduce every
           column p <= b - 4 to at most one bit, from the bottom up: three bits of a column
           become their sum (same column) and their majority (next column) with one AND; two
           bits become their sum and their AND.  Columns b-3, b-2, b-1 are left alone.
phase      the bits in columns b-3, b-2 and b-1 get T, S and Z (phases pi/4, pi/2, pi); the one
           surviving bit of each column p <= b-4 is added, as an integer, into the gradient
           register (Gidney's adder with the top qubit |->, as in circuit_q04's WindowAdd).
uncompute  the compute part backwards; every AND is undone by measurement, with no T gate.

The ANDs of the compute part number at most the number of digits at positions <= b - 4 (each AND
of a full adder removes a bit, and the carries into column p + 1 are at most half the bits of
column p); the gradient addition has at most b - 2.  Classes of bits that share one angle can be
added up first (Hamming-weight phasing), and the digits then multiply the bits of the sum.

The output is a list of operations on integer qubit ids:

    ('x', q) ('cx', c, t) ('cz', a, b) ('z', q) ('s', q) ('t', q) ('tdg', q)
    ('and', a, b, t)      t is fresh: t = a AND b (4 T)
    ('anddg', a, b, t)    t = a AND b is uncomputed by an X measurement and a CZ fix-up (0 T)
    ('alloc', q) ('free', q)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .digits import digits_mod


class Circ:
    """A gate list with qubit allocation; freed ids are reused, so the ids used number the peak."""

    def __init__(self, n_fixed: int = 0):
        self.ops: List[tuple] = []
        self.next_id = n_fixed
        self.pool: List[int] = []
        self.live = n_fixed
        self.peak = n_fixed

    def alloc(self) -> int:
        q = self.pool.pop() if self.pool else self._fresh()
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.ops.append(('alloc', q))
        return q

    def _fresh(self) -> int:
        q = self.next_id
        self.next_id += 1
        return q

    def free(self, q: int):
        self.ops.append(('free', q))
        self.pool.append(q)
        self.live -= 1

    def add(self, *op):
        self.ops.append(tuple(op))

    def and_(self, a: int, b: int) -> int:
        t = self.alloc()
        self.add('and', a, b, t)
        return t

    def anddg(self, a: int, b: int, t: int):
        self.add('anddg', a, b, t)
        self.free(t)


def inverse(ops: Sequence[tuple]) -> List[tuple]:
    """The inverse of a list of compute operations (x, cx, alloc, and; no phases)."""
    out = []
    for op in reversed(ops):
        k = op[0]
        if k in ('x', 'cx'):
            out.append(op)
        elif k == 'and':
            out.append(('anddg',) + op[1:])     # its 'alloc' becomes the 'free'
        elif k == 'alloc':
            out.append(('free', op[1]))
        elif k == 'free':
            raise ValueError('compute parts do not free qubits')
        else:
            raise ValueError(f'not a compute operation: {op}')
    return out


def window_add_minus_top(c: Circ, addend: Dict[int, int], window: Sequence[int]):
    """window += sum_p 2^p addend[p] (mod 2^W) on a gradient register whose top qubit is |->.

    The gate sequence of WindowAdd(W, positions, minus_top=True) in circuit_q04/ftfr_q04/
    phasegrad.py: one AND per carry position below W - 2 that has an addend bit or an incoming
    carry; the carry into the top qubit is applied as a phase.  `addend` maps positions to qubits,
    and every position is below W - 1 here.
    """
    W = len(window)
    s = list(window)
    a: List[Optional[int]] = [None] * W
    for p, q in addend.items():
        a[p] = q
    cr: List[Optional[int]] = [None] * (W + 1)
    P = W - 2
    for p in range(max(P, 0)):
        if a[p] is None and cr[p] is None:
            continue
        if cr[p] is None:
            cr[p + 1] = c.and_(a[p], s[p])
        elif a[p] is None:
            cr[p + 1] = c.and_(s[p], cr[p])
        else:
            c.add('cx', cr[p], a[p])
            c.add('cx', cr[p], s[p])
            cr[p + 1] = c.and_(a[p], s[p])
            c.add('cx', cr[p], cr[p + 1])
    top = W - 1
    if a[top] is not None:
        c.add('z', a[top])
    p = W - 2
    if a[p] is not None and cr[p] is not None:
        c.add('cx', cr[p], a[p])
        c.add('cx', cr[p], s[p])
        c.add('cz', a[p], s[p])
        c.add('z', cr[p])
        c.add('cx', cr[p], a[p])
        c.add('cx', a[p], s[p])
    elif a[p] is not None:
        c.add('cz', a[p], s[p])
        c.add('cx', a[p], s[p])
    elif cr[p] is not None:
        c.add('cz', s[p], cr[p])
        c.add('cx', cr[p], s[p])
    for p in reversed(range(max(P, 0))):
        if a[p] is None and cr[p] is None:
            continue
        if cr[p] is None:
            c.anddg(a[p], s[p], cr[p + 1])
            c.add('cx', a[p], s[p])
        elif a[p] is None:
            c.anddg(s[p], cr[p], cr[p + 1])
            c.add('cx', cr[p], s[p])
        else:
            c.add('cx', cr[p], cr[p + 1])
            c.anddg(a[p], s[p], cr[p + 1])
            c.add('cx', cr[p], a[p])
            c.add('cx', a[p], s[p])


class _Compute:
    """The compute part: records only x, cx, alloc and and, so that it can be inverted."""

    def __init__(self, c: Circ):
        self.c = c
        self.start = len(c.ops)
        self.n_and = 0

    def copy(self, src: int, negate: bool = False) -> int:
        q = self.c.alloc()
        self.c.add('cx', src, q)
        if negate:
            self.c.add('x', q)
        return q

    def full_adder(self, a: int, b: int, cc: int) -> Tuple[int, int]:
        """(a, b, c) -> sum in c, majority in a fresh qubit; a and a XOR b are left behind."""
        self.c.add('cx', a, b)
        self.c.add('cx', a, cc)
        t = self.c.and_(b, cc)
        self.c.add('cx', a, t)
        self.c.add('cx', b, cc)
        self.c.add('cx', a, cc)
        self.n_and += 1
        return cc, t

    def half_adder(self, a: int, b: int) -> Tuple[int, int]:
        """(a, b) -> sum in b, AND in a fresh qubit."""
        t = self.c.and_(a, b)
        self.c.add('cx', a, b)
        self.n_and += 1
        return b, t

    def compress(self, cols: Dict[int, List[int]], last: int) -> Dict[int, List[int]]:
        """Reduce every column p <= last to at most one bit, from the bottom up; carries out of
        column `last` land in column last + 1, which is not reduced."""
        cols = {p: list(v) for p, v in cols.items()}
        top = max(list(cols) + [0])
        p = -1
        while p < min(last, top):
            p += 1
            col = cols.get(p, [])
            while len(col) >= 2:
                if len(col) >= 3:
                    a, b, cc = col.pop(), col.pop(), col.pop()
                    s, t = self.full_adder(a, b, cc)
                else:
                    a, b = col.pop(), col.pop()
                    s, t = self.half_adder(a, b)
                col.append(s)
                cols.setdefault(p + 1, []).append(t)
                top = max(top, p + 1)
            cols[p] = col
        return cols

    def ops(self) -> List[tuple]:
        return self.c.ops[self.start:]


@dataclass
class Layer:
    """A compiled layer: the operations, and what they cost."""
    ops: List[tuple]
    b: int
    system: List[int]
    gradient: List[int]
    constant: int                 # C: the circuit applies exp(2 pi i (f(x) + C) / 2^b)
    n_and_compute: int
    n_and_gradient: int
    n_t_direct: int
    peak: int
    digits: int                   # signed digits at positions <= b - 4 (incl. Hamming-weight bits)
    survivors: int
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n_and(self) -> int:
        return self.n_and_compute + self.n_and_gradient

    @property
    def t_count(self) -> int:
        return 4 * self.n_and + self.n_t_direct


def compile_layer(classes: Sequence[Tuple[int, Sequence[int]]], b: int, n_system: int) -> Layer:
    """Compile exp(2 pi i sum_c K_c W_c(x) / 2^b), W_c(x) = sum_{j in c} x_j.

    `classes` is a list of (K, [system qubit ids]); a class of one qubit is a single rotation, a
    larger class is a set of bits that share the angle pi K / 2^b (Hamming-weight phasing).  K
    may instead be a list [K_0, K_1, ...], one integer per bit of W_c (the bit of weight 2^k
    gets the phase 2 pi K_k / 2^b in place of 2 pi 2^k K / 2^b), so that each bit can be rounded
    on its own.  The
    system qubits are 0 .. n_system - 1, the gradient register is the next b ids (least
    significant first, top qubit |->), and ancillas follow.
    """
    if b < 4:
        raise ValueError('b >= 4')
    c = Circ(n_fixed=n_system + b)
    system = list(range(n_system))
    gradient = list(range(n_system, n_system + b))
    comp = _Compute(c)
    cols: Dict[int, List[int]] = {}
    direct: List[Tuple[int, int]] = []     # (column >= b - 3, qubit): phases applied directly
    C = 0
    n_digits = 0
    for K, members in classes:
        per_bit = not isinstance(K, int)
        if not any(digits_mod(k, b) for k in (K if per_bit else [K])):
            continue
        # the bits of W_c with their weights
        if len(members) == 1:
            wbits = [(0, members[0])]
        else:
            hw = comp.compress({0: [comp.copy(q) for q in members]}, last=10 ** 6)
            wbits = [(k, v[0]) for k, v in sorted(hw.items()) if v]
        items = []                                   # (column, digit, bit)
        for k, w in wbits:
            if per_bit:                              # the bit of weight 2^k has its own angle K[k]
                items += [(p, d, w) for p, d in digits_mod(K[k], b)]
            else:
                items += [(p + k, d, w) for p, d in digits_mod(K, b)]
        for col, d, w in items:
            if col >= b:
                continue                             # a whole turn
            if d < 0:
                C += 2 ** col
            if col >= b - 3:
                direct.append((col, comp.copy(w, negate=True) if d < 0 else w))
            else:
                cols.setdefault(col, []).append(comp.copy(w, negate=d < 0))
                n_digits += 1
    cols = comp.compress(cols, last=b - 4)
    for p in range(b - 3, b):
        for q in cols.get(p, []):
            direct.append((p, q))
    survivors = {p: v[0] for p, v in cols.items() if p <= b - 4 and v}
    compute_ops = comp.ops()
    # phase
    n_t = 0
    for p, q in direct:
        if p == b - 3:
            c.add('t', q)
            n_t += 1
        elif p == b - 2:
            c.add('s', q)
        else:
            c.add('z', q)
    before = sum(1 for op in c.ops if op[0] == 'and')
    if survivors:
        window_add_minus_top(c, survivors, gradient)
    n_and_gradient = sum(1 for op in c.ops if op[0] == 'and') - before
    # uncompute
    c.ops.extend(inverse(compute_ops))
    n_and_compute = sum(1 for op in compute_ops if op[0] == 'and')
    counts: Dict[str, int] = {}
    for op in c.ops:
        counts[op[0]] = counts.get(op[0], 0) + 1
    return Layer(ops=c.ops, b=b, system=system, gradient=gradient, constant=C,
                 n_and_compute=n_and_compute, n_and_gradient=n_and_gradient, n_t_direct=n_t,
                 peak=c.peak, digits=n_digits, survivors=len(survivors), counts=counts)


def compile_rotations(Ks: Sequence[int], b: int) -> Layer:
    """A layer of single rotations exp(-i pi K_j Z_j / 2^b) on qubits 0 .. L-1, distinct angles."""
    return compile_layer([(K, [j]) for j, K in enumerate(Ks)], b, len(Ks))

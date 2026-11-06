"""Squaring with temporary ANDs: b^2 - b ANDs, as Qualtran charges for its Square (Qualtran 0.4)."""
from __future__ import annotations

from functools import cached_property
from typing import Dict

from attrs import frozen

from qualtran import Bloq, BloqBuilder, QUInt, Register, Side, Signature, SoquetT

from .phasegrad import WindowAdd
from .wires import Wires


@frozen
class Square(Bloq):
    r"""$|a\rangle|0\rangle\mapsto|a\rangle|a^2\rangle$, $a$ of $b$ bits, result of $2b$ bits.

    $a^2=\sum_k a_k4^k+\sum_{i<j}a_ia_j2^{i+j+1}$.  Bit 0 of the result is $a_0$ (a CNOT) and bit 1
    is always 0.  Row $i=0,\dots,b-2$ computes the products $a_ia_j$, $j>i$ ($b-1-i$ ANDs) and adds
    them, with $a_{i+1}$ as carry-in (the term $4^{i+1}$), into result bits $2i+2,\dots,b+i+1$: the
    partial sum after row $i$ is $a_{\le i+1}^2+2a_{\le i}a_{>i+1}<2^{b+i+2}$, so no carry leaves
    that window, which costs $b-1-i$ ANDs.  Total $2\sum_{k=1}^{b-1}k=b^2-b$ ANDs.  The inverse,
    Adjoint(Square(b)), costs the same: its subtractions recompute the products.
    """
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QUInt(self.bitsize)),
                          Register('result', QUInt(2 * self.bitsize), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, a) -> Dict[str, SoquetT]:
        w = Wires(bb)
        b = self.bitsize
        x = w.split(a, b)
        acc = w.alloc(2 * b)
        w.cnot(x[0], acc[0])
        for i in range(b - 1):
            prods = [w.and_(x[i], x[j]) for j in range(i + 1, b)]
            lo, hi = 2 * i + 2, b + i + 1
            add = WindowAdd(hi - lo + 1, tuple(range(len(prods))), carry_in=True)
            w.apply(add, x=prods, window=acc[lo:hi + 1], cin=x[i + 1])
            for j, t in reversed(list(zip(range(i + 1, b), prods))):
                w.anddg(x[i], x[j], t)
        return {'a': w.join(x), 'result': w.join(acc)}

    def __str__(self):
        return f'Square({self.bitsize})'


# ================================================================================================
# Arithmetic of the reference implementation that Qualtran 0.4 lacks or does not decompose (the
# reference took it from Qualtran 0.7), written in Qualtran 0.4 with the same gate sequences, so
# that every count equals the reference implementation's (tests/test_reference_counts.py).
# Qualtran 0.4's own Add, AddK, LessThanConstant, LessThanEqual, PrepareUniformSuperposition and
# PhaseGradientState already decompose to the leaf gates with the reference counts and are used as
# they are.
# ================================================================================================

import numpy as np  # noqa: E402

from qualtran import QBit as _QBit  # noqa: E402
from qualtran.bloqs.arithmetic import Add as _Add, LessThanEqual as _LessThanEqual  # noqa: E402
from qualtran.bloqs.basic_gates import CNOT as _CNOT, XGate as _XGate  # noqa: E402
from qualtran.bloqs.bookkeeping import Cast as _Cast  # noqa: E402
from qualtran.bloqs.mcmt import And as _And  # noqa: E402

from .gates import RegisterCSwap  # noqa: E402


def add_mod(bb: BloqBuilder, a, b, n: int):
    """b += a mod 2^n for n-qubit QUInt registers: Qualtran's Add, or for n = 1 a CNOT (as the
    reference implementation's Add(QUInt(1)) decomposes; Qualtran 0.4's Add needs n >= 2)."""
    if n > 1:
        return bb.add(_Add(QUInt(n)), a=a, b=b)
    qa = bb.add(_Cast(QUInt(1), _QBit()), reg=a)
    qb = bb.add(_Cast(QUInt(1), _QBit()), reg=b)
    qa, qb = bb.add(_CNOT(), ctrl=qa, target=qb)
    return bb.add(_Cast(_QBit(), QUInt(1)), reg=qa), bb.add(_Cast(_QBit(), QUInt(1)), reg=qb)


@frozen
class CAdd(Bloq):
    r"""`b` += `a` if `ctrl` = `cv` (the reference's CAdd): the AND of the control with each bit of
    $a$ into fresh qubits (most significant first), Gidney's adder of those into $b$, and the ANDs
    uncomputed in reverse.  $n$ + ($n-1$) ANDs for $n$-bit registers."""
    bitsize: int
    cv: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', _QBit()), Register('a', QUInt(self.bitsize)),
                          Register('b', QUInt(self.bitsize))])

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, a, b) -> Dict[str, SoquetT]:
        n = self.bitsize
        a_arr = bb.split(a) if n > 1 else [bb.add(_Cast(QUInt(1), _QBit()), reg=a)]
        ctrl_q = ctrl
        anc = []
        for i in range(len(a_arr)):
            (ctrl_q, a_arr[i]), t = bb.add(_And(self.cv, 1), ctrl=np.array([ctrl_q, a_arr[i]]))
            anc.append(t)
        ancilla = (bb.join(np.array(anc), QUInt(n)) if n > 1
                   else bb.add(_Cast(_QBit(), QUInt(1)), reg=anc[0]))
        ancilla, b = add_mod(bb, ancilla, b, n)
        anc = list(bb.split(ancilla)) if n > 1 else [bb.add(_Cast(QUInt(1), _QBit()), reg=ancilla)]
        for i in reversed(range(len(a_arr))):
            ctrl_q, a_arr[i] = bb.add(_And(self.cv, 1, uncompute=True),
                                      ctrl=np.array([ctrl_q, a_arr[i]]), target=anc[i])
        a_out = (bb.join(np.array(a_arr), QUInt(n)) if n > 1
                 else bb.add(_Cast(_QBit(), QUInt(1)), reg=a_arr[0]))
        return {'ctrl': ctrl_q, 'a': a_out, 'b': b}

    def __str__(self):
        return f'CAdd({self.bitsize})'


@frozen
class BitwiseNot(Bloq):
    """X on every qubit of an n-qubit register, most significant first (the reference's
    BitwiseNot on unsigned integers)."""
    n: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', QUInt(self.n))])

    def build_composite_bloq(self, bb: BloqBuilder, x) -> Dict[str, SoquetT]:
        qs = bb.split(x)
        for i in range(self.n):
            qs[i] = bb.add(_XGate(), q=qs[i])
        return {'x': bb.join(qs, QUInt(self.n))}

    def adjoint(self):
        return self


@frozen
class OutOfPlaceAdder(Bloq):
    r"""$|a\rangle|b\rangle|0\rangle\mapsto|a\rangle|b\rangle|a+b\bmod 2^n\rangle$ for $n$-bit $a,b$
    (the reference implementation's adder: Qualtran 0.7's OutOfPlaceAdder with
    include_most_significant_bit=False, an option Qualtran 0.4's OutOfPlaceAdder lacks): per bit
    $i<n-1$, CNOT a->b, CNOT a->c_i, $c_{i+1}=b_i\wedge c_i$, CNOT a->b, CNOT a->c_{i+1},
    CNOT b->c_i; then $c_{n-1}\mathrel{\oplus}=a_{n-1}\oplus b_{n-1}$.  $n-1$ ANDs; `is_adjoint`
    uncomputes $c$ with measurement-based uncomputes."""
    bitsize: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        side = Side.LEFT if self.is_adjoint else Side.RIGHT
        return Signature([Register('a', QUInt(self.bitsize)), Register('b', QUInt(self.bitsize)),
                          Register('c', QUInt(self.bitsize), side=side)])

    def adjoint(self):
        return OutOfPlaceAdder(self.bitsize, not self.is_adjoint)

    def build_composite_bloq(self, bb: BloqBuilder, a, b, c=None) -> Dict[str, SoquetT]:
        n = self.bitsize
        w = Wires(bb)
        A, B = w.split(a, n), w.split(b, n)
        if not self.is_adjoint:
            C = [None] * n
            C[0] = w.alloc(1)[0]
            for i in range(n - 1):
                w.cnot(A[i], B[i])
                w.cnot(A[i], C[i])
                C[i + 1] = w.and_(B[i], C[i])
                w.cnot(A[i], B[i])
                w.cnot(A[i], C[i + 1])
                w.cnot(B[i], C[i])
            w.cnot(A[n - 1], C[n - 1])
            w.cnot(B[n - 1], C[n - 1])
            return {'a': w.join(A), 'b': w.join(B), 'c': w.join(C)}
        C = w.split(c, n)
        w.cnot(B[n - 1], C[n - 1])
        w.cnot(A[n - 1], C[n - 1])
        for i in reversed(range(n - 1)):
            w.cnot(B[i], C[i])
            w.cnot(A[i], C[i + 1])
            w.cnot(A[i], B[i])
            w.anddg(B[i], C[i], C[i + 1])
            w.cnot(A[i], C[i])
            w.cnot(A[i], B[i])
        w.free([C[0]])
        return {'a': w.join(A), 'b': w.join(B)}


@frozen
class HalfLessThan(Bloq):
    r"""$|a\rangle|b\rangle\mapsto|a\rangle|b\rangle|a-b\rangle|[a<b]\rangle$ for $n$-bit unsigned
    $a,b$, keeping $c=a-b$ ($n+1$ bits) for a measurement-based uncompute: the reference's
    LinearDepthHalfLessThan, with the same sub-bloqs in the same order.  Both registers are
    extended by a zero most significant qubit; with $A=b$, $B=a$: $B\to\bar B$,
    $c=A+\bar B$ ($n$ ANDs), $c\to\bar c$ (so $c=a-b$), $B$ restored, `target` = $c_{\rm msb}$.
    With `uncompute` the inverse, with the ANDs uncomputed by measurement."""
    bitsize: int
    uncompute: bool = False

    @cached_property
    def signature(self) -> Signature:
        side = Side.LEFT if self.uncompute else Side.RIGHT
        return Signature([Register('a', QUInt(self.bitsize)), Register('b', QUInt(self.bitsize)),
                          Register('c', QUInt(self.bitsize + 1), side=side),
                          Register('target', _QBit(), side=side)])

    def adjoint(self):
        return HalfLessThan(self.bitsize, not self.uncompute)

    def build_composite_bloq(self, bb: BloqBuilder, a, b, c=None, target=None):
        n, N = self.bitsize, self.bitsize + 1
        # the reference's half greater-than, called with its a := our b and its b := our a
        A = bb.join(np.concatenate([[bb.allocate(1)], bb.split(b)]), QUInt(N))
        B = bb.join(np.concatenate([[bb.allocate(1)], bb.split(a)]), QUInt(N))
        B = bb.add(BitwiseNot(N), x=B)
        if not self.uncompute:
            A, B, cc = bb.add(OutOfPlaceAdder(N), a=A, b=B)
            cc = bb.add(BitwiseNot(N), x=cc)
            c_arr = bb.split(cc)
            t = bb.allocate(1)
            c_arr[0], t = bb.add(_CNOT(), ctrl=c_arr[0], target=t)
            cc = bb.join(c_arr, QUInt(N))
        else:
            c_arr = bb.split(c)
            c_arr[0], target = bb.add(_CNOT(), ctrl=c_arr[0], target=target)
            cc = bb.join(c_arr, QUInt(N))
            cc = bb.add(BitwiseNot(N), x=cc)
            A, B = bb.add(OutOfPlaceAdder(N, is_adjoint=True), a=A, b=B, c=cc)
        B = bb.add(BitwiseNot(N), x=B)
        a_arr, b_arr = bb.split(B), bb.split(A)
        out_a, out_b = bb.join(a_arr[1:], QUInt(n)), bb.join(b_arr[1:], QUInt(n))
        bb.free(a_arr[0])
        bb.free(b_arr[0])
        if self.uncompute:
            bb.free(target)
            return {'a': out_a, 'b': out_b}
        return {'a': out_a, 'b': out_b, 'c': cc, 'target': t}

    def __str__(self):
        return f'HalfLessThan({self.bitsize})' + ('^-1' if self.uncompute else '')


@frozen
class GreaterThan(Bloq):
    r"""`target` ^= [a > b]: Qualtran's LessThanEqual(a, b) and an X (the reference's
    GreaterThan)."""
    a_bitsize: int
    b_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QUInt(self.a_bitsize)), Register('b', QUInt(self.b_bitsize)),
                          Register('target', _QBit())])

    def build_composite_bloq(self, bb: BloqBuilder, a, b, target) -> Dict[str, SoquetT]:
        a, b, target = bb.add(_LessThanEqual(self.a_bitsize, self.b_bitsize), x=a, y=b,
                              target=target)
        target = bb.add(_XGate(), q=target)
        return {'a': a, 'b': b, 'target': target}


@frozen
class Comparator(Bloq):
    r"""Sort two $n$-bit keys: `out` = [a > b] into a fresh qubit, then swap $a,b$ if `out`
    (the reference's Comparator; the swap is one AND per qubit pair)."""
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QUInt(self.bitsize)), Register('b', QUInt(self.bitsize)),
                          Register('out', _QBit(), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, a, b) -> Dict[str, SoquetT]:
        out = bb.allocate(dtype=_QBit())
        a, b, out = bb.add(GreaterThan(self.bitsize, self.bitsize), a=a, b=b, target=out)
        out, a, b = bb.add(RegisterCSwap(self.bitsize), ctrl=out, x=a, y=b)
        return {'a': a, 'b': b, 'out': out}


@frozen
class ParallelComparators(Bloq):
    """Comparators on index pairs (i, i + offset) in blocks of 2*offset (as the reference)."""
    k: int
    offset: int
    bitsize: int

    @cached_property
    def num_comparisons(self) -> int:
        full = (self.k // (2 * self.offset)) * self.offset
        rest = self.k % (2 * self.offset)
        return full + max(rest - self.offset, 0)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('xs', QUInt(self.bitsize), shape=(self.k,)),
                          Register('junk', _QBit(), shape=(self.num_comparisons,), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, xs) -> Dict[str, SoquetT]:
        xs = np.array(xs)
        junk = []
        for start in range(0, self.k, 2 * self.offset):
            for step in range(self.offset):
                i, j = start + step, start + step + self.offset
                if j >= self.k:
                    break
                xs[i], xs[j], anc = bb.add(Comparator(self.bitsize), a=xs[i], b=xs[j])
                junk.append(anc)
        return {'xs': xs, 'junk': np.array(junk)}


@frozen
class BitonicMerge(Bloq):
    """Merge two sorted lists of `half_length` keys (as the reference)."""
    half_length: int
    bitsize: int

    @cached_property
    def num_comparisons(self) -> int:
        return self.half_length * ((self.half_length - 1).bit_length() + 1)

    @cached_property
    def signature(self) -> Signature:
        k = self.half_length
        return Signature([Register('xs', QUInt(self.bitsize), shape=(k,), side=Side.LEFT),
                          Register('ys', QUInt(self.bitsize), shape=(k,), side=Side.LEFT),
                          Register('result', QUInt(self.bitsize), shape=(2 * k,), side=Side.RIGHT),
                          Register('junk', _QBit(), shape=(self.num_comparisons,), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, xs, ys) -> Dict[str, SoquetT]:
        k = self.half_length
        if k & (k - 1):
            raise ValueError('half_length must be a power of 2')
        xs, ys = np.array(xs), np.array(ys)
        first = []
        for i in range(k):
            xs[i], ys[k - 1 - i], anc = bb.add(Comparator(self.bitsize), a=xs[i], b=ys[k - 1 - i])
            first.append(anc)
        result = np.concatenate([xs, ys])
        logk = (k - 1).bit_length()
        junks = [np.array(first)]
        for log_offset in reversed(range(logk)):
            result, ancs = bb.add(ParallelComparators(2 * k, 2 ** log_offset, self.bitsize),
                                  xs=result)
            junks.append(np.array(ancs).reshape(-1))
        return {'result': result, 'junk': np.concatenate(junks)}


@frozen
class BitonicSort(Bloq):
    """Sort `k` keys ascending (k a power of 2), keeping one comparison qubit per comparator
    (the reference's BitonicSort)."""
    k: int
    bitsize: int

    @cached_property
    def num_comparisons(self) -> int:
        logk = (self.k - 1).bit_length()
        return (self.k // 2) * ((logk * (logk + 1)) // 2)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('xs', QUInt(self.bitsize), shape=(self.k,)),
                          Register('junk', _QBit(), shape=(self.num_comparisons,), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, xs) -> Dict[str, SoquetT]:
        xs = np.array(xs)
        if self.k == 1:
            return {'xs': xs, 'junk': np.array([])}
        if self.k == 2:
            xs[0], xs[1], anc = bb.add(Comparator(self.bitsize), a=xs[0], b=xs[1])
            return {'xs': xs, 'junk': np.array([anc])}
        h = self.k // 2
        xl, jl = bb.add(BitonicSort(h, self.bitsize), xs=xs[:h])
        xr, jr = bb.add(BitonicSort(h, self.bitsize), xs=xs[h:])
        xs, jm = bb.add(BitonicMerge(h, self.bitsize), xs=xl, ys=xr)
        junk = np.concatenate([np.array(jl).reshape(-1), np.array(jr).reshape(-1),
                               np.array(jm).reshape(-1)])
        return {'xs': xs, 'junk': junk}

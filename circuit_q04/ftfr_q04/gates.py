"""Small exact gadgets: controlled Hadamard, phases on one basis state, reflections, controlled
swaps and routing.  Qualtran 0.4."""
from __future__ import annotations

from functools import cached_property
from typing import Dict, Tuple

import numpy as np
from attrs import frozen

from qualtran import (Bloq, BloqBuilder, BoundedQUInt, QAny, QBit, QUInt, Register, Signature,
                      SoquetT)

from .wires import Wires


def nbits(x: int) -> int:
    """Qubits needed for the integers 0..x-1."""
    return 0 if x <= 1 else (x - 1).bit_length()


@frozen
class CHadamardExact(Bloq):
    r"""Controlled Hadamard in Clifford+T: on the target $S^\dagger, H, T^\dagger$, a CNOT from the
    control, then $T, H, S$.  Exact (no global phase); two T gates."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), target=QBit())

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, target) -> Dict[str, SoquetT]:
        w = Wires(bb)
        c, t = w.put(ctrl), w.put(target)
        w.s(t, dagger=True)
        w.h(t)
        w.t(t, dagger=True)
        w.cnot(c, t)
        w.t(t)
        w.h(t)
        w.s(t)
        return {'ctrl': w.take(c), 'target': w.take(t)}

    def adjoint(self):
        return self

    def __str__(self):
        return 'CH'


@frozen
class FanHadamard(Bloq):
    r"""$H^{\otimes w}$ on `target` if `ctrl` = 1: $w$ controlled Hadamards from one control."""
    width: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), target=QUInt(self.width))

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, target) -> Dict[str, SoquetT]:
        w = Wires(bb)
        c = w.put(ctrl)
        ts = w.split(target, self.width)
        for q in reversed(ts):            # most significant qubit first, as drawn
            w.apply(CHadamardExact(), ctrl=c, target=q)
        return {'ctrl': w.take(c), 'target': w.join(ts)}

    def adjoint(self):
        return self

    def __str__(self):
        return f'C-H^{self.width}'


@frozen
class BasisPhase(Bloq):
    r"""Multiply one computational basis state of `n` qubits by $e^{i\pi t}$ (`exponent` $=t$).

    The state is $q_k=v_k$ with `values` $=(v_0,\dots,v_{n-1})$ in register order (qubit 0 of `x`
    is its most significant qubit, as Qualtran splits it).  An AND ladder over the $n$ conditions
    ($n-1$ ANDs, zero-conditions by negated controls) and one $Z^t$ on the last AND output; for
    $n=1$ a single $Z^t$, conjugated by $X$ when $v_0=0$.  The generalised reflection
    $R_0(\varphi)=I-(1-e^{i\varphi})|0\rangle\langle0|$ is BasisPhase with all values 0 and
    $t=\varphi/\pi$."""
    n: int
    exponent: float
    values: Tuple[int, ...] = None

    @cached_property
    def vals(self) -> Tuple[int, ...]:
        return tuple(self.values) if self.values is not None else (0,) * self.n

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QAny(self.n) if self.n > 1 else QBit())

    def build_composite_bloq(self, bb: BloqBuilder, x) -> Dict[str, SoquetT]:
        w = Wires(bb)
        qs = w.split(x, self.n)[::-1]          # register order: qs[0] is Qualtran's first qubit
        v = self.vals
        if self.n == 1:
            if v[0] == 0:
                w.x(qs[0])
            w.zpow(qs[0], self.exponent)
            if v[0] == 0:
                w.x(qs[0])
            return {'x': w.take(qs[0])}
        ladder = [w.and_(qs[0], qs[1], v[0], v[1])]
        for k in range(2, self.n):
            ladder.append(w.and_(ladder[-1], qs[k], 1, v[k]))
        w.zpow(ladder[-1], self.exponent)
        for k in range(self.n - 1, 1, -1):
            w.anddg(ladder[k - 2], qs[k], ladder.pop(), 1, v[k])
        w.anddg(qs[0], qs[1], ladder.pop(), v[0], v[1])
        return {'x': w.join(qs[::-1], dtype=QAny(self.n))}

    def adjoint(self):
        return BasisPhase(self.n, -self.exponent, self.values)

    def __str__(self):
        return f'Phase(|{"".join(map(str, self.vals))}>, {self.exponent:.4g}pi)'


@frozen
class ZeroReflection(Bloq):
    r"""$I-2|0^n\rangle\langle0^n|$ with $n-2$ ANDs: an AND ladder over the first $n-1$ qubits
    (negated controls), a CZ from its last output onto the last qubit (conjugated by X), and the
    ladder uncomputed.  $n=1$: $XZX$; $n=2$: $X^{\otimes2}\,CZ\,X^{\otimes2}$."""
    n: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=QAny(self.n) if self.n > 1 else QBit())

    def build_composite_bloq(self, bb: BloqBuilder, x) -> Dict[str, SoquetT]:
        w = Wires(bb)
        qs = w.split(x, self.n)[::-1]
        n = self.n
        if n == 1:
            w.x(qs[0])
            w.z(qs[0])
            w.x(qs[0])
            return {'x': w.take(qs[0])}
        if n == 2:
            w.x(qs[0])
            w.x(qs[1])
            w.cz(qs[0], qs[1])
            w.x(qs[0])
            w.x(qs[1])
            return {'x': w.join(qs[::-1], dtype=QAny(2))}
        ladder = [w.and_(qs[0], qs[1], 0, 0)]
        for k in range(2, n - 1):
            ladder.append(w.and_(ladder[-1], qs[k], 1, 0))
        w.x(qs[n - 1])
        w.cz(ladder[-1], qs[n - 1])
        w.x(qs[n - 1])
        for k in range(n - 2, 1, -1):
            w.anddg(ladder[k - 2], qs[k], ladder.pop(), 1, 0)
        w.anddg(qs[0], qs[1], ladder.pop(), 0, 0)
        return {'x': w.join(qs[::-1], dtype=QAny(n))}

    def adjoint(self):
        return self

    def __str__(self):
        return f'R0({self.n})'


@frozen
class RegisterCSwap(Bloq):
    r"""Swap registers `x` and `y` of `width` qubits if `ctrl` = 1: for each qubit pair, from the
    most significant, CNOT y->x, a Toffoli c,x->y as a temporary AND, CNOT y->x.  One AND (4 T)
    per qubit pair; exact."""
    width: int

    @cached_property
    def signature(self) -> Signature:
        dt = QAny(self.width) if self.width > 1 else QBit()
        return Signature.build_from_dtypes(ctrl=QBit(), x=dt, y=dt)

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, x, y) -> Dict[str, SoquetT]:
        w = Wires(bb)
        c = w.put(ctrl)
        xs, ys = w.split(x, self.width), w.split(y, self.width)
        for i in reversed(range(self.width)):
            w.cswap(c, xs[i], ys[i])
        dt = QAny(self.width) if self.width > 1 else QBit()
        return {'ctrl': w.take(c), 'x': w.join(xs, dtype=dt), 'y': w.join(ys, dtype=dt)}

    def adjoint(self):
        return self

    def __str__(self):
        return f'CSwap({self.width})'


def swap_sequence(sel_bits: int, r: int):
    """The SwapWithZero order: (selection bit from the LSB, slot a, slot b), so that after the
    sequence slot 0 holds the register that started in slot `sel`."""
    out = []
    for bit in range(sel_bits):
        step = 1 << bit
        for a in range(0, r, 2 * step):
            if a + step < r:
                out.append((bit, a, a + step))
    return out


@frozen
class RouteToZero(Bloq):
    r"""Move target register `sel` into slot 0: the SwapWithZero network of exact controlled swaps
    ($r-1$ register swaps of `width` qubits, one AND per qubit pair).  The other slots are
    permuted; `uncompute=True` runs the network backwards."""
    sel_bits: int
    width: int
    r: int
    uncompute: bool = False

    @cached_property
    def signature(self) -> Signature:
        dt = QAny(self.width) if self.width > 1 else QBit()
        return Signature([Register('sel', BoundedQUInt(self.sel_bits, self.r)),
                          Register('targets', dt, shape=(self.r,))])

    def build_composite_bloq(self, bb: BloqBuilder, sel, targets) -> Dict[str, SoquetT]:
        w = Wires(bb)
        s = w.split(sel, self.sel_bits)
        targets = np.array(targets)
        seq = swap_sequence(self.sel_bits, self.r)
        if self.uncompute:
            seq = seq[::-1]
        for bit, a, b in seq:
            c2, x, y = bb.add(RegisterCSwap(self.width), ctrl=w.take(s[bit]), x=targets[a],
                              y=targets[b])
            s[bit] = w.put(c2)
            targets[a], targets[b] = x, y
        return {'sel': w.join(s, dtype=BoundedQUInt(self.sel_bits, self.r)), 'targets': targets}

    def adjoint(self):
        return RouteToZero(self.sel_bits, self.width, self.r, not self.uncompute)

    def __str__(self):
        return 'Route' + ('^-1' if self.uncompute else '')

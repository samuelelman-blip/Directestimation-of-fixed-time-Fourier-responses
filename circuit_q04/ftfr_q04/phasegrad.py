"""Additions with temporary ANDs, and additions into the phase-gradient register (Qualtran 0.4)."""
from __future__ import annotations

from functools import cached_property
from typing import Dict, Tuple

from attrs import frozen

from qualtran import Bloq, BloqBuilder, QAny, QBit, QFxp, QUInt, Register, Signature, SoquetT

from .wires import Wires


@frozen
class WindowAdd(Bloq):
    r"""`window` += addend (+ `cin`) mod $2^W$, Gidney's adder (arXiv:1709.06648).

    The addend qubits sit at the window positions `positions` (ascending; the addend register is
    least significant first in that order); absent positions are zero.  Each carry is one
    temporary AND: $c_{p+1}=c_p\oplus(a_p\oplus c_p)(s_p\oplus c_p)$, uncomputed by measurement
    after the sum bit is written.  With `minus_top` the window's top qubit is assumed to be $|-\rangle$
    (the top qubit of a phase-gradient register): writing $a\oplus c$ into it is the phase
    $(-1)^{a\oplus c}$, so the last carry is applied as a phase and never computed.  ANDs: one per
    carry position that has an addend bit or an incoming carry below the top, i.e. $W-1$, or
    $W-2$ with `minus_top`, for a contiguous addend starting at position 0.
    """
    W: int
    positions: Tuple[int, ...]
    carry_in: bool = False
    minus_top: bool = False

    @cached_property
    def L(self) -> int:
        return len(self.positions)

    @cached_property
    def signature(self) -> Signature:
        regs = []
        if self.L:
            regs.append(Register('x', QAny(self.L) if self.L > 1 else QBit()))
        regs.append(Register('window', QAny(self.W) if self.W > 1 else QBit()))
        if self.carry_in:
            regs.append(Register('cin', QBit()))
        return Signature(regs)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        wr = Wires(bb)
        W = self.W
        xs = wr.split(soqs['x'], self.L) if self.L else []
        s = wr.split(soqs['window'], W)
        a = [None] * W
        for k, p in enumerate(self.positions):
            a[p] = xs[k]
        c = [None] * (W + 1)
        c[0] = wr.put(soqs['cin']) if self.carry_in else None
        P = W - 2 if self.minus_top else W - 1
        # forward: carries c[1..P]
        for p in range(max(P, 0)):
            if a[p] is None and c[p] is None:
                continue
            if c[p] is None:
                c[p + 1] = wr.and_(a[p], s[p])
            elif a[p] is None:
                c[p + 1] = wr.and_(s[p], c[p])
            else:
                wr.cnot(c[p], a[p])
                wr.cnot(c[p], s[p])
                c[p + 1] = wr.and_(a[p], s[p])
                wr.cnot(c[p], c[p + 1])
        top = W - 1
        if not self.minus_top:
            if c[top] is not None:
                wr.cnot(c[top], s[top])
            if a[top] is not None:
                wr.cnot(a[top], s[top])
        else:
            if a[top] is not None:
                wr.z(a[top])
            if W == 1:
                if c[0] is not None:
                    wr.z(c[0])
            else:
                p = W - 2                       # the carry into the top, as a phase
                if a[p] is not None and c[p] is not None:
                    wr.cnot(c[p], a[p])
                    wr.cnot(c[p], s[p])
                    wr.cz(a[p], s[p])
                    wr.z(c[p])
                    wr.cnot(c[p], a[p])
                    wr.cnot(a[p], s[p])
                elif a[p] is not None:
                    wr.cz(a[p], s[p])
                    wr.cnot(a[p], s[p])
                elif c[p] is not None:
                    wr.cz(s[p], c[p])
                    wr.cnot(c[p], s[p])
        # backward: uncompute carries, write sums
        for p in reversed(range(max(P, 0))):
            if a[p] is None and c[p] is None:
                continue
            if c[p] is None:
                wr.anddg(a[p], s[p], c[p + 1])
                wr.cnot(a[p], s[p])
            elif a[p] is None:
                wr.anddg(s[p], c[p], c[p + 1])
                wr.cnot(c[p], s[p])
            else:
                wr.cnot(c[p], c[p + 1])
                wr.anddg(a[p], s[p], c[p + 1])
                wr.cnot(c[p], a[p])
                wr.cnot(a[p], s[p])
        out = {'window': wr.join(s, dtype=QAny(W)) if W > 1 else wr.take(s[0])}
        if self.L:
            out['x'] = wr.join(xs, dtype=QAny(self.L)) if self.L > 1 else wr.take(xs[0])
        if self.carry_in:
            out['cin'] = wr.take(c[0])
        return out

    def __str__(self):
        return f'+= ({self.L} bits into {self.W})'


@frozen
class PhaseGradAdd(Bloq):
    r"""$|x\rangle|\phi\rangle\mapsto e^{2\pi i\,x\,2^{k}/2^{w}}|x\rangle|\phi\rangle$ on the top $w$ qubits
    of the catalytic phase-gradient register, $k=w-x_b-s\ge0$.

    The same action as Qualtran's AddIntoPhaseGrad($x_b$, $w$, right_shift=$s$,
    controlled_by=1 if `controlled`).  The register is the $w$-qubit phase-gradient state, whose
    top qubit is $|-\rangle$, so the addition is a WindowAdd with `minus_top`: $w-2$ ANDs.  Controlled:
    $x_j\wedge$ctrl into fresh qubits first (one AND per bit of $x$ below the top position, except
    bits listed in `known_zero`), the top bit as CZ(ctrl, $x_j$).  The inverse, Adjoint(...),
    subtracts.
    """
    x_bits: int
    phase_bits: int
    right_shift: int = 0
    controlled: bool = False
    known_zero: Tuple[int, ...] = ()

    @cached_property
    def offset(self) -> int:
        k = self.phase_bits - self.x_bits - self.right_shift
        if k < 0:
            raise ValueError('x would be truncated: need phase_bits >= x_bits + right_shift')
        return k

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('ctrl', QBit())] if self.controlled else []
        regs += [Register('x', QUInt(self.x_bits)),
                 Register('phase_grad', QFxp(self.phase_bits, self.phase_bits))]
        return Signature(regs)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        wr = Wires(bb)
        w, k = self.phase_bits, self.offset
        xs = wr.split(soqs['x'], self.x_bits)
        ph = wr.split(soqs['phase_grad'], w)
        top = w - 1
        bits = [j for j in range(self.x_bits) if j not in self.known_zero]
        if self.controlled:
            c = wr.put(soqs['ctrl'])
            low = [j for j in bits if k + j < top]
            prods = [wr.and_(c, xs[j]) for j in low]
            for j in bits:
                if k + j == top:
                    wr.cz(c, xs[j])
            add = WindowAdd(w, tuple(k + j for j in low), minus_top=True)
            if low:
                wr.apply(add, x=prods, window=ph)
            else:
                wr.apply(add, window=ph)
            for j, t in reversed(list(zip(low, prods))):
                wr.anddg(c, xs[j], t)
        else:
            add = WindowAdd(w, tuple(k + j for j in bits), minus_top=True)
            if bits:
                wr.apply(add, x=[xs[j] for j in bits], window=ph)
        out = {'x': wr.join(xs, dtype=QUInt(self.x_bits)),
               'phase_grad': wr.join(ph, dtype=QFxp(w, w))}
        if self.controlled:
            out['ctrl'] = wr.take(c)
        return out

    def __str__(self):
        return ('C-' if self.controlled else '') + f'PhaseGradAdd({self.x_bits},{self.phase_bits})'

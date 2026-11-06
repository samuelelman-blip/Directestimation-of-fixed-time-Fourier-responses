"""Row access for the profiled families: the row norm w_j as an amplitude, the row phase e^{i psi_j}
(Qualtran 0.4; the same bloqs as ../circuit/ftfr/access.py).

Closed form (F): w_j = 1 - j/(2n) and psi_j = (pi/4)(2j/n - 1)^2, exact with b_v = b + 1 value bits
and a (2b+1)-qubit phase.  Tables (T): W_j = round(2^b_v w_j) and Psi_j = round(2^b_psi psi_j / 2 pi)
mod 2^b_psi from one select-swap lookup.  The profiled dense family (PD) needs both, the profiled
sparse family (PS) the norm only.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Dict, Optional

import numpy as np
from attrs import frozen

from qualtran import (Adjoint, Bloq, BloqBuilder, QBit, QFxp, QUInt, Register, Side, Signature,
                      SoquetT)
from qualtran.bloqs.arithmetic import AddK

from .arith import HalfLessThan, Square
from .lookup import LookupSpec, LookupUncompute, SelectSwapLookup, default_lookup_spec
from .phasegrad import PhaseGradAdd
from .wires import Wires


def profile_w(b: int) -> np.ndarray:
    n = 2 ** b
    return 1 - np.arange(n) / (2 * n)


def profile_psi(b: int) -> np.ndarray:
    n = 2 ** b
    return (math.pi / 4) * (2 * np.arange(n) / n - 1) ** 2


def table_columns(b: int, bv: int, bpsi: Optional[int]):
    """Integer columns of the tabulated access: W_j (bv + 1 bits) and, for PD, Psi_j (bpsi bits)."""
    W = [int(round(x * 2 ** bv)) for x in profile_w(b)]
    cols = [W]
    if bpsi:
        cols.append([int(round(x * 2 ** bpsi / (2 * math.pi))) % 2 ** bpsi for x in profile_psi(b)])
    return cols


@frozen
class ValueOracleF(Bloq):
    r"""$|j\rangle|0\rangle\mapsto|j\rangle|2n-j\rangle$ on a new $(b+2)$-qubit register: CNOT $j$ into
    its low $b$ qubits, $X$ on all, $+1$ (AddK), $X$ on the top qubit.  $W_j/2^{b+1}=w_j$."""
    b: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('j', QUInt(self.b)),
                          Register('val', QUInt(self.b + 2), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, j) -> Dict[str, SoquetT]:
        b = self.b
        w = Wires(bb)
        jq = w.split(j, b)
        v = w.alloc(b + 2)
        for k in range(b):
            w.cnot(jq[k], v[k])
        for q in v:
            w.x(q)
        val = bb.add(AddK(b + 2, 1), x=w.join(v))
        v = w.split(val, b + 2)
        w.x(v[b + 1])
        return {'j': w.join(jq), 'val': w.join(v)}

    def __str__(self):
        return 'W_j=2n-j'


@frozen
class InequalityTest(Bloq):
    r"""Block-encode $W/2^{b_v}$: $H^{\otimes b_v}$ on $m$, $\ell t=[m<W]$ (HalfLessThan on
    $b_v+1$ bits, $m$ padded on top), $g\mathrel{\oplus}=\mathrm{ctrl}\wedge\lnot\ell t$ (one AND),
    the comparison uncomputed by measurement, $H^{\otimes b_v}$.  With ctrl $=1$ the amplitude of
    $m=0$, $g=0$ is exactly $W/2^{b_v}$; with ctrl $=0$ the bloq is the identity on $m=0$, $g=0$."""
    bv: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), val=QUInt(self.bv + 1), m=QUInt(self.bv),
                                           g=QBit())

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, val, m, g) -> Dict[str, SoquetT]:
        bv = self.bv
        w = Wires(bb)
        mq = w.split(m, bv)
        for q in reversed(mq):
            w.h(q)
        pad = w.alloc(1)
        mext = w.join(mq + pad, dtype=QUInt(bv + 1))
        mext, val, cq, tq = bb.add(HalfLessThan(bv + 1), a=mext, b=val)
        c, t, gq = w.put(ctrl), w.put(tq), w.put(g)
        u = w.and_(c, t, 1, 0)
        w.cnot(u, gq)
        w.anddg(c, t, u, 1, 0)
        mext, val = bb.add(HalfLessThan(bv + 1, uncompute=True), a=mext, b=val, c=cq,
                           target=w.take(t))
        mq = w.split(mext, bv + 1)
        w.free([mq.pop()])
        for q in reversed(mq):
            w.h(q)
        return {'ctrl': w.take(c), 'val': val, 'm': w.join(mq), 'g': w.take(gq)}

    def __str__(self):
        return '[m<W]'


@frozen
class RowPhaseF(Bloq):
    r"""$e^{i\psi_j}$, $\psi_j/2\pi=(j-n/2)^2/2^{2b+1}$, on ctrl $=1$, added into the top $2b+1$ qubits
    of the phase gradient.  $x=j-n/2$ by flipping the top bit of $j$; its sign copied out; $|x|$ by
    CNOTs from the sign and a controlled increment; Square($b$); a controlled PhaseGradAdd of
    $|x|^2$ (bit 1 of a square is always 0, so it needs no AND); all undone."""
    b: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), j=QUInt(self.b),
                                           phase_grad=QFxp(2 * self.b + 1, 2 * self.b + 1))

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, j, phase_grad) -> Dict[str, SoquetT]:
        b = self.b
        w = Wires(bb)
        jq = w.split(j, b)
        top = jq[b - 1]
        w.x(top)                                         # two's complement x = j - n/2
        (sg,) = w.alloc(1)
        w.cnot(top, sg)
        for q in reversed(jq):
            w.cnot(sg, q)
        inc = AddK(b, 1, cvs=(1,))
        w.apply(inc, ctrls=[sg], x=jq)
        sq = Square(b)
        res = w.apply(sq, a=jq)
        r = res['result']
        add = PhaseGradAdd(2 * b, 2 * b + 1, right_shift=1, controlled=True, known_zero=(1,))
        c = w.put(ctrl)
        ph = w.split(phase_grad, 2 * b + 1)
        w.apply(add, ctrl=c, x=r, phase_grad=ph)
        w.apply(Adjoint(sq), a=jq, result=r)
        w.apply(Adjoint(inc), ctrls=[sg], x=jq)
        for q in jq:
            w.cnot(sg, q)
        w.cnot(top, sg)
        w.free([sg])
        w.x(top)
        return {'ctrl': w.take(c), 'j': w.join(jq),
                'phase_grad': w.join(ph, dtype=QFxp(2 * b + 1, 2 * b + 1))}

    def __str__(self):
        return 'e^{i psi_j}'


@frozen
class RowAccess(Bloq):
    r"""Lemma 2's row data on the parent label $j$, under the contraction's control.

    With ctrl $=1$, $\langle j,m{=}0,g{=}0|\cdot|j,m{=}0,g{=}0\rangle=w_je^{i\psi_j}$ (PD) or $w_j$
    (PS); with ctrl $=0$ the identity on $m=0$, $g=0$.  F: ValueOracleF, InequalityTest, ValueOracleF
    undone, then RowPhaseF (PD).  T: SelectSwapLookup of $(W_j,\Psi_j)$, InequalityTest on $W_j$,
    a controlled PhaseGradAdd of $\Psi_j$ (PD), LookupUncompute."""
    family: str            # 'PD' or 'PS'
    level: str             # 'F' or 'T'
    b: int
    bv: int
    bpsi: int = 0
    spec: Optional[LookupSpec] = None

    @cached_property
    def phase_bits(self) -> int:
        if self.family != 'PD':
            return 0
        return 2 * self.b + 1 if self.level == 'F' else self.bpsi

    @cached_property
    def signature(self) -> Signature:
        regs = dict(ctrl=QBit(), j=QUInt(self.b), m=QUInt(self.bv), g=QBit())
        if self.phase_bits:
            regs['phase_grad'] = QFxp(self.phase_bits, self.phase_bits)
        return Signature.build_from_dtypes(**regs)

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, j, m, g, phase_grad=None) -> Dict[str, SoquetT]:
        out = {}
        if self.level == 'F':
            assert self.bv == self.b + 1
            j, val = bb.add(ValueOracleF(self.b), j=j)
            ctrl, val, m, g = bb.add(InequalityTest(self.bv), ctrl=ctrl, val=val, m=m, g=g)
            j = bb.add(Adjoint(ValueOracleF(self.b)), j=j, val=val)
            if self.family == 'PD':
                ctrl, j, phase_grad = bb.add(RowPhaseF(self.b), ctrl=ctrl, j=j, phase_grad=phase_grad)
                out['phase_grad'] = phase_grad
            out.update(ctrl=ctrl, j=j, m=m, g=g)
            return out
        lk = SelectSwapLookup(self.spec)
        regs = bb.add_d(lk, selection=j)
        w = Wires(bb)
        val_q = w.split(regs['target0'], self.bv + 1)
        ctrl, val, m, g = bb.add(InequalityTest(self.bv), ctrl=ctrl,
                                 val=w.join(val_q, dtype=QUInt(self.bv + 1)), m=m, g=g)
        regs['target0'] = w.join(w.split(val, self.bv + 1),
                                 dtype=lk.signature.get_right('target0').dtype)
        if self.family == 'PD':
            psi = w.join(w.split(regs['target1'], self.bpsi), dtype=QUInt(self.bpsi))
            add = PhaseGradAdd(self.bpsi, self.bpsi, controlled=True)
            ctrl, psi, phase_grad = bb.add(add, ctrl=ctrl, x=psi, phase_grad=phase_grad)
            regs['target1'] = w.join(w.split(psi, self.bpsi),
                                     dtype=lk.signature.get_right('target1').dtype)
            out['phase_grad'] = phase_grad
        sel = regs.pop('selection')
        j = bb.add(LookupUncompute(self.spec, key='rowdata'), selection=sel, **regs)
        out.update(ctrl=ctrl, j=j, m=m, g=g)
        return out

    def __str__(self):
        return f'Access({self.family}-{self.level})'


def make_access(family: str, level: str, b: int, bv: int, bpsi: int) -> RowAccess:
    if level == 'F':
        return RowAccess(family, 'F', b, b + 1, 0 if family == 'PS' else 2 * b + 1)
    cols = table_columns(b, bv, bpsi if family == 'PD' else None)
    widths = (bv + 1,) + ((bpsi,) if family == 'PD' else ())
    spec = default_lookup_spec(b, widths, cols)
    return RowAccess(family, 'T', b, bv, bpsi if family == 'PD' else 0, spec)

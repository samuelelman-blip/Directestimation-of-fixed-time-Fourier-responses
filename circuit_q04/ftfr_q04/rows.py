"""The row gadget R of Lemma 2 for the dense skeleton, its controlled inverse, and the sparse
skeleton's shift (Qualtran 0.4; the same bloqs as the reference implementation, which is not included)."""
from __future__ import annotations

import math
from functools import cached_property
from typing import Dict

from attrs import frozen

from qualtran import Adjoint, Bloq, BloqBuilder, QBit, QUInt, Signature, SoquetT
from qualtran.bloqs.basic_gates import SGate, ZPowGate

from .arith import CAdd, add_mod
from .gates import BasisPhase, FanHadamard
from .wires import Wires


def row_angles(n: int):
    """(phi, alpha) of the exact amplification step: cos phi = 1/(sqrt n + 1), alpha = pi/2 + phi/2."""
    phi = math.acos(1.0 / (math.sqrt(n) + 1.0))
    return phi, math.pi / 2 + phi / 2


@frozen
class AChi(Bloq):
    r"""$A_\chi=(H\otimes I)(|0\rangle\langle0|\otimes H^{\otimes b}+|1\rangle\langle1|\otimes I)(H\otimes I)$
    on $(a,\ell)$; controlled, only the middle factor needs the control (the outer $H$ cancel):
    $u=\mathrm{ctrl}\wedge[a=0]$ by one AND, $b$ controlled Hadamards from $u$, AND uncomputed."""
    b: int
    controlled: bool = False

    @cached_property
    def signature(self) -> Signature:
        regs = dict(a=QBit(), l=QUInt(self.b))
        if self.controlled:
            regs = dict(ctrl=QBit(), **regs)
        return Signature.build_from_dtypes(**regs)

    def build_composite_bloq(self, bb: BloqBuilder, a, l, ctrl=None) -> Dict[str, SoquetT]:
        w = Wires(bb)
        aq = w.put(a)
        lq = w.split(l, self.b)
        w.h(aq)
        if self.controlled:
            c = w.put(ctrl)
            u = w.and_(c, aq, 1, 0)
            w.apply(FanHadamard(self.b), ctrl=u, target=lq)
            w.anddg(c, aq, u, 1, 0)
        else:
            w.x(aq)
            w.apply(FanHadamard(self.b), ctrl=aq, target=lq)
            w.x(aq)
        w.h(aq)
        out = {'a': w.take(aq), 'l': w.join(lq)}
        if self.controlled:
            out['ctrl'] = w.take(c)
        return out

    def adjoint(self):
        return self

    def __str__(self):
        return 'A_chi' + ('^c' if self.controlled else '')


@frozen
class RowPrepDense(Bloq):
    r"""$R|j\rangle|0\rangle|0\rangle=e^{i\alpha}|j\rangle|0\rangle(|f\rangle+|j-1\rangle)/\nu_n$: $A_\chi$,
    $Z_\varphi$, $A_\chi$, $R_0(\varphi)$, $A_\chi$, $X^{\otimes b}$, $\ell\mathrel{+}=j$.  The phase
    $e^{i\alpha}$ is removed by the contraction's phase gate."""
    b: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(j=QUInt(self.b), a=QBit(), l=QUInt(self.b))

    def build_composite_bloq(self, bb: BloqBuilder, j, a, l) -> Dict[str, SoquetT]:
        b = self.b
        phi, _ = row_angles(2 ** b)
        w = Wires(bb)
        aq, lq = w.put(a), w.split(l, b)
        w.apply(AChi(b), a=aq, l=lq)
        w.apply(BasisPhase(1, phi / math.pi, (0,)), x=[aq])
        w.apply(AChi(b), a=aq, l=lq)
        w.apply(BasisPhase(b + 1, phi / math.pi), x=(lq + [aq]))
        w.apply(AChi(b), a=aq, l=lq)
        for q in lq:
            w.x(q)
        j, l2 = add_mod(bb, j, w.join(lq), b)
        return {'j': j, 'a': w.take(aq), 'l': l2}

    def __str__(self):
        return 'R'


@frozen
class ControlledRowInverse(Bloq):
    r"""$R^\dagger$ on (parent $j$, flag $a$, child $\ell$) if ctrl $=1$, identity if ctrl $=0$.

    Dense skeleton: $\ell\mathrel{-}=j$ (CAdd adjoint), $X^{\otimes b}$ on $\ell$ by CNOTs from ctrl,
    $A_\chi^c$, $R_0(-\varphi)$ on (ctrl $=1$, $a=0$, $\ell=0$), $A_\chi^c$, $Z_{-\varphi}$ on
    (ctrl $=1$, $a=0$), $A_\chi^c$.  Sparse skeleton ($R$ is the shift alone, no flag): the first two
    steps only."""
    b: int
    skeleton: str = 'dense'

    @cached_property
    def signature(self) -> Signature:
        if self.skeleton == 'dense':
            return Signature.build_from_dtypes(ctrl=QBit(), j=QUInt(self.b), a=QBit(), l=QUInt(self.b))
        return Signature.build_from_dtypes(ctrl=QBit(), j=QUInt(self.b), l=QUInt(self.b))

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, j, l, a=None) -> Dict[str, SoquetT]:
        b = self.b
        ctrl, j, l = bb.add(Adjoint(CAdd(b, 1)), ctrl=ctrl, a=j, b=l)
        w = Wires(bb)
        c, lq = w.put(ctrl), w.split(l, b)
        for q in reversed(lq):
            w.cnot(c, q)
        if self.skeleton == 'dense':
            phi, _ = row_angles(2 ** b)
            aq = w.put(a)
            w.apply(AChi(b, True), ctrl=c, a=aq, l=lq)
            # R_0(-phi): the register order of BasisPhase is (ctrl, a, l from its top bit)
            w.apply(BasisPhase(b + 2, -phi / math.pi, (1, 0) + (0,) * b), x=(lq + [aq, c]))
            w.apply(AChi(b, True), ctrl=c, a=aq, l=lq)
            w.apply(BasisPhase(2, -phi / math.pi, (1, 0)), x=[aq, c])
            w.apply(AChi(b, True), ctrl=c, a=aq, l=lq)
            return {'ctrl': w.take(c), 'j': j, 'a': w.take(aq), 'l': w.join(lq)}
        return {'ctrl': w.take(c), 'j': j, 'l': w.join(lq)}

    def __str__(self):
        return 'C-R^dag' + ('' if self.skeleton == 'dense' else '(shift)')


@frozen
class ContractionPhase(Bloq):
    r"""$\mathrm{diag}(1,\,ie^{i\alpha})=Z^{1+\varphi/2\pi}$ on the contraction's control: the factor
    $i$ of $B=iQ$ and the removal of $e^{i\alpha}$.  Sparse skeleton: $S=\mathrm{diag}(1,i)$."""
    b: int
    skeleton: str = 'dense'

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit())

    @cached_property
    def exponent(self) -> float:
        if self.skeleton == 'dense':
            phi, _ = row_angles(2 ** self.b)
            return 1.0 + phi / (2 * math.pi)
        return 0.5

    def build_composite_bloq(self, bb: BloqBuilder, ctrl) -> Dict[str, SoquetT]:
        if self.skeleton == 'dense':
            return {'ctrl': bb.add(ZPowGate(exponent=self.exponent), q=ctrl)}
        return {'ctrl': bb.add(SGate(), q=ctrl)}

    def __str__(self):
        return f'Z^{self.exponent:.6g}'

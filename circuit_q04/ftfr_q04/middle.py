"""The middle block X: folded controls, U_0, root and vertex phases, the contractions (Qualtran 0.4;
the same bloqs as the reference implementation, which is not included)."""
from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
from attrs import frozen

from qualtran import Adjoint, Bloq, BloqBuilder, QBit, QFxp, QUInt, Register, Signature, SoquetT

from .access import RowAccess
from .gates import FanHadamard, RouteToZero, nbits
from .history import parent_dtype
from .phasegrad import PhaseGradAdd
from .rows import ContractionPhase, ControlledRowInverse
from .wires import Wires


@frozen
class VertexPhase(Bloq):
    r"""$|j\rangle|t\rangle|\phi\rangle\mapsto|j\rangle|t\rangle|\phi+j(2t+1)\bmod2^{b_\phi}\rangle$,
    $b_\phi=b+b_t+1$: with $\phi$ the phase-gradient state this is $D_j(t)$ at the cell midpoint for
    $\mu T=1$.  For the bit $j_i$ of weight $2^i$: $x=j_i\wedge t'$ with $t'=2t+1$ ($b_t$ ANDs; the
    last bit of $x$ is $j_i$ itself), PhaseGradAdd($b_t+1$, $b_\phi-i$, right_shift $=b-i$) on the top
    $b_\phi-i$ qubits, ANDs uncomputed.  No control: unused labels are zero."""
    b: int
    b_t: int

    @property
    def b_phi(self) -> int:
        return self.b + self.b_t + 1

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(j=QUInt(self.b), t=QUInt(self.b_t),
                                           phase_grad=QFxp(self.b_phi, self.b_phi))

    def build_composite_bloq(self, bb: BloqBuilder, j, t, phase_grad) -> Dict[str, SoquetT]:
        b, bt, bphi = self.b, self.b_t, self.b_phi
        w = Wires(bb)
        jq, tq, ph = w.split(j, b), w.split(t, bt), w.split(phase_grad, bphi)
        for i in range(b):
            prods = [w.and_(jq[i], tq[p]) for p in range(bt)]       # x bit p+1 = j_i t_p
            x = [jq[i]] + prods                                     # LSB first: t' = 2t + 1
            add = PhaseGradAdd(bt + 1, bphi - i, right_shift=b - i)
            w.apply(add, x=x, phase_grad=ph[i:])
            for p in reversed(range(bt)):
                w.anddg(jq[i], tq[p], prods[p])
        return {'j': w.join(jq), 't': w.join(tq), 'phase_grad': w.join(ph, dtype=QFxp(bphi, bphi))}

    def __str__(self):
        return 'D(t)'


@frozen
class Contraction(Bloq):
    r"""Contraction $\mathcal C_r$: route the parent label $j_{p_r}$ to slot 0, controlled $R^\dagger$ on
    (slot 0, $a_r$, $j_r$), the phase gate on $c_r$, the row access on (slot 0, $m_r$, $g_r$, $\phi$)
    for the profiled families, route back.  With $c_r=1$ the block from $j_r=\ell$ to the good
    subspace is $i(G_1)_{j\ell}/\gamma$; with $c_r=0$ it is the identity."""
    r: int
    b: int
    skeleton: str
    access: Optional[RowAccess] = None

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('ctrl', QBit())]
        if self.r >= 2:
            regs.append(Register('sel', parent_dtype(self.r)))
        regs.append(Register('labels', QUInt(self.b), shape=(self.r,)))
        regs.append(Register('child', QUInt(self.b)))
        if self.skeleton == 'dense':
            regs.append(Register('a', QBit()))
        if self.access is not None:
            regs += [Register('m', QUInt(self.access.bv)), Register('g', QBit())]
            if self.access.phase_bits:
                regs.append(Register('phase_grad', QFxp(self.access.phase_bits, self.access.phase_bits)))
        return Signature(regs)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        w = Wires(bb)
        r, b = self.r, self.b
        c = w.put(soqs['ctrl'])
        labels = [w.split(s, b) for s in soqs['labels']]
        child = w.split(soqs['child'], b)
        route = RouteToZero(nbits(r), b, r) if r >= 2 else None
        if route is not None:
            sel = w.split(soqs['sel'], nbits(r))
            w.apply(route, sel=sel, targets=labels)
        regs = dict(ctrl=c, j=labels[0], l=child)
        if self.skeleton == 'dense':
            a = w.put(soqs['a'])
            regs['a'] = a
        w.apply(ControlledRowInverse(b, self.skeleton), **regs)
        w.apply(ContractionPhase(b, self.skeleton), ctrl=c)
        out = {}
        if self.access is not None:
            m, g = w.split(soqs['m'], self.access.bv), w.put(soqs['g'])
            areg = dict(ctrl=c, j=labels[0], m=m, g=g)
            if self.access.phase_bits:
                ph = w.split(soqs['phase_grad'], self.access.phase_bits)
                areg['phase_grad'] = ph
            w.apply(self.access, **areg)
            out['m'] = w.join(m)
            out['g'] = w.take(g)
            if self.access.phase_bits:
                pb = self.access.phase_bits
                out['phase_grad'] = w.join(ph, dtype=QFxp(pb, pb))
        if route is not None:
            w.apply(route.adjoint(), sel=sel, targets=labels)
            out['sel'] = w.join(sel, dtype=parent_dtype(r)) if nbits(r) > 1 else w.take(sel[0])
        out['ctrl'] = w.take(c)
        out['labels'] = np.array([w.join(l) for l in labels])
        out['child'] = w.join(child)
        if self.skeleton == 'dense':
            out['a'] = w.take(a)
        return out

    def __str__(self):
        return f'C_{self.r}'


@frozen
class MiddleBlock(Bloq):
    r"""$X$ with the Hadamard-test control `anc`, for $K=1$.

    1. $c_r=\mathsf{anc}\wedge f_r$ (one AND each); 2. $U_0$: $H^{\otimes b}$ on $j_0$ controlled by
    anc and on $j_r$ by $c_r$; 3. $D(T)$: $j_0$ added into the top $b$ qubits of $\phi$;
    4. $D(t_r)$: VertexPhase; 5. contractions $\mathcal C_M,\dots,\mathcal C_1$; 6. $U_c^\dagger$:
    $j_0$ subtracted, $H^{\otimes b}$ on $j_0$ controlled by anc; 7. the $c_r$ uncomputed."""
    M: int
    b: int
    b_t: int
    w_phi: int
    skeleton: str
    access: Optional[RowAccess] = None

    def work_registers(self) -> List[Register]:
        M = self.M
        regs = [Register('j', QUInt(self.b), shape=(M + 1,))]
        if M and self.skeleton == 'dense':
            regs.append(Register('a', QBit(), shape=(M,)))
        if M and self.access is not None:
            regs += [Register('m', QUInt(self.access.bv), shape=(M,)), Register('g', QBit(), shape=(M,))]
        return regs

    @cached_property
    def signature(self) -> Signature:
        M = self.M
        regs = [Register('anc', QBit())]
        if M:
            regs += [Register('f', QBit(), shape=(M,)), Register('t', QUInt(self.b_t), shape=(M,))]
            regs += [Register(f'p{r}', parent_dtype(r)) for r in range(2, M + 1)]
        regs += self.work_registers()
        regs.append(Register('phase_grad', QFxp(self.w_phi, self.w_phi)))
        return Signature(regs)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        M, b, bt, W = self.M, self.b, self.b_t, self.w_phi
        w = Wires(bb)
        anc = w.put(soqs['anc'])
        j = [w.split(s, b) for s in soqs['j']]
        ph = w.split(soqs['phase_grad'], W)

        def top(k):
            return ph[W - k:]

        f = [w.put(s) for s in soqs['f']] if M else []
        t = [w.split(s, bt) for s in soqs['t']] if M else []
        par = {r: w.split(soqs[f'p{r}'], nbits(r)) for r in range(2, M + 1)}
        a = [w.put(s) for s in soqs['a']] if 'a' in soqs else None
        m = [w.split(s, self.access.bv) for s in soqs['m']] if 'm' in soqs else None
        g = [w.put(s) for s in soqs['g']] if 'g' in soqs else None
        # 1. folded controls
        c = [w.and_(anc, f[r]) for r in range(M)]
        # 2. U_0
        w.apply(FanHadamard(b), ctrl=anc, target=j[0])
        for r in range(M):
            w.apply(FanHadamard(b), ctrl=c[r], target=j[r + 1])
        # 3. D(T)
        root = PhaseGradAdd(b, b)
        w.apply(root, x=j[0], phase_grad=top(b))
        # 4. D(t_r)
        for r in range(M):
            w.apply(VertexPhase(b, bt), j=j[r + 1], t=t[r], phase_grad=top(b + bt + 1))
        # 5. contractions in reverse creation order
        for r in range(M, 0, -1):
            con = Contraction(r, b, self.skeleton, self.access)
            regs = dict(ctrl=c[r - 1], labels=j[:r], child=j[r])
            if r >= 2:
                regs['sel'] = par[r]
            if self.skeleton == 'dense':
                regs['a'] = a[r - 1]
            if self.access is not None:
                regs['m'] = m[r - 1]
                regs['g'] = g[r - 1]
                if self.access.phase_bits:
                    regs['phase_grad'] = top(self.access.phase_bits)
            w.apply(con, **regs)
        # 6. U_c^dagger
        w.apply(Adjoint(root), x=j[0], phase_grad=top(b))
        w.apply(FanHadamard(b), ctrl=anc, target=j[0])
        # 7. uncompute the folded controls
        for r in reversed(range(M)):
            w.anddg(anc, f[r], c[r])
        out = {'anc': w.take(anc), 'j': np.array([w.join(x) for x in j]),
               'phase_grad': w.join(ph, dtype=QFxp(W, W))}
        if M:
            out['f'] = np.array([w.take(x) for x in f])
            out['t'] = np.array([w.join(x) for x in t])
            for r in range(2, M + 1):
                out[f'p{r}'] = w.join(par[r], dtype=parent_dtype(r)) if nbits(r) > 1 else w.take(par[r][0])
        if a is not None:
            out['a'] = np.array([w.take(x) for x in a])
        if m is not None:
            out['m'] = np.array([w.join(x) for x in m])
            out['g'] = np.array([w.take(x) for x in g])
        return out

    def __str__(self):
        return 'C-X'

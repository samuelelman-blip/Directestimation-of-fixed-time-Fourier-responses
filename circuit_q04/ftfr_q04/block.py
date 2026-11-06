"""The block W = P^dagger (C-X) P, the Hadamard test A, the reflection S_0, the Grover iterate Q, and
the run (Qualtran 0.4; the same bloqs as the reference implementation, which is not included).

A run at one cell is the phase-gradient state prepared once, then the Grover iterate repeated `nq`
times.  `run_plan(cell)` gives the two bloqs and that repeat count, so a compiler walks the iterate
once and repeats it; the full run is never written out.
"""
from __future__ import annotations

import json
import math
import os
from functools import cached_property
from typing import Dict, List, Optional

import numpy as np
from attrs import frozen

from qualtran import Adjoint, Bloq, BloqBuilder, QFxp, QBit, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, SGate
from qualtran.bloqs.rotations import PhaseGradientState

from .access import RowAccess, make_access
from .gates import ZeroReflection
from .history import HistoryPrep
from .middle import MiddleBlock
from .wires import Wires

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


@frozen
class Cell:
    """One costed configuration: a family, its row-data access, and the best reading's parameters.

    family: 'dense', 'sparse', 'PD' (profiled dense) or 'PS' (profiled sparse);
    access: 'flat' (no row data), 'F' (closed form) or 'T' (tables)."""
    family: str
    access: str
    b: int
    M: int
    b_t: int
    w_phi: int
    sel_precision: float
    bv: int = 0
    bpsi: int = 0
    mu_T: float = 1.0
    r_couple: float = 0.125
    ts_per_rot: int = 0
    nq: int = 0
    eps: float = 0.0

    @property
    def skeleton(self) -> str:
        return 'dense' if self.family in ('dense', 'PD') else 'sparse'

    @property
    def eta(self) -> float:
        """eta = rho gamma tau_mu(T) = r (1 - e^{-mu T}) for rho = 1, gamma = r mu."""
        return self.r_couple * (1 - math.exp(-self.mu_T))

    @cached_property
    def row_access(self) -> Optional[RowAccess]:
        if self.family not in ('PD', 'PS'):
            return None
        return make_access(self.family, self.access, self.b, self.bv, self.bpsi)

    @property
    def label(self) -> str:
        acc = {'flat': '', 'F': '-F', 'T': '-T'}[self.access]
        return f'{self.family}{acc} eps={self.eps:g}'


def load_cells(path: Optional[str] = None) -> List[Cell]:
    """The nine costed cells of the resource estimate, with the best reading's parameters, from
    data/cells.json."""
    path = path or os.path.join(DATA, 'cells.json')
    out = []
    for c in json.load(open(path)):
        out.append(Cell(c['family'], c['access'], c['b'], c['M'], c['b_t'], c['w_phi'],
                        c['sel_precision'], c['bv'], c['bphi'], ts_per_rot=c['ts_per_rot'],
                        nq=c['nq'], eps=c['eps']))
    return out


def cell_model(cell: Cell) -> Dict:
    """The cost model's numbers for a cell (data/cells.json): T per part, the run total, qubits."""
    for c in json.load(open(os.path.join(DATA, 'cells.json'))):
        if (c['family'], c['access'], c['eps']) == (cell.family, cell.access, cell.eps):
            return c['model']
    raise KeyError(cell.label)


def small_cell(family: str, access: str, b: int, M: int, b_t: int, sel_precision: float = 1e-3,
               bv: Optional[int] = None, bpsi: Optional[int] = None) -> Cell:
    """A configuration at small n for simulation."""
    if access == 'F':
        bv = b + 1
        bpsi = 2 * b + 1 if family == 'PD' else 0
    elif access == 'T':
        bv = bv or b + 2
        bpsi = (bpsi or 2 * b + 2) if family == 'PD' else 0
    else:
        bv, bpsi = 0, 0
    w_phi = max(b + b_t + 1, bpsi if family == 'PD' else 0)
    return Cell(family, access, b, M, b_t, w_phi, sel_precision, bv, bpsi)


def block_parts(cell: Cell):
    prep = HistoryPrep(cell.M, cell.b_t, cell.eta, cell.sel_precision, cell.mu_T) if cell.M else None
    mid = MiddleBlock(cell.M, cell.b, cell.b_t, cell.w_phi, cell.skeleton, cell.row_access)
    return prep, mid


def persistent_registers(cell: Cell) -> List[Register]:
    """anc, the history registers, the work registers: everything the readout reflects, in order."""
    prep, mid = block_parts(cell)
    regs = [Register('anc', QBit())]
    if prep is not None:
        regs += prep.history_registers()
    regs += mid.work_registers()
    return regs


@frozen
class Block(Bloq):
    r"""$W$ with its Hadamard-test control: $P$, $C$-$X$, $P^\dagger$.  On anc $=1$,
    $\langle0|W|0\rangle=h_M/\Xi_M$ up to the time discretisation; on anc $=0$ the identity.  The
    comparison record of the sort lives only inside the block."""
    cell: Cell

    @cached_property
    def signature(self) -> Signature:
        W = self.cell.w_phi
        return Signature(persistent_registers(self.cell) + [Register('phase_grad', QFxp(W, W))])

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        prep, mid = block_parts(self.cell)
        hist_names = [r.name for r in prep.history_registers()] if prep is not None else []
        record = None
        if prep is not None:
            out = bb.add_d(prep, **{k: soqs[k] for k in hist_names})
            record = out.pop('record', None)
            soqs.update(out)
        mid_names = [r.name for r in mid.signature]
        soqs.update(bb.add_d(mid, **{k: soqs[k] for k in mid_names}))
        if prep is not None:
            kw = {k: soqs[k] for k in hist_names}
            if record is not None:
                kw['record'] = record
            soqs.update(bb.add_d(Adjoint(prep), **kw))
        return soqs

    def __str__(self):
        return 'W'


@frozen
class HadamardTest(Bloq):
    r"""$A$: $H$ on anc, ($S^\dagger$ for the imaginary part), the block, $H$ on anc.  The amplitude of
    anc $=0$, work $=0$ is $(1+w)/2$, or $(1-iw)/2$ with $S^\dagger$, $w=\langle0|W|0\rangle$."""
    cell: Cell
    imag: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Block(self.cell).signature

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        soqs['anc'] = bb.add(Hadamard(), q=soqs['anc'])
        if self.imag:
            soqs['anc'] = bb.add(SGate(is_adjoint=True), q=soqs['anc'])
        soqs.update(bb.add_d(Block(self.cell), **soqs))
        soqs['anc'] = bb.add(Hadamard(), q=soqs['anc'])
        return soqs

    def __str__(self):
        return 'A' + ('_Im' if self.imag else '_Re')


@frozen
class ReflectZero(Bloq):
    r"""$I-2|0\rangle\langle0|$ on anc and every history and work register (not the phase gradient)."""
    cell: Cell

    @cached_property
    def signature(self) -> Signature:
        return Signature(persistent_registers(self.cell))

    @cached_property
    def width(self) -> int:
        return sum(r.total_bits() for r in self.signature)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        w = Wires(bb)
        hs, layout = [], []
        for reg in self.signature:
            if reg.shape:
                per = []
                for idx in reg.all_idxs():
                    q = w.split(soqs[reg.name][idx], reg.bitsize)
                    per.append(q)
                    hs += q
                layout.append((reg, per))
            else:
                q = w.split(soqs[reg.name], reg.bitsize)
                hs += q
                layout.append((reg, q))
        w.apply(ZeroReflection(len(hs)), x=hs[::-1])
        out = {}
        for reg, q in layout:
            if reg.shape:
                arr = np.empty(reg.shape, dtype=object)
                for idx, qq in zip(reg.all_idxs(), q):
                    arr[idx] = w.join(qq, dtype=reg.dtype) if reg.bitsize > 1 else w.take(qq[0])
                out[reg.name] = arr
            else:
                out[reg.name] = w.join(q, dtype=reg.dtype) if reg.bitsize > 1 else w.take(q[0])
        return out

    def adjoint(self):
        return self

    def __str__(self):
        return f'R0({self.width})'


@frozen
class GroverIterate(Bloq):
    r"""$Q=A\,S_0\,A^\dagger S_0$ (time order: $S_0$, $A^\dagger$, $S_0$, $A$) with the linear projector:
    the good subspace is anc $=0$, work $=0$, the same reflection as the initial state's."""
    cell: Cell
    imag: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Block(self.cell).signature

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        A = HadamardTest(self.cell, self.imag)
        refl = ReflectZero(self.cell)
        names = [r.name for r in refl.signature]
        soqs.update(bb.add_d(refl, **{k: soqs[k] for k in names}))
        soqs.update(bb.add_d(Adjoint(A), **soqs))
        soqs.update(bb.add_d(refl, **{k: soqs[k] for k in names}))
        soqs.update(bb.add_d(A, **soqs))
        return soqs

    def __str__(self):
        return 'Q'


def run_plan(cell: Cell, counts: Optional[Dict] = None) -> Dict:
    """The run at a cell as a compiler should take it: `prep` once, then `iterate` repeated
    `repeats` times.  The header fields come from the cost model (data/cells.json); with `counts`
    (as written by tools/count_cells.py) it also carries the streamed T counts and peak, which
    tests/test_block.py checks against the model."""
    m = cell_model(cell)
    plan = dict(label=cell.label, n=2 ** cell.b, M=cell.M, b_t=cell.b_t, w_phi=cell.w_phi,
                ts_per_rot=cell.ts_per_rot, prep=PhaseGradientState(cell.w_phi),
                iterate=GroverIterate(cell), repeats=cell.nq,
                T_prep=m['T_pg'], T_iterate=m['T_Q'], T_total=m['T_total'], qubits=m['qubits'])
    if counts is not None:
        plan.update(streamed=counts)
    return plan

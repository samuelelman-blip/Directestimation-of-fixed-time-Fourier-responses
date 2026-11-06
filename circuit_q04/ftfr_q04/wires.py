"""A thin layer over Qualtran 0.4's BloqBuilder for writing decompositions gate by gate.

Every qubit is an integer handle; each gate replaces the soquet behind its handles.  Registers are
split into handles least significant bit first: for a QUInt(w) register `r`, `w.split(r)[k]` is the
qubit of weight 2^k (Qualtran's own split is most significant bit first).

Only the compiler's leaf gates are emitted: X, Y, Z, H, S, S^dagger, T, T^dagger, CNOT, CZ (as
CZPowGate(1)), Z**t (as ZPowGate, or as Z, S, T and their adjoints when t is a multiple of 1/4),
R_y, And and And^dagger.  A controlled swap is written as CNOT, And, CNOT, And^dagger, CNOT
(one AND, 4 T), never as Toffoli or TwoBitCSwap.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from qualtran import BloqBuilder, QBit, QUInt, Side, Soquet
from qualtran.bloqs.basic_gates import (CNOT, CZPowGate, Hadamard, Ry, Rz, SGate, TGate, XGate,
                                        YGate, ZGate, ZPowGate)
from qualtran.bloqs.mcmt import And

from .stream import is_clifford_power


class Wires:
    def __init__(self, bb: BloqBuilder):
        self.bb = bb
        self.soq: Dict[int, Soquet] = {}
        self._next = 0

    # -- handles -----------------------------------------------------------------------------------
    def _new(self, soq) -> int:
        h = self._next
        self._next += 1
        self.soq[h] = soq
        return h

    def split(self, soq, nbits: int = None) -> List[int]:
        """LSB-first handles for the qubits of a register soquet."""
        if nbits is None:
            nbits = soq.reg.bitsize
        if nbits == 1:
            return [self._new(soq)]
        qs = self.bb.split(soq)
        return [self._new(q) for q in list(qs)[::-1]]

    def join(self, handles: Sequence[int], dtype=None):
        """Join LSB-first handles into one register soquet (default QUInt)."""
        handles = list(handles)
        if dtype is None:
            dtype = QUInt(len(handles))
        if len(handles) == 1 and isinstance(dtype, QBit):
            return self.soq.pop(handles[0])
        soqs = [self.soq.pop(h) for h in handles[::-1]]
        return self.bb.join(np.array(soqs), dtype=dtype)

    def take(self, h: int):
        return self.soq.pop(h)

    def put(self, soq) -> int:
        return self._new(soq)

    def alloc(self, n: int = 1) -> List[int]:
        if n == 1:
            return [self._new(self.bb.allocate(dtype=QBit()))]
        return self.split(self.bb.allocate(dtype=QUInt(n)), n)

    def free(self, handles: Sequence[int]):
        handles = list(handles)
        if len(handles) == 1:
            self.bb.free(self.soq.pop(handles[0]))
        else:
            self.bb.free(self.join(handles))

    # -- gates -------------------------------------------------------------------------------------
    def _one(self, gate, q):
        self.soq[q] = self.bb.add(gate, q=self.soq[q])

    def x(self, q):
        self._one(XGate(), q)

    def y(self, q):
        self._one(YGate(), q)

    def z(self, q):
        self._one(ZGate(), q)

    def h(self, q):
        self._one(Hadamard(), q)

    def s(self, q, dagger=False):
        self._one(SGate(is_adjoint=dagger), q)

    def t(self, q, dagger=False):
        self._one(TGate(is_adjoint=dagger), q)

    def zpow(self, q, exponent: float):
        """Z**t: as Clifford+T gates when t is within 1e-12 of a multiple of 1/4, else ZPowGate."""
        k = is_clifford_power(exponent)
        if k is None:
            self._one(ZPowGate(exponent=float(exponent)), q)
            return
        for g in {0: [], 1: ['t'], 2: ['s'], 3: ['s', 't'], 4: ['z'], 5: ['z', 't'], 6: ['sdg'],
                  7: ['tdg']}[k]:
            if g == 't':
                self.t(q)
            elif g == 's':
                self.s(q)
            elif g == 'z':
                self.z(q)
            elif g == 'sdg':
                self.s(q, dagger=True)
            elif g == 'tdg':
                self.t(q, dagger=True)

    def ry(self, q, theta: float):
        self._one(Ry(angle=float(theta)), q)

    def rz(self, q, theta: float):
        self._one(Rz(angle=float(theta)), q)

    def cnot(self, c, t):
        self.soq[c], self.soq[t] = self.bb.add(CNOT(), ctrl=self.soq[c], target=self.soq[t])

    def cz(self, a, b):
        out = self.bb.add(CZPowGate(exponent=1.0), q=np.array([self.soq[a], self.soq[b]]))
        self.soq[a], self.soq[b] = out

    def and_(self, c1, c2, cv1: int = 1, cv2: int = 1) -> int:
        """A fresh qubit holding [c1 = cv1] AND [c2 = cv2] (4 T)."""
        (a, b), t = self.bb.add(And(cv1, cv2), ctrl=np.array([self.soq[c1], self.soq[c2]]))
        self.soq[c1], self.soq[c2] = a, b
        return self._new(t)

    def anddg(self, c1, c2, t, cv1: int = 1, cv2: int = 1):
        """Measurement-based uncompute of `t` = [c1 = cv1] AND [c2 = cv2] (0 T)."""
        a, b = self.bb.add(And(cv1, cv2, uncompute=True),
                           ctrl=np.array([self.soq[c1], self.soq[c2]]), target=self.soq.pop(t))
        self.soq[c1], self.soq[c2] = a, b

    def toffoli(self, c1, c2, t):
        """t ^= c1 c2 as a temporary AND (4 T): AND into a fresh qubit, CNOT, uncompute."""
        a = self.and_(c1, c2)
        self.cnot(a, t)
        self.anddg(c1, c2, a)

    def cswap(self, c, x, y):
        """Controlled swap of one qubit pair: CNOT y->x, Toffoli(c, x -> y), CNOT y->x."""
        self.cnot(y, x)
        self.toffoli(c, x, y)
        self.cnot(y, x)

    def apply(self, bloq, **regs):
        """Apply a bloq.  Each register is given as a handle (one-qubit registers), a list of
        LSB-first handles, or, for shaped registers, a list of those.  Through-registers keep
        their handles; new right registers get new handles.  Returns the right registers in the
        same form, as a dict."""
        in_soqs = {}
        for reg in bloq.signature.lefts():
            hs = regs[reg.name]
            if reg.shape:
                arr = np.empty(reg.shape, dtype=object)
                for idx in reg.all_idxs():
                    arr[idx] = self._pop_reg(_index(hs, idx), reg)
                in_soqs[reg.name] = arr
            else:
                in_soqs[reg.name] = self._pop_reg(hs, reg)
        out = self.bb.add_d(bloq, **in_soqs)
        res = {}
        for reg in bloq.signature.rights():
            v = out[reg.name]
            reuse = reg.side == Side.THRU and reg.name in regs
            if reg.shape:
                hs_all = []
                for idx in reg.all_idxs():
                    hs = _index(regs[reg.name], idx) if reuse else None
                    hs_all.append(self._push_reg(v[idx], reg, hs))
                res[reg.name] = np.array(hs_all, dtype=object).reshape(reg.shape) \
                    if reg.bitsize == 1 else _reshape_lists(hs_all, reg.shape)
            else:
                res[reg.name] = self._push_reg(v, reg, regs[reg.name] if reuse else None)
        return res

    def _pop_reg(self, hs, reg):
        if reg.bitsize == 1:
            h = hs if isinstance(hs, (int, np.integer)) else list(hs)[0]
            return self.soq.pop(int(h))
        return self.join(list(hs), dtype=reg.dtype)

    def _push_reg(self, soq, reg, hs=None):
        if reg.bitsize == 1:
            if hs is None:
                return self._new(soq)
            h = hs if isinstance(hs, (int, np.integer)) else list(hs)[0]
            self.soq[int(h)] = soq
            return hs
        qs = list(self.bb.split(soq))[::-1]
        if hs is None:
            return [self._new(q) for q in qs]
        for h, q in zip(hs, qs):
            self.soq[h] = q
        return hs


def _index(hs, idx):
    out = hs
    for i in idx:
        out = out[i]
    return out


def _reshape_lists(flat, shape):
    if len(shape) == 1:
        return list(flat)
    step = int(np.prod(shape[1:]))
    return [_reshape_lists(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]

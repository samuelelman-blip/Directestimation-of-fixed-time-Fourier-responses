"""Walk a Qualtran 0.4 bloq down to the compiler's leaf gates, and simulate or count the result.

The compiler consumes Qualtran 0.4 bloqs and stops decomposing at these leaves:

    XGate YGate ZGate Hadamard SGate TGate (and the adjoints of S and T)
    CNOT Toffoli ZPowGate CZPowGate Rz Ry And And-dagger

Every other bloq must decompose, through `decompose_bloq()`, into leaves and the bookkeeping bloqs
(Split, Join, Partition, Cast, Allocate, Free).  The one exception is a bloq that measures or uses
measurement outcomes: as Qualtran 0.4's own And-dagger does, it yields cirq operations (a
measurement, gates conditioned on outcomes) from `decompose_from_registers`.  Such bloqs set the
class attribute `measured = True`; the walker takes their gates from `measured_ops`, which gives the
same gates as the cirq decomposition (tests/test_lookup.py).

Bloqs whose decomposition is a large, data-dependent tree (the table lookups) also have
`stream_ops`, a generator of exactly the gates their decomposition gives; with `ctx.fast` (the
default) the walker uses it instead of building the tree.

`walk` produces the leaves lazily, one level of one decomposition at a time, as tuples
(name, qubits, arg) on integer qubit ids:

    ALLOC q / FREE q                 a qubit enters or leaves in |0>
    X Y Z H S SDG T TDG q
    CNOT c t
    CZPOW a b  t                     CZ**t (t = 1 is CZ)
    TOFFOLI c1 c2 t
    ZPOW q  t                        Z**t = diag(1, e^{i pi t})
    RZ q  theta / RY q theta         exp(-i theta Z/2), exp(-i theta Y/2)
    AND c1 c2 t  (cv1, cv2)          t = [c1 = cv1] and [c2 = cv2] into a fresh t (4 T)
    ANDDG c1 c2 t  (cv1, cv2)        its measurement-based uncompute (0 T)
    MX q key / MZ q key              X or Z measurement into classical bit `key`, then a reset
                                     (measured bloqs only)
    Z q / CZ a b  with a Condition   applied only if the condition holds (measured bloqs only)

`SparseSim` simulates such a stream exactly; `count` and `t_count` count it.  `paper_counts`
converts the counts to the conventions of the paper's Table V (an AND with a control value 0
carries two X gates; a rotation within 1e-12 of a Clifford+T gate is that gate).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from qualtran import Bloq, CompositeBloq, LeftDangle, RightDangle, Soquet
from qualtran.bloqs.basic_gates import (CNOT, CZPowGate, Hadamard, Ry, Rz, SGate, TGate, Toffoli,
                                        XGate, YGate, ZGate, ZPowGate)
from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Partition, Split
from qualtran.bloqs.mcmt import And

Op = Tuple[str, Tuple[int, ...], Any]

GATE_LEAVES = (XGate, YGate, ZGate, Hadamard, SGate, TGate, CNOT, Toffoli, ZPowGate, CZPowGate,
               Rz, Ry, And)
BOOKKEEPING = (Split, Join, Cast, Partition, Allocate, Free)
LEAF_TYPES = GATE_LEAVES + BOOKKEEPING
ANGLE_ATOL = 1e-12


# ------------------------------------------------------------------------------------------------
# classical conditions on measurement outcomes
# ------------------------------------------------------------------------------------------------

class Condition:
    def evaluate(self, cbits: Dict[int, int]) -> int:
        raise NotImplementedError

    def keys(self) -> Tuple[int, ...]:
        raise NotImplementedError


class Parity(Condition):
    """XOR of the listed classical bits, optionally complemented."""

    def __init__(self, keys: Iterable[int], invert: bool = False):
        self._keys = tuple(int(k) for k in keys)
        self.invert = bool(invert)

    def evaluate(self, cbits):
        v = 0
        for k in self._keys:
            v ^= cbits[k]
        return v ^ int(self.invert)

    def keys(self):
        return self._keys

    def __repr__(self):
        body = ' ^ '.join(f'c[{k}]' for k in self._keys) or '0'
        return f'!({body})' if self.invert else body


# ------------------------------------------------------------------------------------------------
# qubit ids and the walking context
# ------------------------------------------------------------------------------------------------

class QubitAllocator:
    """Integer qubit ids; released ids are reused, lowest first.  `peak` is the circuit width."""

    def __init__(self):
        self.next_id = 0
        self.free_ids: List[int] = []
        self.live = 0
        self.peak = 0

    def allocate(self, n: int) -> List[int]:
        out = []
        for _ in range(int(n)):
            if self.free_ids:
                out.append(self.free_ids.pop())
            else:
                out.append(self.next_id)
                self.next_id += 1
        self.live += len(out)
        self.peak = max(self.peak, self.live)
        return out

    def free(self, ids) -> None:
        ids = [int(q) for q in np.ravel(ids)]
        self.free_ids.extend(ids)
        self.free_ids.sort(reverse=True)
        self.live -= len(ids)


@dataclass
class StreamContext:
    alloc: QubitAllocator = field(default_factory=QubitAllocator)
    n_cbits: int = 0
    cache: Dict[Any, Any] = field(default_factory=dict)
    cache_limit: int = 20000
    record: Optional[Counter] = None          # if set, counts every bloq type decomposed
    greedy: bool = True                       # order each decomposition greedily (allocations as
                                              # late and frees as early as possible), as the
                                              # paper's qubit counts assume; False: Qualtran 0.4's
                                              # iter_bloqnections order
    compiled: bool = True                     # replay each distinct bloq from a template recorded
                                              # once (compiled.py); False walks every instance
    fast: bool = True                         # use `stream_ops` where a bloq has it
    named_cbits: Dict[Any, int] = field(default_factory=dict)   # outcome names -> classical bits
    sub_hook: Any = None                      # used by compiled.py while recording a template

    def new_cbit(self) -> int:
        self.n_cbits += 1
        return self.n_cbits - 1

    def decompose(self, bloq: Bloq):
        try:
            hit = self.cache.get(bloq)
            hashable = True
        except TypeError:
            hit, hashable = None, False
        if hit is None:
            cb = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            final = [(cxn.left, cxn.right.reg.name, cxn.right.idx) for cxn in cb.connections
                     if cxn.right.binst is RightDangle]
            steps = list(greedy_bloqnections(cb)) if self.greedy else list(cb.iter_bloqnections())
            hit = (cb, steps, final)
            if hashable and len(self.cache) < self.cache_limit:
                self.cache[bloq] = hit
        return hit


# ------------------------------------------------------------------------------------------------
# the order of a decomposition
# ------------------------------------------------------------------------------------------------

_ALLOCATION_PRIORITY = int(1e16)


def _priority(node) -> int:
    from qualtran._infra.quantum_graph import DanglingT
    if isinstance(node, DanglingT):
        return 0
    if isinstance(node.bloq, Allocate):
        return _ALLOCATION_PRIORITY
    if isinstance(node.bloq, Free):
        return -_ALLOCATION_PRIORITY
    sig = node.bloq.signature
    return (sum(r.total_bits() for r in sig.rights())
            - sum(r.total_bits() for r in sig.lefts()))


def greedy_bloqnections(cb: CompositeBloq):
    """`cb.iter_bloqnections()` in a greedy topological order: a stable lexicographic
    topological sort that runs allocations as late and frees as early as possible, then bloqs that
    release qubits before bloqs that take them, ties in insertion order.  Qualtran 0.4 sorts by
    topological generations instead, which runs every allocation of a decomposition at its start.
    The qubit counts of the paper and of the reference implementation assume the greedy order
    (equivalently, a qubit is allocated at its first use and released at its last); it is the
    order of Qualtran 0.7, which the reference implementation was written for."""
    import networkx as nx
    from qualtran._infra.composite_bloq import _binst_to_cxns
    from qualtran._infra.quantum_graph import DanglingT
    g = cb._binst_graph
    for binst in nx.lexicographical_topological_sort(g, key=_priority):
        if isinstance(binst, DanglingT):
            continue
        pred, succ = _binst_to_cxns(binst, binst_graph=g)
        yield binst, pred, succ


# ------------------------------------------------------------------------------------------------
# leaves
# ------------------------------------------------------------------------------------------------

def _q(a) -> int:
    return int(np.ravel(a)[0])


def _leaf(bloq: Bloq, q: Dict[str, np.ndarray], ctx: StreamContext):
    tp = type(bloq)
    if tp is Split:
        return {'reg': q['reg'].reshape(-1, 1)}
    if tp is Join:
        return {'reg': q['reg'].reshape(-1)}
    if tp is Cast:
        return {'reg': q['reg']}
    if tp is Partition:
        if bloq.partition:
            flat = q['x'].reshape(-1)
            out, k = {}, 0
            for reg in bloq.regs:
                n = reg.total_bits()
                out[reg.name] = flat[k:k + n].reshape(reg.shape + (reg.bitsize,))
                k += n
            return out
        return {'x': np.concatenate([q[reg.name].reshape(-1) for reg in bloq.regs])}
    if tp is Allocate:
        ids = ctx.alloc.allocate(bloq.dtype.num_qubits)
        for i in ids:
            yield ('ALLOC', (i,), None)
        return {'reg': np.array(ids, dtype=np.int64)}
    if tp is Free:
        ids = [int(i) for i in np.ravel(q['reg'])]
        for i in ids:
            yield ('FREE', (i,), None)
        ctx.alloc.free(ids)
        return {}
    if tp is XGate:
        yield ('X', (_q(q['q']),), None)
        return q
    if tp is YGate:
        yield ('Y', (_q(q['q']),), None)
        return q
    if tp is ZGate:
        yield ('Z', (_q(q['q']),), None)
        return q
    if tp is Hadamard:
        yield ('H', (_q(q['q']),), None)
        return q
    if tp is SGate:
        yield ('SDG' if bloq.is_adjoint else 'S', (_q(q['q']),), None)
        return q
    if tp is TGate:
        yield ('TDG' if bloq.is_adjoint else 'T', (_q(q['q']),), None)
        return q
    if tp is CNOT:
        yield ('CNOT', (_q(q['ctrl']), _q(q['target'])), None)
        return q
    if tp is CZPowGate:
        a, b = (int(x) for x in np.ravel(q['q']))
        yield ('CZPOW', (a, b), float(bloq.exponent))
        return q
    if tp is Toffoli:
        c1, c2 = (int(x) for x in np.ravel(q['ctrl']))
        yield ('TOFFOLI', (c1, c2, _q(q['target'])), None)
        return q
    if tp is ZPowGate:
        if getattr(bloq, 'global_shift', 0):
            raise ValueError('ZPowGate with a global shift is not a leaf here')
        yield ('ZPOW', (_q(q['q']),), float(bloq.exponent))
        return q
    if tp is Rz:
        yield ('RZ', (_q(q['q']),), float(bloq.angle))
        return q
    if tp is Ry:
        yield ('RY', (_q(q['q']),), float(bloq.angle))
        return q
    if tp is And:
        c1, c2 = (int(x) for x in np.ravel(q['ctrl']))
        cv = (int(bloq.cv1), int(bloq.cv2))
        if bloq.uncompute:
            t = _q(q['target'])
            yield ('ANDDG', (c1, c2, t), cv)
            yield ('FREE', (t,), None)
            ctx.alloc.free([t])
            return {'ctrl': q['ctrl']}
        (t,) = ctx.alloc.allocate(1)
        yield ('ALLOC', (t,), None)
        yield ('AND', (c1, c2, t), cv)
        return {'ctrl': q['ctrl'], 'target': np.array([t], dtype=np.int64)}
    raise TypeError(f'not a leaf: {bloq}')


def walk(bloq: Bloq, q: Dict[str, np.ndarray], ctx: StreamContext):
    """Generator of leaf ops; its return value maps the bloq's right registers to qubit ids."""
    if type(bloq) in LEAF_TYPES:
        out = yield from _leaf(bloq, q, ctx)
        return out
    if ctx.record is not None:
        ctx.record[type(bloq).__name__] += 1
    if getattr(bloq, 'measured', False):
        out = yield from _walk_cirq(bloq, q, ctx)
        return out
    if ctx.fast and hasattr(bloq, 'stream_ops'):
        out = yield from bloq.stream_ops(q, ctx)
        return out
    cb, steps, final = ctx.decompose(bloq)
    soqmap: Dict[Soquet, np.ndarray] = {}
    for reg in cb.signature.lefts():
        arr = np.asarray(q[reg.name], dtype=np.int64).reshape(reg.shape + (reg.bitsize,))
        for idx in reg.all_idxs():
            soqmap[Soquet(LeftDangle, reg, idx)] = arr[idx].reshape(-1)
    for binst, preds, succs in steps:
        sub = binst.bloq
        inq = {reg.name: np.empty(reg.shape + (reg.bitsize,), dtype=np.int64)
               for reg in sub.signature.lefts()}
        for cxn in preds:
            arr = soqmap.pop(cxn.left)
            if cxn.right.reg.shape:
                inq[cxn.right.reg.name][cxn.right.idx] = arr
            else:
                inq[cxn.right.reg.name][...] = arr
        if ctx.sub_hook is not None and type(sub) not in LEAF_TYPES:
            outq = yield from ctx.sub_hook(sub, inq)
        else:
            outq = yield from walk(sub, inq, ctx)
        for cxn in succs:
            src = np.asarray(outq[cxn.left.reg.name])
            soqmap[cxn.left] = (src[cxn.left.idx] if cxn.left.reg.shape else src).reshape(-1)
    out = {reg.name: np.empty(reg.shape + (reg.bitsize,), dtype=np.int64)
           for reg in cb.signature.rights()}
    for left, name, idx in final:
        if idx:
            out[name][idx] = soqmap.pop(left)
        else:
            out[name][...] = soqmap.pop(left)
    return out


def _walk_cirq(bloq: Bloq, q: Dict[str, np.ndarray], ctx: StreamContext):
    """Follow the cirq decomposition of a measuring bloq (see `measured_ops`)."""
    out = yield from bloq.measured_ops(q, ctx)
    return out


def flatten_registers(q: Dict[str, Any], regs) -> List[int]:
    """The qubit ids of registers `regs` (signature order, each as Qualtran splits it) as a list."""
    out: List[int] = []
    for reg in regs:
        out.extend(int(x) for x in np.ravel(q[reg.name]))
    return out


def unflatten_registers(flat: List[int], regs) -> Dict[str, np.ndarray]:
    out, k = {}, 0
    for reg in regs:
        n = reg.total_bits()
        out[reg.name] = np.array(flat[k:k + n], dtype=np.int64).reshape(reg.shape + (reg.bitsize,))
        k += n
    return out


class BloqStream:
    """Iterate over the leaf ops of `bloq`.  Left registers get fresh qubits (announced by ALLOC)
    unless `inputs` gives their ids.  After iteration `inputs` and `outputs` map register names to
    qubit ids and `ctx.alloc.peak` is the peak number of live qubits.  With `ctx.compiled` (the
    default) each distinct bloq is replayed from a template (compiled.py), which gives the same
    ops as walking every instance."""

    def __init__(self, bloq: Bloq, ctx: Optional[StreamContext] = None,
                 inputs: Optional[Dict[str, Any]] = None):
        self.bloq = bloq
        self.ctx = ctx or StreamContext()
        self._given = inputs or {}
        self.inputs: Dict[str, np.ndarray] = {}
        self.outputs: Optional[Dict[str, np.ndarray]] = None

    def __iter__(self) -> Iterator[Op]:
        ctx = self.ctx
        for reg in self.bloq.signature.lefts():
            shape = reg.shape + (reg.bitsize,)
            if reg.name in self._given:
                self.inputs[reg.name] = np.asarray(self._given[reg.name], dtype=np.int64).reshape(shape)
            else:
                ids = ctx.alloc.allocate(reg.total_bits())
                for i in ids:
                    yield ('ALLOC', (i,), None)
                self.inputs[reg.name] = np.array(ids, dtype=np.int64).reshape(shape)
        b = self.bloq
        if (ctx.compiled and type(b) not in LEAF_TYPES and not getattr(b, 'measured', False)
                and not hasattr(b, 'stream_ops')):
            from .compiled import replay, template
            lefts, rights = list(b.signature.lefts()), list(b.signature.rights())
            out = yield from replay(template(b, ctx), flatten_registers(self.inputs, lefts), ctx)
            self.outputs = unflatten_registers(out, rights)
        else:
            self.outputs = yield from walk(b, dict(self.inputs), ctx)


# ------------------------------------------------------------------------------------------------
# counting
# ------------------------------------------------------------------------------------------------

def is_clifford_power(t: float) -> Optional[int]:
    """k if Z**t is within ANGLE_ATOL of Z**(k/4) (k mod 8), else None."""
    t = float(t) % 2.0
    k = round(t * 4)
    return k % 8 if abs(t - k / 4) < ANGLE_ATOL else None


def count(ops: Iterable[Op]) -> Counter:
    """Counts by leaf name; conditional gates are counted as NAME_IF."""
    c = Counter()
    for name, qs, arg in ops:
        c[name + ('_IF' if isinstance(arg, Condition) else '')] += 1
    return c


def paper_counts(ops: Iterable[Op]) -> Counter:
    """Counts in the conventions of the paper's gate stream (Table V): an AND or AND-dagger with a
    control value 0 carries two X gates per such control; ZPOW, RZ and CZPOW within 1e-12 of a
    Clifford+T gate count as that gate (T, S, Z ...); other ZPOW and RZ count as RZ."""
    c = Counter()
    for name, qs, arg in ops:
        cond = isinstance(arg, Condition)
        sfx = '_IF' if cond else ''
        if name in ('AND', 'ANDDG'):
            cv = arg if isinstance(arg, tuple) else (1, 1)
            if sum(cv) < 2:
                c['X'] += 2 * (2 - sum(cv))
            c[name] += 1
        elif name == 'ZPOW':
            k = is_clifford_power(arg)
            if k is None:
                c['RZ' + sfx] += 1
            else:
                for g in {0: [], 1: ['T'], 2: ['S'], 3: ['S', 'T'], 4: ['Z'], 5: ['Z', 'T'],
                          6: ['SDG'], 7: ['TDG']}[k]:
                    c[g + sfx] += 1
        elif name == 'RZ':
            k = is_clifford_power(arg / math.pi)
            if k is None:
                c['RZ' + sfx] += 1
            else:
                for g in {0: [], 1: ['T'], 2: ['S'], 3: ['S', 'T'], 4: ['Z'], 5: ['Z', 'T'],
                          6: ['SDG'], 7: ['TDG']}[k]:
                    c[g + sfx] += 1
        elif name == 'CZPOW':
            if abs((arg % 2.0) - 1.0) < ANGLE_ATOL:
                c['CZ' + sfx] += 1
            elif abs(arg % 2.0) < ANGLE_ATOL:
                pass
            else:
                c['CZPOW' + sfx] += 1
        elif name in ('ALLOC', 'FREE'):
            c[name] += 1
        else:
            c[name + sfx] += 1
    return c


def t_count(c: Counter, t_per_rotation: int = 0) -> int:
    """T gates of paper counts: T and TDG, 4 per AND, 4 per Toffoli, t_per_rotation per RZ, RY."""
    return (c['T'] + c['TDG'] + 4 * c['AND'] + 4 * c['TOFFOLI']
            + t_per_rotation * (c['RZ'] + c['RY']))


# ------------------------------------------------------------------------------------------------
# sparse state-vector simulation
# ------------------------------------------------------------------------------------------------

class SparseSim:
    """The state as basis indices (rows of 64-bit words) and amplitudes over the live qubits.

    FREE asserts the freed qubit is |0>; ANDDG is simulated as Gidney's uncompute: with
    `measure_ands` an X measurement of the target with a random outcome and the CZ fix-up,
    otherwise a check that the target equals the AND and its removal.  MZ draws outcomes from
    `seed`."""

    PHASES = {'Z': -1.0 + 0j, 'S': 1j, 'SDG': -1j, 'T': np.exp(1j * np.pi / 4),
              'TDG': np.exp(-1j * np.pi / 4)}

    def __init__(self, seed: int = 0, tol: float = 1e-14, words: int = 1, measure_ands=False):
        self.W = words
        self.idx = np.zeros((self.W, 1), dtype=np.uint64)
        self.amp = np.ones(1, dtype=complex)
        self.slot: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(64 * self.W - 1, -1, -1))
        self.cbits: Dict[int, int] = {}
        self.rng = np.random.default_rng(seed)
        self.tol = tol
        self.measure_ands = measure_ands
        self._k = 10 ** 6

    def _ws(self, q):
        s = self.slot[q]
        return s >> 6, np.uint64(s & 63)

    def bit(self, q) -> np.ndarray:
        w, s = self._ws(q)
        return (self.idx[w] >> s) & np.uint64(1)

    def _grow(self):
        self.idx = np.vstack([self.idx, np.zeros((1, self.idx.shape[1]), dtype=np.uint64)])
        self.free_slots = list(range(64 * (self.W + 1) - 1, 64 * self.W - 1, -1)) + self.free_slots
        self.W += 1

    def _compress(self):
        if self.idx.shape[1] > 1:
            order = np.lexsort(self.idx)
            idx, amp = self.idx[:, order], self.amp[order]
            new = np.any(idx[:, 1:] != idx[:, :-1], axis=0)
            starts = np.concatenate([[0], np.nonzero(new)[0] + 1])
            summed = np.add.reduceat(amp, starts)
            idx = idx[:, starts]
        else:
            idx, summed = self.idx, self.amp
        keep = np.abs(summed) > self.tol
        self.idx, self.amp = idx[:, keep], summed[keep]

    def _x(self, q):
        w, s = self._ws(q)
        self.idx[w] ^= (np.uint64(1) << s)

    def _phase(self, q, ph):
        b = self.bit(q) == 1
        self.amp = np.where(b, self.amp * ph, self.amp)

    def _h(self, q):
        w, s = self._ws(q)
        b = (self.idx[w] >> s) & np.uint64(1)
        i0 = self.idx.copy()
        i0[w] &= ~(np.uint64(1) << s)
        i1 = i0.copy()
        i1[w] |= (np.uint64(1) << s)
        r = 1 / math.sqrt(2)
        self.idx = np.hstack([i0, i1])
        self.amp = np.concatenate([self.amp * r, self.amp * r * np.where(b == 1, -1, 1)])
        self._compress()

    def _measure_z(self, q) -> int:
        w, s = self._ws(q)
        b = (self.idx[w] >> s) & np.uint64(1)
        p1 = float(np.sum(np.abs(self.amp[b == 1]) ** 2) / np.sum(np.abs(self.amp) ** 2))
        outcome = int(self.rng.random() < p1)
        keep = b == outcome
        self.idx, self.amp = self.idx[:, keep], self.amp[keep]
        self.amp = self.amp / math.sqrt(float(np.sum(np.abs(self.amp) ** 2)))
        return outcome

    def apply(self, op: Op):
        name, qs, arg = op
        if isinstance(arg, Condition):
            if not arg.evaluate(self.cbits):
                return
            arg = None
        if name == 'ALLOC':
            if not self.free_slots:
                self._grow()
            self.slot[qs[0]] = self.free_slots.pop()
        elif name == 'FREE':
            b = self.bit(qs[0]) == 1
            if np.any(np.abs(self.amp[b]) > 1e-9):
                raise AssertionError(f'qubit {qs[0]} freed while not |0> '
                                     f'(weight {np.sum(np.abs(self.amp[b]) ** 2):.3e})')
            self.idx, self.amp = self.idx[:, ~b], self.amp[~b]
            self.free_slots.append(self.slot.pop(qs[0]))
        elif name == 'X':
            self._x(qs[0])
        elif name == 'Y':
            b = self.bit(qs[0])
            self.amp = self.amp * np.where(b == 0, 1j, -1j)
            self._x(qs[0])
        elif name in self.PHASES:
            self._phase(qs[0], self.PHASES[name])
        elif name == 'H':
            self._h(qs[0])
        elif name == 'ZPOW':
            self._phase(qs[0], np.exp(1j * math.pi * arg))
        elif name == 'RZ':
            b = self.bit(qs[0])
            self.amp = self.amp * np.where(b == 0, np.exp(-0.5j * arg), np.exp(0.5j * arg))
        elif name == 'RY':
            # R_y(theta) = S H R_z(theta) H S^dagger
            q0 = qs[0]
            self._phase(q0, -1j)
            self._h(q0)
            b = self.bit(q0)
            self.amp = self.amp * np.where(b == 0, np.exp(-0.5j * arg), np.exp(0.5j * arg))
            self._h(q0)
            self._phase(q0, 1j)
        elif name == 'CNOT':
            c, t = qs
            wt, st = self._ws(t)
            self.idx[wt] ^= self.bit(c) << st
        elif name in ('CZPOW', 'CZ'):
            ph = -1.0 if name == 'CZ' or arg is None else np.exp(1j * math.pi * arg)
            both = (self.bit(qs[0]) & self.bit(qs[1])) == 1
            self.amp = np.where(both, self.amp * ph, self.amp)
        elif name == 'TOFFOLI':
            c1, c2, t = qs
            wt, st = self._ws(t)
            self.idx[wt] ^= (self.bit(c1) & self.bit(c2)) << st
        elif name == 'AND':
            c1, c2, t = qs
            cv = arg if isinstance(arg, tuple) else (1, 1)
            if np.any(self.bit(t)):
                raise AssertionError('AND target not |0>')
            wt, st = self._ws(t)
            v1 = self.bit(c1) ^ np.uint64(1 - cv[0])
            v2 = self.bit(c2) ^ np.uint64(1 - cv[1])
            self.idx[wt] |= (v1 & v2) << st
        elif name == 'ANDDG':
            c1, c2, t = qs
            cv = arg if isinstance(arg, tuple) else (1, 1)
            v1 = self.bit(c1) ^ np.uint64(1 - cv[0])
            v2 = self.bit(c2) ^ np.uint64(1 - cv[1])
            if self.measure_ands:
                # Gidney: X-measure the target; on outcome 1 apply CZ to the (flipped) controls
                self._h(t)
                outcome = self._measure_z(t)
                if outcome:
                    v1 = self.bit(c1) ^ np.uint64(1 - cv[0])
                    v2 = self.bit(c2) ^ np.uint64(1 - cv[1])
                    self.amp = np.where((v1 & v2) == 1, -self.amp, self.amp)
                wt, st = self._ws(t)
                self.idx[wt] &= ~(np.uint64(1) << st)           # reset to |0>
                self._compress()
            else:
                bad = (v1 & v2) != self.bit(t)
                if np.any(np.abs(self.amp[bad]) > 1e-9):
                    raise AssertionError('ANDDG target is not the AND of its controls')
                wt, st = self._ws(t)
                self.idx[wt] &= ~(np.uint64(1) << st)
        elif name in ('MZ', 'MX'):
            if name == 'MX':
                self._h(qs[0])
            outcome = self._measure_z(qs[0])
            w, s = self._ws(qs[0])
            self.idx[w] &= ~(np.uint64(1) << s)           # reset to |0>
            self._compress()
            self.cbits[arg] = outcome
        else:
            raise ValueError(f'unknown op {name}')

    def run(self, ops: Iterable[Op]) -> 'SparseSim':
        for op in ops:
            self.apply(op)
        return self

    def norm(self) -> float:
        return float(np.sqrt(np.sum(np.abs(self.amp) ** 2)))

    def vector(self, qubits: List[int]) -> np.ndarray:
        """Dense vector over `qubits` (qubits[0] most significant); every other live qubit must be
        |0> in every branch."""
        for q in self.slot:
            if q not in qubits and np.any(np.abs(self.amp[self.bit(q) == 1]) > 1e-9):
                raise AssertionError(f'qubit {q} outside the requested set is not |0>')
        n = len(qubits)
        k = np.zeros(self.amp.size, dtype=np.int64)
        for j, q in enumerate(qubits):
            k |= self.bit(q).astype(np.int64) << (n - 1 - j)
        vec = np.zeros(2 ** n, dtype=complex)
        np.add.at(vec, k, self.amp)
        return vec

    def amplitude_all_zero(self) -> complex:
        zero = np.all(self.idx == 0, axis=0)
        return complex(self.amp[zero].sum())


# ------------------------------------------------------------------------------------------------
# the decompositions a compiler walks
# ------------------------------------------------------------------------------------------------

def bloq_key(b: Bloq):
    """A dictionary key for a bloq: the bloq itself when it is hashable, else its repr."""
    try:
        hash(b)
        return b
    except TypeError:
        return repr(b)


def iter_decompositions(bloq: Bloq, max_depth: int = 64, skip=None):
    """Each distinct composite bloq reachable from `bloq`, depth first, once: tuples (depth, bloq,
    cbloq), where cbloq is bloq.decompose_bloq(), or the exception it raised.  Leaves, bloqs with
    `measured = True` and bloqs for which `skip(bloq)` is true are not decomposed.  A compiler
    walking the same bloq holds at most one such cbloq per depth at a time, so
    len(cbloq.bloq_instances) bounds the memory of each level."""
    seen = set()
    stack = [(0, bloq)]
    while stack:
        depth, b = stack.pop()
        if (type(b) in LEAF_TYPES or getattr(b, 'measured', False) or depth > max_depth
                or (skip is not None and skip(b))):
            continue
        k = bloq_key(b)
        if k in seen:
            continue
        seen.add(k)
        try:
            cb = b if isinstance(b, CompositeBloq) else b.decompose_bloq()
        except Exception as e:                         # reported by the caller
            yield depth, b, e
            continue
        yield depth, b, cb
        kids = sorted((bi.bloq for bi in cb.bloq_instances if type(bi.bloq) not in LEAF_TYPES),
                      key=repr)
        stack.extend((depth + 1, c) for c in reversed(kids))


def largest_level(bloq: Bloq, skip=None):
    """(number of bloq instances, bloq) for the largest single decomposition reachable from
    `bloq`, counting every instance including Split, Join, Allocate and Free; bloqs for which
    `skip(bloq)` is true are not decomposed."""
    best = (0, bloq)
    for _, b, cb in iter_decompositions(bloq, skip=skip):
        if isinstance(cb, Exception):
            raise cb
        n = len(cb.bloq_instances)
        if n > best[0]:
            best = (n, b)
    return best


# ------------------------------------------------------------------------------------------------
# leaf report (what the compiler will see)
# ------------------------------------------------------------------------------------------------

def leaf_report(bloq: Bloq) -> Dict[str, Any]:
    """Walk `bloq` and report the leaf gates reached (Qualtran class names and counts), the
    composite bloqs passed through, and the peak qubit count."""
    ctx = StreamContext(record=Counter(), compiled=False)
    bs = BloqStream(bloq, ctx)
    leaves = Counter()
    for name, qs, arg in bs:
        if name in ('ALLOC', 'FREE'):
            continue
        leaves[name + ('_IF' if isinstance(arg, Condition) else '')] += 1
    return dict(leaves=leaves, decomposed=ctx.record, peak=ctx.alloc.peak)

"""Table lookups in Qualtran 0.4: unary iteration, the select-swap lookup, and its
measurement-based uncompute (Berry et al., arXiv:1902.02134, App. C).

Every bloq here decomposes through `decompose_bloq()` into small decompositions: a unary iteration
is a tree of bloqs, one per node, each with its two children, an And, a CNOT and an And-dagger, and
a leaf writes one table entry with one `FanOut` per slot.  The bloqs hold the table through a
`LookupSpec`, compared and hashed by a digest of its contents, never as an array.  The only bloqs
that do not decompose into bloqs are `MeasureX` (X-basis measurements) and `PhaseLeaf` (CZ gates
conditioned on parities of those outcomes): as Qualtran's And-dagger does, they give their gates as
cirq operations from `decompose_from_registers`, and set `measured = True`.

`UnaryLoad` and `PhaseFixup` also have `stream_ops`, a generator of exactly the gates of their
decomposition trees, which the walker uses (`ctx.fast`) so that a lookup over 2^20 entries is
streamed without building 10^6 Qualtran objects.  tests/test_lookup.py checks that both give the
same gates.

Conventions.  Column k of the table holds integers of widths[k] bits.  A select-swap slot holds the
concatenation of the columns, column 0 first, each most significant bit first, so slot qubit 0
(in Qualtran's order) is the most significant bit of column 0.  The one-hot register of the
uncompute has 2^kk qubits; its qubit l (in Qualtran's order) is 1 when the kk low selection bits
equal l.  The outcome of slot l, qubit beta is named f'{key}_{l}_{beta}' in the cirq decompositions;
every use of an uncompute with the same key reuses these names, as Qualtran's And-dagger reuses the
name of its target, and a condition always refers to the latest outcome of that name, measured
earlier in the same uncompute.
"""
from __future__ import annotations

import hashlib
import math
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen

from qualtran import (Adjoint, Bloq, BloqBuilder, BoundedQUInt, GateWithRegisters, QAny, QBit,
                      QUInt, Register, Side, Signature, SoquetT)
from qualtran.bloqs.mcmt import And
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

from .cirq_stream import ParityCondition
from .gates import RouteToZero, swap_sequence
from .stream import Condition, StreamContext
from .wires import Wires


def _dt(n: int):
    return QBit() if n == 1 else QAny(n)


def _udt(n: int):
    return QBit() if n == 1 else QUInt(n)


def _bits(bb: BloqBuilder, soq, n: int) -> List:
    """The qubit soquets of an n-qubit register, in Qualtran's order (most significant first)."""
    return [soq] if n == 1 else list(bb.split(soq))


def _join(bb: BloqBuilder, bits: Sequence, dtype=None):
    bits = list(bits)
    if len(bits) == 1 and (dtype is None or isinstance(dtype, QBit)):
        return bits[0]
    return bb.join(np.array(bits), dtype=dtype if dtype is not None else QUInt(len(bits)))


# ------------------------------------------------------------------------------------------------
# the table
# ------------------------------------------------------------------------------------------------

class LookupSpec:
    """A table of `columns` (2^index_bits integers each, of `widths` bits) looked up with
    2^log_block select-swap slots and uncomputed with a one-hot register of 2^log_block_unc qubits.
    Two specs are equal when their digests are: the index size, widths, block sizes and contents."""

    def __init__(self, index_bits: int, widths: Sequence[int], columns, log_block: int,
                 log_block_unc: int):
        self.index_bits = int(index_bits)
        self.widths = tuple(int(w) for w in widths)
        self.log_block = int(log_block)
        self.log_block_unc = int(log_block_unc)
        N = 2 ** self.index_bits
        cols = [[int(x) & ((1 << w) - 1) for x in c] for c, w in zip(columns, self.widths)]
        if len(cols) != len(self.widths) or any(len(c) != N for c in cols):
            raise ValueError('need one column of 2^index_bits entries per width')
        if not 0 <= self.log_block <= self.index_bits:
            raise ValueError('log_block must be between 0 and index_bits')
        words = []
        for e in range(N):
            v = 0
            for c, w in zip(cols, self.widths):
                v = (v << w) | c[e]
            words.append(v)
        self.columns = tuple(tuple(c) for c in cols)
        self.words = words
        h = hashlib.sha256(repr((self.index_bits, self.widths, self.log_block,
                                 self.log_block_unc)).encode())
        h.update(np.array(words, dtype=object).tobytes() if self.W > 62 else
                 np.array(words, dtype=np.int64).tobytes())
        self.digest = h.hexdigest()

    @property
    def W(self) -> int:
        return sum(self.widths)

    @property
    def nslot(self) -> int:
        return 1 << self.log_block

    @property
    def kk(self) -> int:
        """Qubits of the uncompute's one-hot encoding (the low selection bits it covers)."""
        return min(self.log_block_unc, self.index_bits)

    def __eq__(self, other):
        return isinstance(other, LookupSpec) and self.digest == other.digest

    def __hash__(self):
        return hash(self.digest)

    def __repr__(self):
        return (f'LookupSpec(index_bits={self.index_bits}, widths={self.widths}, '
                f'log_block={self.log_block}, log_block_unc={self.log_block_unc}, '
                f'digest={self.digest[:10]})')

    def slot_values(self, j: int) -> List[int]:
        """The words in the 2^log_block slots after looking up index j: slot 0 holds entry j."""
        lam = self.log_block
        hi, lo = j >> lam, j & ((1 << lam) - 1)
        slots = [self.words[(hi << lam) + l] for l in range(1 << lam)]
        for bit, a, b in swap_sequence(lam, 1 << lam):
            if (lo >> bit) & 1:
                slots[a], slots[b] = slots[b], slots[a]
        return slots

    def parity_positions(self, j: int) -> List[Tuple[int, int]]:
        """(slot l, qubit beta) of every measured qubit whose outcome enters the phase f(j): the
        qubits that hold a 1 after the lookup of j."""
        W = self.W
        return [(l, beta) for l, v in enumerate(self.slot_values(j)) for beta in range(W)
                if (v >> (W - 1 - beta)) & 1]


def _optimal_log_block(data_size: int, bitsize: int, adjoint: bool) -> int:
    """The reference implementation's choice of select-swap block size (the rule of Qualtran
    0.7's QROAMClean and QROAMCleanAdjoint; Berry et al., arXiv:1902.02134, App. B)."""
    k = 0.5 * math.log2(data_size) if adjoint else 0.5 * math.log2(data_size / bitsize)
    if k < 0:
        return 0

    def cost(x):
        return data_size / x + x if adjoint else data_size / x + bitsize * (x - 1)

    lo, hi = math.floor(k), math.ceil(k)
    return int(lo if cost(2 ** lo) <= cost(2 ** hi) else hi)


def default_lookup_spec(index_bits: int, widths: Sequence[int], columns) -> LookupSpec:
    """Block sizes as the reference implementation chooses them for the lookup and its uncompute."""
    N = 2 ** index_bits
    lam = _optimal_log_block(N, sum(widths), adjoint=False)
    kun = _optimal_log_block(N, sum(widths), adjoint=True)
    return LookupSpec(index_bits, widths, columns, lam, kun)


# ------------------------------------------------------------------------------------------------
# unary iteration as a generator (the fast path)
# ------------------------------------------------------------------------------------------------

def unary_iteration(sel_lsb: Sequence[int], ctx: StreamContext, leaf):
    """Gates of the unary iteration over the selection qubits `sel_lsb` (least significant first):
    leaf(v, ctrl) yields the gates for the value v, under ctrl = (qubit, value), meaning that
    "qubit == value" holds exactly for v, or ctrl = None when there are no selection qubits.
    The split on the most significant qubit uses that qubit itself; every other split is an And
    (2^d - 2 in all for d >= 1 qubits).  The same gates as the trees of UnaryLoad and PhaseFixup."""
    d = len(sel_lsb)
    if d == 0:
        yield from leaf(0, None)
        return

    def rec(ctrl, level, prefix):
        if level < 0:
            yield from leaf(prefix, ctrl)
            return
        b = sel_lsb[level]
        q, cv = ctrl
        (t,) = ctx.alloc.allocate(1)
        yield ('ALLOC', (t,), None)
        yield ('AND', (q, b, t), (cv, 0))                  # t = [q = cv] and [b = 0]
        yield from rec((t, 1), level - 1, 2 * prefix)
        if cv == 0:
            yield ('X', (q,), None)
        yield ('CNOT', (q, t), None)                       # t = [q = cv] and [b = 1]
        if cv == 0:
            yield ('X', (q,), None)
        yield from rec((t, 1), level - 1, 2 * prefix + 1)
        yield ('ANDDG', (q, b, t), (cv, 1))
        yield ('FREE', (t,), None)
        ctx.alloc.free([t])

    top = sel_lsb[d - 1]
    yield from rec((top, 0), d - 2, 0)
    yield from rec((top, 1), d - 2, 1)


class _IterationNode:
    """The decomposition of one node of a unary iteration, shared by LoadNode and PhaseNode:
    t = [ctrl = cv] and [b = 0] by an And, the first child under t, t flipped to
    [ctrl = cv] and [b = 1] by a CNOT (with X on ctrl when cv = 0), the second child, And-dagger.
    `payload` maps payload register names to soquets and is threaded through the children."""

    @staticmethod
    def build(bb: BloqBuilder, ctrl, sel, level: int, cv: int, child, payload: Dict):
        bits = _bits(bb, sel, level + 1)
        b, lower = bits[0], bits[1:]
        (ctrl, b), t = bb.add(And(cv, 0), ctrl=np.array([ctrl, b]))
        lower_soq = _join(bb, lower, _udt(len(lower))) if lower else None
        t, lower_soq, payload = child(bb, t, lower_soq, payload, 0)
        if cv == 0:
            ctrl = bb.add(XGate(), q=ctrl)
        ctrl, t = bb.add(CNOT(), ctrl=ctrl, target=t)
        if cv == 0:
            ctrl = bb.add(XGate(), q=ctrl)
        t, lower_soq, payload = child(bb, t, lower_soq, payload, 1)
        lower = _bits(bb, lower_soq, len(lower)) if lower else []
        (ctrl, b) = bb.add(And(cv, 1, uncompute=True), ctrl=np.array([ctrl, b]), target=t)
        return ctrl, _join(bb, [b] + lower, _udt(level + 1)), payload


def _top_level(bb: BloqBuilder, sel, d: int, make, payload: Dict):
    """The two halves of a unary iteration on the most significant selection qubit itself."""
    bits = _bits(bb, sel, d)
    top, lower = bits[0], bits[1:]
    lower_soq = _join(bb, lower, _udt(len(lower))) if lower else None
    for half in (0, 1):
        top, lower_soq, payload = make(bb, top, lower_soq, payload, half)
    lower = _bits(bb, lower_soq, len(lower)) if lower else []
    return _join(bb, [top] + lower, _udt(d)), payload


# ------------------------------------------------------------------------------------------------
# the load: target ^= table entry, under a unary iteration
# ------------------------------------------------------------------------------------------------

@frozen
class FanOut(Bloq):
    r"""`target` $\oplus$= `mask` if `ctrl` = 1: a CNOT from ctrl (an X when not `controlled`) on
    each target qubit whose bit of mask is 1, least significant first."""
    width: int
    mask: int
    controlled: bool = True

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('ctrl', QBit())] if self.controlled else []
        return Signature(regs + [Register('target', _dt(self.width))])

    def build_composite_bloq(self, bb: BloqBuilder, target, ctrl=None) -> Dict[str, SoquetT]:
        w = Wires(bb)
        ts = w.split(target, self.width)
        c = w.put(ctrl) if self.controlled else None
        for p in range(self.width):
            if (self.mask >> p) & 1:
                if self.controlled:
                    w.cnot(c, ts[p])
                else:
                    w.x(ts[p])
        out = {'target': w.join(ts, dtype=_dt(self.width))}
        if self.controlled:
            out['ctrl'] = w.take(c)
        return out

    def adjoint(self):
        return self

    def __str__(self):
        return f'FanOut({self.width}, {self.mask:#x})'


def _slots_register(spec: LookupSpec) -> Register:
    return Register('slots', _dt(spec.W), shape=(spec.nslot,))


@frozen
class LoadLeaf(Bloq):
    r"""Write block `block` of the table into the slots under ctrl = `cv` (X on ctrl around the
    CNOTs when cv = 0): one FanOut per slot, the last slot first (so the CNOTs run from the least
    significant bit of the concatenated block upward)."""
    spec: LookupSpec
    block: int
    cv: int = 1
    controlled: bool = True

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('ctrl', QBit())] if self.controlled else []
        return Signature(regs + [_slots_register(self.spec)])

    def build_composite_bloq(self, bb: BloqBuilder, slots, ctrl=None) -> Dict[str, SoquetT]:
        sp = self.spec
        slots = np.array(slots)
        base = self.block << sp.log_block
        if self.controlled and self.cv == 0:
            ctrl = bb.add(XGate(), q=ctrl)
        for l in reversed(range(sp.nslot)):
            fan = FanOut(sp.W, sp.words[base + l], self.controlled)
            if self.controlled:
                ctrl, slots[l] = bb.add(fan, ctrl=ctrl, target=slots[l])
            else:
                slots[l] = bb.add(fan, target=slots[l])
        if self.controlled and self.cv == 0:
            ctrl = bb.add(XGate(), q=ctrl)
        out = {'slots': slots}
        if self.controlled:
            out['ctrl'] = ctrl
        return out

    def adjoint(self):
        return self

    def __str__(self):
        return f'Load[{self.block}]'


def _load_child(spec: LookupSpec, level: int, prefix: int):
    """The child maker of a load node at `level` (children at level - 1, or leaves)."""
    def child(bb, ctrl, sel, payload, half):
        p = 2 * prefix + half
        if level - 1 < 0:
            ctrl, slots = bb.add(LoadLeaf(spec, p, 1), ctrl=ctrl, slots=payload['slots'])
        else:
            ctrl, sel, slots = bb.add(LoadNode(spec, level - 1, p, 1), ctrl=ctrl, sel=sel,
                                      slots=payload['slots'])
        return ctrl, sel, {'slots': slots}
    return child


@frozen
class LoadNode(Bloq):
    r"""A node of the load's unary iteration: blocks prefix * 2^(level+1) ... of the table under
    ctrl = `cv`, split on `sel`'s most significant qubit (the selection bit of weight 2^level)."""
    spec: LookupSpec
    level: int
    prefix: int
    cv: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), Register('sel', _udt(self.level + 1)),
                          _slots_register(self.spec)])

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, sel, slots) -> Dict[str, SoquetT]:
        ctrl, sel, payload = _IterationNode.build(
            bb, ctrl, sel, self.level, self.cv, _load_child(self.spec, self.level, self.prefix),
            {'slots': np.array(slots)})
        return {'ctrl': ctrl, 'sel': sel, 'slots': payload['slots']}

    def adjoint(self):
        return self

    def __str__(self):
        return f'LoadNode(level {self.level}, prefix {self.prefix})'


@frozen
class UnaryLoad(Bloq):
    r"""`slots` $\oplus$= block `sel` of the table: a unary iteration over the d = index_bits -
    log_block high selection bits ($2^d-2$ ANDs) with a FanOut of each slot's word at each value.
    With d = 0 the block is written with X gates."""
    spec: LookupSpec

    @cached_property
    def d(self) -> int:
        return self.spec.index_bits - self.spec.log_block

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('sel', _udt(self.d))] if self.d else []
        return Signature(regs + [_slots_register(self.spec)])

    def build_composite_bloq(self, bb: BloqBuilder, slots, sel=None) -> Dict[str, SoquetT]:
        sp, d = self.spec, self.d
        slots = np.array(slots)
        if d == 0:
            return {'slots': bb.add(LoadLeaf(sp, 0, 1, controlled=False), slots=slots)}

        def make(bb_, top, lower, payload, half):
            if d == 1:
                top, s = bb_.add(LoadLeaf(sp, half, half), ctrl=top, slots=payload['slots'])
            else:
                top, lower, s = bb_.add(LoadNode(sp, d - 2, half, half), ctrl=top, sel=lower,
                                        slots=payload['slots'])
            return top, lower, {'slots': s}

        sel, payload = _top_level(bb, sel, d, make, {'slots': slots})
        return {'sel': sel, 'slots': payload['slots']}

    def stream_ops(self, q, ctx: StreamContext):
        sp = self.spec
        W, nslot, lam = sp.W, sp.nslot, sp.log_block
        sel = [int(s) for s in np.ravel(q['sel'])[::-1]] if self.d else []
        slots = np.asarray(q['slots']).reshape(nslot, W)

        def leaf(v, ctrl):
            base = v << lam
            if ctrl is None:
                for l in reversed(range(nslot)):
                    word = sp.words[base + l]
                    for p in range(W):
                        if (word >> p) & 1:
                            yield ('X', (int(slots[l, W - 1 - p]),), None)
                return
            c, cv = ctrl
            if cv == 0:
                yield ('X', (c,), None)
            for l in reversed(range(nslot)):
                word = sp.words[base + l]
                for p in range(W):
                    if (word >> p) & 1:
                        yield ('CNOT', (c, int(slots[l, W - 1 - p])), None)
            if cv == 0:
                yield ('X', (c,), None)

        yield from unary_iteration(sel, ctx, leaf)
        return dict(q)

    def adjoint(self):
        return self

    def __str__(self):
        return f'UnaryLoad({2 ** self.d} blocks)'


@frozen
class SelectSwapLookup(Bloq):
    r"""Select-swap lookup with exact controlled swaps (the reference implementation's, which is
    Qualtran 0.7's QROAMClean; Qualtran 0.4 has no QROAMClean).

    $2^\lambda$ new slots of $W$ qubits; UnaryLoad of the block selected by the high $b-\lambda$
    selection bits; RouteToZero on the low $\lambda$ bits moves slot $j \bmod 2^\lambda$ to slot 0.
    Slot 0 is the output (`target0`, `target1`, ... one register per column); the other slots are
    junk, cleared by LookupUncompute.  ANDs: $2^{b-\lambda}-2+(2^\lambda-1)W$."""
    spec: LookupSpec

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('selection', QUInt(self.spec.index_bits))]
                         + lookup_registers(self.spec, Side.RIGHT))

    def build_composite_bloq(self, bb: BloqBuilder, selection) -> Dict[str, SoquetT]:
        sp = self.spec
        b, lam, nslot = sp.index_bits, sp.log_block, sp.nslot
        d = b - lam
        bits = _bits(bb, selection, b)
        hi, lo = bits[:d], bits[d:]
        slots = np.array([bb.allocate(dtype=_dt(sp.W)) for _ in range(nslot)])
        load = UnaryLoad(sp)
        if d:
            hi_soq, slots = bb.add(load, sel=_join(bb, hi, _udt(d)), slots=slots)
            hi = _bits(bb, hi_soq, d)
        else:
            slots = bb.add(load, slots=slots)
        if lam:
            lo_soq, slots = bb.add(RouteToZero(lam, sp.W, nslot),
                                   sel=_join(bb, lo, BoundedQUInt(lam, nslot)), targets=slots)
            lo = _bits(bb, lo_soq, lam)
        out = {'selection': _join(bb, hi + lo, QUInt(b))}
        out.update(_slots_to_columns(bb, sp, slots))
        return out

    def adjoint(self):
        return LookupUncompute(self.spec)

    def __str__(self):
        return 'Lookup'


def lookup_registers(spec: LookupSpec, side: Side) -> List[Register]:
    regs = [Register(f'target{k}', _dt(wk), side=side) for k, wk in enumerate(spec.widths)]
    if spec.nslot > 1:
        regs += [Register(f'junk{k}', _dt(wk), shape=(spec.nslot - 1,), side=side)
                 for k, wk in enumerate(spec.widths)]
    return regs


def _slots_to_columns(bb: BloqBuilder, sp: LookupSpec, slots) -> Dict[str, SoquetT]:
    offs = np.cumsum((0,) + sp.widths)
    out = {}
    per_slot = [_bits(bb, s, sp.W) for s in slots]
    for k, wk in enumerate(sp.widths):
        out[f'target{k}'] = _join(bb, per_slot[0][offs[k]:offs[k + 1]], _dt(wk))
        if sp.nslot > 1:
            out[f'junk{k}'] = np.array([_join(bb, per_slot[l][offs[k]:offs[k + 1]], _dt(wk))
                                        for l in range(1, sp.nslot)])
    return out


def _columns_to_slots(bb: BloqBuilder, sp: LookupSpec, soqs) -> List:
    per_col = []
    for k, wk in enumerate(sp.widths):
        rows = [_bits(bb, soqs[f'target{k}'], wk)]
        if sp.nslot > 1:
            rows += [_bits(bb, s, wk) for s in soqs[f'junk{k}']]
        per_col.append(rows)
    return [_join(bb, [q for k in range(len(sp.widths)) for q in per_col[k][l]], _dt(sp.W))
            for l in range(sp.nslot)]


# ------------------------------------------------------------------------------------------------
# the measurement-based uncompute
# ------------------------------------------------------------------------------------------------

@frozen
class MeasureX(GateWithRegisters):
    r"""Measure the `width` qubits of slot `slot` in the X basis and release them: per qubit $\beta$
    (in Qualtran's order) H, a measurement named f'{key}_{slot}_{beta}', and a reset."""
    width: int
    key: str
    slot: int

    measured = True

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('q', _dt(self.width), side=Side.LEFT)])

    def decompose_from_registers(self, *, context, q):
        for beta, qb in enumerate(np.ravel(q)):
            yield cirq.H(qb)
            yield cirq.measure(qb, key=f'{self.key}_{self.slot}_{beta}')
            yield cirq.reset(qb)

    def measured_ops(self, q, ctx: StreamContext):
        ids = [int(x) for x in np.ravel(q['q'])]
        row = []
        for qid in ids:
            k = ctx.new_cbit()
            row.append(k)
            yield ('MX', (qid,), k)
            yield ('FREE', (qid,), None)
            ctx.alloc.free([qid])
        ctx.named_cbits[(self.key, self.slot)] = np.array(row, dtype=np.int64)
        return {}

    def _t_complexity_(self):
        return TComplexity(clifford=self.width)

    def __str__(self):
        return f'MeasureX(slot {self.slot})'


@frozen
class OneHotEncode(Bloq):
    r"""A new register `oh` of $2^{kk}$ qubits with qubit $l$ (Qualtran's order) equal to
    [`low` = $l$]: X on qubit 0; for bit 0 of `low` two CNOTs; for bit $i\ge1$, $2^i$ ANDs, each
    followed by a CNOT.  $2^{kk}-2$ ANDs for $kk\ge1$.  Its inverse is Qualtran's Adjoint."""
    kk: int

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('low', _udt(self.kk))] if self.kk else []
        return Signature(regs + [Register('oh', _dt(2 ** self.kk), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, low=None) -> Dict[str, SoquetT]:
        w = Wires(bb)
        kk = self.kk
        lw = w.split(low, kk) if kk else []                 # least significant first
        oh = [None] * (2 ** kk)
        oh[0] = w.alloc(1)[0]
        w.x(oh[0])
        if kk:
            oh[1] = w.alloc(1)[0]
            w.cnot(lw[0], oh[1])
            w.cnot(oh[1], oh[0])
        for i in range(1, kk):
            for m in range(2 ** i):
                oh[m + 2 ** i] = w.and_(oh[m], lw[i])
                w.cnot(oh[m + 2 ** i], oh[m])
        out = {'oh': w.join(oh[::-1], dtype=_dt(2 ** kk)) if kk else w.take(oh[0])}
        if kk:
            out['low'] = w.join(lw, dtype=_udt(kk))
        return out

    def __str__(self):
        return f'OneHot({self.kk})'


class LookupParity(Condition):
    """f(j): the XOR of the outcomes of the measured qubits that hold a 1 after the lookup of j
    (slot l, qubit beta has classical bit key_table[l, beta]).  The keys are found only when read."""

    lazy = True

    def __init__(self, spec: LookupSpec, key_table: np.ndarray, j: int):
        self.spec, self.key_table, self.j = spec, key_table, j

    def keys(self):
        return tuple(int(self.key_table[l, b]) for l, b in self.spec.parity_positions(self.j))

    def evaluate(self, cbits):
        v = 0
        for k in self.keys():
            v ^= cbits[k]
        return v

    def __repr__(self):
        return f'f({self.j})'


def _key_table(ctx: StreamContext, spec: LookupSpec, key: str) -> np.ndarray:
    return np.stack([ctx.named_cbits[(key, l)] for l in range(spec.nslot)])


def _cirq_parity(spec: LookupSpec, key: str, j: int) -> ParityCondition:
    return ParityCondition(key_fn=lambda: [f'{key}_{l}_{b}' for l, b in spec.parity_positions(j)],
                           label=f'f({j})')


@frozen
class PhaseLeaf(GateWithRegisters):
    r"""The phase fix-up for one block of $2^{kk}$ indices $j=(h\cdot2^{kk})+l$: a CZ between `ctrl`
    and qubit $l$ of `oh`, applied when $f(j)=1$ (with X on ctrl around each CZ when `cv` = 0), or,
    without a control, a Z on qubit $l$.  The conditions are parities of the outcomes of
    MeasureX(key, ...)."""
    spec: LookupSpec
    key: str
    h: int
    cv: int = 1
    controlled: bool = True

    measured = True

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('ctrl', QBit())] if self.controlled else []
        return Signature(regs + [Register('oh', _dt(2 ** self.spec.kk))])

    def decompose_from_registers(self, *, context, oh, ctrl=None):
        kk = self.spec.kk
        ohq = list(np.ravel(oh))
        c = np.ravel(ctrl)[0] if self.controlled else None
        for l in range(2 ** kk):
            cond = _cirq_parity(self.spec, self.key, (self.h << kk) + l)
            if c is None:
                yield cirq.Z(ohq[l]).with_classical_controls(cond)
                continue
            if self.cv == 0:
                yield cirq.X(c)
            yield cirq.CZ(c, ohq[l]).with_classical_controls(cond)
            if self.cv == 0:
                yield cirq.X(c)

    def measured_ops(self, q, ctx: StreamContext):
        kk = self.spec.kk
        table = _key_table(ctx, self.spec, self.key)
        oh = [int(x) for x in np.ravel(q['oh'])]
        c = int(np.ravel(q['ctrl'])[0]) if self.controlled else None
        for l in range(2 ** kk):
            cond = LookupParity(self.spec, table, (self.h << kk) + l)
            if c is None:
                yield ('Z', (oh[l],), cond)
                continue
            if self.cv == 0:
                yield ('X', (c,), None)
            yield ('CZ', (c, oh[l]), cond)
            if self.cv == 0:
                yield ('X', (c,), None)
        return dict(q)

    def _t_complexity_(self):
        n = 2 ** self.spec.kk
        return TComplexity(clifford=n * (3 if self.controlled and self.cv == 0 else 1))

    def __str__(self):
        return f'Phases[{self.h}]'


def _phase_child(spec: LookupSpec, key: str, level: int, prefix: int):
    def child(bb, ctrl, sel, payload, half):
        p = 2 * prefix + half
        if level - 1 < 0:
            ctrl, oh = bb.add(PhaseLeaf(spec, key, p, 1), ctrl=ctrl, oh=payload['oh'])
        else:
            ctrl, sel, oh = bb.add(PhaseNode(spec, key, level - 1, p, 1), ctrl=ctrl, sel=sel,
                                   oh=payload['oh'])
        return ctrl, sel, {'oh': oh}
    return child


@frozen
class PhaseNode(Bloq):
    """A node of the fix-up's unary iteration over the high selection bits (as LoadNode)."""
    spec: LookupSpec
    key: str
    level: int
    prefix: int
    cv: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit()), Register('sel', _udt(self.level + 1)),
                          Register('oh', _dt(2 ** self.spec.kk))])

    def build_composite_bloq(self, bb: BloqBuilder, ctrl, sel, oh) -> Dict[str, SoquetT]:
        ctrl, sel, payload = _IterationNode.build(
            bb, ctrl, sel, self.level, self.cv,
            _phase_child(self.spec, self.key, self.level, self.prefix), {'oh': oh})
        return {'ctrl': ctrl, 'sel': sel, 'oh': payload['oh']}

    def __str__(self):
        return f'PhaseNode(level {self.level}, prefix {self.prefix})'


@frozen
class PhaseFixup(Bloq):
    r"""$(-1)^{f(j)}$ on the index $j$: a unary iteration over the $b-kk$ high selection bits
    ($2^{b-kk}-2$ ANDs) whose leaf for the high value $h$ is PhaseLeaf($h$), acting on the one-hot
    encoding of the $kk$ low bits."""
    spec: LookupSpec
    key: str

    @cached_property
    def d(self) -> int:
        return self.spec.index_bits - self.spec.kk

    @cached_property
    def signature(self) -> Signature:
        regs = [Register('high', _udt(self.d))] if self.d else []
        return Signature(regs + [Register('oh', _dt(2 ** self.spec.kk))])

    def build_composite_bloq(self, bb: BloqBuilder, oh, high=None) -> Dict[str, SoquetT]:
        sp, d, key = self.spec, self.d, self.key
        if d == 0:
            return {'oh': bb.add(PhaseLeaf(sp, key, 0, 1, controlled=False), oh=oh)}

        def make(bb_, top, lower, payload, half):
            if d == 1:
                top, o = bb_.add(PhaseLeaf(sp, key, half, half), ctrl=top, oh=payload['oh'])
            else:
                top, lower, o = bb_.add(PhaseNode(sp, key, d - 2, half, half), ctrl=top,
                                        sel=lower, oh=payload['oh'])
            return top, lower, {'oh': o}

        high, payload = _top_level(bb, high, d, make, {'oh': oh})
        return {'high': high, 'oh': payload['oh']}

    def stream_ops(self, q, ctx: StreamContext):
        sp, kk = self.spec, self.spec.kk
        high = [int(s) for s in np.ravel(q['high'])[::-1]] if self.d else []
        oh = [int(x) for x in np.ravel(q['oh'])]
        table = _key_table(ctx, sp, self.key)

        def leaf(h, ctrl):
            for l in range(2 ** kk):
                cond = LookupParity(sp, table, (h << kk) + l)
                if ctrl is None:
                    yield ('Z', (oh[l],), cond)
                    continue
                c, cv = ctrl
                if cv == 0:
                    yield ('X', (c,), None)
                yield ('CZ', (c, oh[l]), cond)
                if cv == 0:
                    yield ('X', (c,), None)

        yield from unary_iteration(high, ctx, leaf)
        return dict(q)

    def __str__(self):
        return 'PhaseFixup'


@frozen
class LookupUncompute(Bloq):
    r"""Measurement-based uncompute of SelectSwapLookup (Berry et al., arXiv:1902.02134, App. C).

    X-measure every target and junk qubit (MeasureX per slot); the outcomes $m$ leave the phase
    $(-1)^{f(j)}$, $f(j)=m\cdot(\text{slot contents after looking up } j)$.  It is removed with a
    one-hot encoding of the $kk$ low selection bits ($2^{kk}-2$ ANDs) and PhaseFixup over the
    high bits ($2^{b-kk}-2$ ANDs), then the encoding undone."""
    spec: LookupSpec
    key: str = 'lookup'

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('selection', QUInt(self.spec.index_bits))]
                         + lookup_registers(self.spec, Side.LEFT))

    def build_composite_bloq(self, bb: BloqBuilder, selection, **soqs) -> Dict[str, SoquetT]:
        sp = self.spec
        b, kk = sp.index_bits, sp.kk
        d = b - kk
        for l, slot in enumerate(_columns_to_slots(bb, sp, soqs)):
            bb.add(MeasureX(sp.W, self.key, l), q=slot)
        bits = _bits(bb, selection, b)
        high, low = bits[:d], bits[d:]
        enc = OneHotEncode(kk)
        if kk:
            low_soq, oh = bb.add(enc, low=_join(bb, low, _udt(kk)))
        else:
            oh = bb.add(enc)
        fix = PhaseFixup(sp, self.key)
        if d:
            high_soq, oh = bb.add(fix, high=_join(bb, high, _udt(d)), oh=oh)
            high = _bits(bb, high_soq, d)
        else:
            oh = bb.add(fix, oh=oh)
        if kk:
            low_soq = bb.add(Adjoint(enc), low=low_soq, oh=oh)
            low = _bits(bb, low_soq, kk)
        else:
            bb.add(Adjoint(enc), oh=oh)
        return {'selection': _join(bb, high + low, QUInt(b))}

    def adjoint(self):
        return SelectSwapLookup(self.spec)

    def __str__(self):
        return 'Unlookup'

"""Compile each distinct bloq once and replay it.

Walking Qualtran 0.4 decompositions costs tens of microseconds per gate, most of it in Qualtran's
bookkeeping of soquets and connections, while the circuit repeats itself at every level: the same
adder, comparator or swap is called many times.  `template(bloq, ctx)` walks the decomposition of
each distinct bloq once, on symbolic qubit slots, and records it as a list of entries

    (GATE, name, (s1, ...), arg)        a leaf gate of stream.py on slots s1, ...
    (ALLOC, 'ALLOC', (s,), None)        a qubit enters in |0>; it gets a fresh slot
    (FREE, 'FREE', (s,), None)          it leaves in |0>
    (CALL, 'CALL', (s_in, ...), (sub_template, (s_out, ...)))
    (PLAIN, 'PLAIN', (s_in, ...), (bloq, (s_out, ...)))   walked directly at each replay: a
                                        measuring bloq, or one with `stream_ops` (a table lookup,
                                        whose gates depend on the data and would make a large,
                                        single-use template)

Slots 0 .. n_in-1 are the qubits of the left registers (signature order, each register as Qualtran
splits it); every allocation and every output of a call gets a new slot.  The template of a
sub-bloq with at most INLINE_MAX entries is copied into its caller; a larger one is called.

`replay` runs a template on real qubit ids, allocating and releasing qubits in the recorded order.
It therefore yields exactly what walking every instance yields: the same gates on the same ids,
and the same peak (tests/test_compiled.py).  It runs in one generator frame with an explicit stack.

A compiler that walks the bloqs can use the same scheme: every bloq here is hashable and compares by
value, so a cache keyed on the bloq finds each repeated sub-bloq.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from qualtran import Bloq

from .stream import (LEAF_TYPES, StreamContext, flatten_registers, unflatten_registers, walk)

GATE1, GATE2, GATE3, ALLOC, FREE, CALL, PLAIN = range(7)
INLINE_MAX = 64
CACHE_LIMIT = 200_000

_TEMPLATES: Dict[Any, 'Template'] = {}


class Template:
    __slots__ = ('label', 'n_in', 'n_slots', 'ops', 'out_slots', 'n_gates')

    def __init__(self, label, n_in, n_slots, ops, out_slots):
        self.label = label
        self.n_in = n_in
        self.n_slots = n_slots
        self.ops = ops
        self.out_slots = out_slots
        self.n_gates = sum(1 if k <= GATE3 else (op[3][0].n_gates if k == CALL else 0)
                           for op in ops for k in (op[0],))

    def __repr__(self):
        return f'Template({self.label}: {len(self.ops)} entries, {self.n_gates} gates expanded)'


class _Slots:
    """Symbolic qubit slots: each allocation is a new slot and nothing is reused."""

    def __init__(self, start: int):
        self.next = start
        self.live = self.peak = 0

    def allocate(self, n: int) -> List[int]:
        out = list(range(self.next, self.next + int(n)))
        self.next += int(n)
        return out

    def free(self, ids) -> None:
        pass


def _entry(name, qs, arg):
    if name == 'ALLOC':
        return (ALLOC, name, qs, arg)
    if name == 'FREE':
        return (FREE, name, qs, arg)
    return ({1: GATE1, 2: GATE2}.get(len(qs), GATE3), name, tuple(qs), arg)


def clear_templates():
    _TEMPLATES.clear()


def template(bloq: Bloq, ctx: StreamContext) -> Template:
    """The template of `bloq` (recorded on first use; cached for hashable bloqs)."""
    try:
        key = (ctx.greedy, bloq)
        hit = _TEMPLATES.get(key)
    except TypeError:
        key, hit = None, None
    if hit is not None:
        return hit
    t = _record(bloq, ctx)
    if key is not None and len(_TEMPLATES) < CACHE_LIMIT:
        _TEMPLATES[key] = t
    return t


def _record(bloq: Bloq, ctx: StreamContext) -> Template:
    lefts, rights = list(bloq.signature.lefts()), list(bloq.signature.rights())
    n_in = sum(r.total_bits() for r in lefts)
    slots = _Slots(n_in)
    sym = StreamContext(alloc=slots, cache=ctx.cache, cache_limit=ctx.cache_limit,
                        greedy=ctx.greedy, compiled=False)
    sym.sub_hook = lambda sub, inq: _visit(sub, inq, slots, ctx)
    gen = walk(bloq, unflatten_registers(list(range(n_in)), lefts), sym)
    ops = []
    try:
        while True:
            name, qs, arg = next(gen)
            ops.append(name if isinstance(name, tuple) else _entry(name, qs, arg))
    except StopIteration as stop:
        out = stop.value
    return Template(str(bloq), n_in, slots.next, tuple(ops), tuple(flatten_registers(out, rights)))


def _visit(sub: Bloq, inq, slots: _Slots, ctx: StreamContext):
    """While recording: a sub-bloq becomes a call, a copy of its template, or a plain walk.
    Entries are passed up wrapped as (entry, None, None) so that walk() forwards them unchanged."""
    lefts, rights = list(sub.signature.lefts()), list(sub.signature.rights())
    ins = tuple(flatten_registers(inq, lefts))
    n_out = sum(r.total_bits() for r in rights)
    if getattr(sub, 'measured', False) or hasattr(sub, 'stream_ops'):
        # measuring bloqs, and data-dependent trees (lookups), are walked afresh at each replay
        outs = tuple(slots.allocate(n_out))
        yield ((PLAIN, 'PLAIN', ins, (sub, outs)), None, None)
        return unflatten_registers(list(outs), rights)
    st = template(sub, ctx)
    if len(st.ops) > INLINE_MAX:
        outs = tuple(slots.allocate(n_out))
        yield ((CALL, 'CALL', ins, (st, outs)), None, None)
        return unflatten_registers(list(outs), rights)
    m = dict(enumerate(ins))

    def get(s):
        r = m.get(s)
        if r is None:
            r = m[s] = slots.allocate(1)[0]
        return r

    for kind, name, qs, arg in st.ops:
        if kind in (CALL, PLAIN):
            arg = (arg[0], tuple(get(s) for s in arg[1]))
        yield ((kind, name, tuple(get(s) for s in qs), arg), None, None)
    return unflatten_registers([get(s) for s in st.out_slots], rights)


def replay(tpl: Template, in_ids: List[int], ctx: StreamContext):
    """Generator of the leaf ops of a template on real qubit ids; returns the output ids."""
    alloc = ctx.alloc
    stack = []
    m = list(in_ids) + [0] * (tpl.n_slots - tpl.n_in)
    ops, pc, n = tpl.ops, 0, len(tpl.ops)
    while True:
        if pc == n:
            res = [m[s] for s in tpl.out_slots]
            if not stack:
                return res
            tpl, m, ops, pc, n, outs = stack.pop()
            for s, q in zip(outs, res):
                m[s] = q
            continue
        kind, name, qs, arg = ops[pc]
        pc += 1
        if kind == GATE2:
            yield (name, (m[qs[0]], m[qs[1]]), arg)
        elif kind == GATE1:
            yield (name, (m[qs[0]],), arg)
        elif kind == GATE3:
            yield (name, tuple(m[s] for s in qs), arg)
        elif kind == ALLOC:
            (q,) = alloc.allocate(1)
            m[qs[0]] = q
            yield ('ALLOC', (q,), None)
        elif kind == FREE:
            q = m[qs[0]]
            yield ('FREE', (q,), None)
            alloc.free([q])
        elif kind == CALL:
            sub, outs = arg
            stack.append((tpl, m, ops, pc, n, outs))
            sub_m = [m[s] for s in qs]
            m = sub_m + [0] * (sub.n_slots - sub.n_in)
            tpl, ops, pc, n = sub, sub.ops, 0, len(sub.ops)
        else:                                          # PLAIN: a measuring bloq
            sub, outs = arg
            lefts, rights = list(sub.signature.lefts()), list(sub.signature.rights())
            outq = yield from walk(sub, unflatten_registers([m[s] for s in qs], lefts), ctx)
            for s, q in zip(outs, flatten_registers(outq, rights)):
                m[s] = q

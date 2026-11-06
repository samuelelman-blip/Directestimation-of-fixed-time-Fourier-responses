"""Replaying templates (compiled.py) gives exactly the stream of walking every instance: the same
ops on the same qubit ids, the same outputs and the same peak, in both decomposition orders."""
import numpy as np
import pytest
from qualtran import BloqBuilder

from ftfr_q04.arith import BitonicSort, CAdd, HalfLessThan, Square
from ftfr_q04.catalog import CATALOG
from ftfr_q04.compiled import INLINE_MAX, template
from ftfr_q04.gates import RouteToZero
from ftfr_q04.phasegrad import PhaseGradAdd
from ftfr_q04.stream import BloqStream, Condition, StreamContext


def half_less_than_round_trip(n):
    bb = BloqBuilder()
    a = bb.add_register('a', n)
    b = bb.add_register('b', n)
    a, b, c, t = bb.add(HalfLessThan(n), a=a, b=b)
    a, b = bb.add(HalfLessThan(n, uncompute=True), a=a, b=b, c=c, target=t)
    return bb.finalize(a=a, b=b)


LARGER = {
    'BitonicSort(8 keys, 6 bits)': lambda: BitonicSort(8, 6),
    'Square(16)': lambda: Square(16),
    'RouteToZero(3 bits, width 4, 8 slots)': lambda: RouteToZero(3, 4, 8),
    'PhaseGradAdd(20 into 30, controlled)': lambda: PhaseGradAdd(20, 30, right_shift=2,
                                                                 controlled=True),
    'CAdd(16)': lambda: CAdd(16),
    'HalfLessThan(16) round trip': lambda: half_less_than_round_trip(16),
}
ALL = {**CATALOG, **LARGER}


def both(bloq, greedy):
    out = []
    for compiled in (False, True):
        ctx = StreamContext(compiled=compiled, greedy=greedy)
        bs = BloqStream(bloq, ctx)
        ops = [(n, q, ('if', a.keys()) if isinstance(a, Condition) else a) for n, q, a in bs]
        out.append((ops, {k: v.tolist() for k, v in bs.outputs.items()}, ctx.alloc.peak))
    return out


@pytest.mark.parametrize('greedy', [True, False], ids=['greedy order', '0.4 order'])
@pytest.mark.parametrize('name', list(ALL), ids=str)
def test_replay_equals_walk(name, greedy):
    (ops0, out0, peak0), (ops1, out1, peak1) = both(ALL[name](), greedy)
    assert len(ops1) == len(ops0)
    assert ops1 == ops0
    assert out1 == out0 and peak1 == peak0


@pytest.mark.parametrize('name', list(ALL), ids=str)
def test_every_bloq_reached_is_hashable(name):
    """So a compiler can cache on the bloq itself: every bloq in every decomposition hashes, and a
    second copy built the same way is equal to the first."""
    from ftfr_q04.stream import iter_decompositions
    for _, b, cb in iter_decompositions(ALL[name]()):
        hash(b)
        for bi in cb.bloq_instances:
            hash(bi.bloq)
    assert ALL[name]() == ALL[name]() and hash(ALL[name]()) == hash(ALL[name]())


def test_templates_are_shared_and_calls_are_used():
    """A repeated sub-bloq is recorded once; a large one is called rather than copied."""
    ctx = StreamContext()
    t = template(BitonicSort(8, 6), ctx)
    calls = [op for op in t.ops if op[0] == 5]
    assert calls, 'expected calls to sub-templates'
    subs = {id(op[3][0]) for op in calls}
    assert len(subs) < len(calls)                      # the same template called repeatedly
    assert all(len(op[3][0].ops) > INLINE_MAX for op in calls)
    n_gates = sum(1 for op in BloqStream(BitonicSort(8, 6)) if op[0] not in ('ALLOC', 'FREE'))
    assert t.n_gates == n_gates

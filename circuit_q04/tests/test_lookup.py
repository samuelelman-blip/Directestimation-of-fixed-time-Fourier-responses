"""The select-swap lookup and its measurement-based uncompute, against their specifications; the
fast generators against the bloq trees; the cirq decompositions of the measuring bloqs against
the gates the walker takes from them; the AND counts at n = 2^20."""
import cirq
import numpy as np
import pytest

from ftfr_q04.cirq_stream import ParityCondition, to_cirq
from ftfr_q04.lookup import (LookupSpec, LookupUncompute, MeasureX, OneHotEncode, PhaseLeaf,
                             SelectSwapLookup, UnaryLoad, default_lookup_spec)
from ftfr_q04.stream import BloqStream, Condition, SparseSim, StreamContext, count, paper_counts
from ftfr_q04.testing import assert_close, stream_basis_map


def rand_spec(b, widths, lam, kun, seed=1):
    rng = np.random.default_rng(seed)
    cols = tuple(tuple(int(x) for x in rng.integers(0, 2 ** w, 2 ** b)) for w in widths)
    return LookupSpec(b, tuple(widths), cols, lam, kun)


SPECS = [(2, (2, 1), 1, 1), (3, (2,), 1, 2), (3, (1, 2), 2, 1), (2, (2,), 0, 2), (3, (2, 2), 3, 0),
         (4, (2, 1), 2, 2), (3, (2,), 0, 3), (3, (2,), 3, 0)]
IDS = ['b{}-w{}-lam{}-unc{}'.format(b, ''.join(map(str, w)), l, k) for b, w, l, k in SPECS]


def normalised(ops):
    """Conditions compared by the classical bits they read (they are fresh objects each walk)."""
    return [(n, q, ('if', a.keys()) if isinstance(a, Condition) else a) for n, q, a in ops]


@pytest.mark.parametrize('args', SPECS, ids=IDS)
def test_lookup_outputs(args):
    """Slot 0 holds entry j; slot l the l-th word of j's block after the routing; ANDs
    2^(b - lam) - 2 + (2^lam - 1) W."""
    b, widths, lam, kun = args
    spec = rand_spec(b, widths, lam, kun, seed=b + lam)
    bloq = SelectSwapLookup(spec)
    W = spec.W
    for j in range(2 ** b):
        out, amp = stream_basis_map(bloq, j)
        assert abs(amp - 1) < 1e-12
        bits = format(out, f'0{b + (2 ** lam) * W}b')
        assert int(bits[:b], 2) == j
        rest, k = bits[b:], 0
        slots = spec.slot_values(j)
        for c, wc in enumerate(widths):
            assert int(rest[k:k + wc], 2) == spec.columns[c][j]
            k += wc
        for c, wc in enumerate(widths):
            shift = sum(widths[c + 1:])
            for l in range(1, 2 ** lam):
                assert int(rest[k:k + wc], 2) == (slots[l] >> shift) & (2 ** wc - 1)
                k += wc
    assert count(BloqStream(bloq))['AND'] == max(2 ** (b - lam) - 2, 0) + (2 ** lam - 1) * W


@pytest.mark.parametrize('args', SPECS, ids=IDS)
def test_fast_generators_equal_trees(args):
    """The walker's generators (stream_ops) give the gates of the bloq trees, compiled or not."""
    spec = rand_spec(*args, seed=args[0] + args[2])
    for bloq in (SelectSwapLookup(spec), LookupUncompute(spec), UnaryLoad(spec)):
        runs = []
        for fast in (True, False):
            for compiled in (True, False):
                ctx = StreamContext(fast=fast, compiled=compiled)
                runs.append((normalised(BloqStream(bloq, ctx)), ctx.alloc.peak))
        assert all(r == runs[0] for r in runs[1:]), str(bloq)


@pytest.mark.parametrize('fast', [True, False], ids=['fast', 'tree'])
@pytest.mark.parametrize('args', SPECS, ids=IDS)
def test_lookup_then_uncompute_is_identity(args, fast):
    """A superposition over the index, looked up and then uncomputed by measurement, comes back
    exactly (every measured qubit reset, no phase left), for several measurement records; ANDs of
    the uncompute 2^kk - 2 + 2^(b - kk) - 2."""
    b, widths, lam, kun = args
    spec = rand_spec(b, widths, lam, kun, seed=b + lam)
    rng = np.random.default_rng(7)
    amps = rng.normal(size=2 ** b) + 1j * rng.normal(size=2 ** b)
    amps /= np.linalg.norm(amps)
    for seed in range(4):
        ctx = StreamContext(fast=fast)
        sel = ctx.alloc.allocate(b)
        sim = SparseSim(seed=seed)
        for q in sel:
            sim.apply(('ALLOC', (q,), None))
        sim.idx = np.zeros((1, 2 ** b), dtype=np.uint64)
        for j in range(2 ** b):
            v = 0
            for m, q in enumerate(sel):                     # selection is most significant first
                if (j >> (b - 1 - m)) & 1:
                    v |= 1 << sim.slot[q]
            sim.idx[0, j] = v
        sim.amp = amps.copy()
        f = BloqStream(SelectSwapLookup(spec), ctx, inputs={'selection': sel})
        sim.run(iter(f))
        u = BloqStream(LookupUncompute(spec), ctx, inputs=f.outputs)
        sim.run(iter(u))
        assert_close(sim.vector(list(sel)), amps, 1e-10, f'seed {seed}')
    kk = min(kun, b)
    c = count(BloqStream(LookupUncompute(spec)))
    assert c['AND'] == max(2 ** kk - 2, 0) + max(2 ** (b - kk) - 2, 0)
    assert c['MX'] == (2 ** lam) * spec.W and c['CZ_IF'] + c['Z_IF'] == 2 ** b


@pytest.mark.parametrize('kk', [0, 1, 2, 3])
def test_one_hot(kk):
    for low in range(2 ** kk):
        out, amp = stream_basis_map(OneHotEncode(kk), low)
        oh = out & (2 ** (2 ** kk) - 1)
        assert out >> (2 ** kk) == low and abs(amp - 1) < 1e-12
        assert oh == 1 << (2 ** kk - 1 - low)              # qubit `low`, in Qualtran's order


def test_measure_x_cirq_matches_walker():
    """MeasureX's cirq decomposition is the walker's gates: H, measure, reset per qubit."""
    W, key, slot = 3, 'k', 2
    bloq = MeasureX(W, key, slot)
    qs = cirq.LineQubit.range(W)
    ref = list(cirq.flatten_op_tree(bloq.decompose_from_registers(
        context=cirq.DecompositionContext(cirq.SimpleQubitManager()), q=np.array(qs))))
    ctx = StreamContext()
    ops = list(bloq.measured_ops({'q': np.arange(W)}, ctx))
    table = ctx.named_cbits[(key, slot)]
    names = {f'm{k}': f'{key}_{slot}_{beta}' for beta, k in enumerate(table)}
    got = [cirq.with_measurement_key_mapping(op, names) if cirq.is_measurement(op) else op
           for op in to_cirq(ops)]
    assert got == ref


@pytest.mark.parametrize('controlled,cv', [(True, 1), (True, 0), (False, 1)])
def test_phase_leaf_cirq_matches_walker(controlled, cv):
    """PhaseLeaf's cirq decomposition is the walker's gates, with the same parity conditions."""
    spec = rand_spec(3, (2, 1), 1, 2, seed=3)
    key, h = 'k', 1
    bloq = PhaseLeaf(spec, key, h, cv, controlled)
    n_oh = 2 ** spec.kk
    ctx = StreamContext()
    names = {}
    for l in range(spec.nslot):                             # pretend every slot was measured
        row = [ctx.new_cbit() for _ in range(spec.W)]
        ctx.named_cbits[(key, l)] = np.array(row)
        names.update({f'm{k}': f'{key}_{l}_{beta}' for beta, k in enumerate(row)})
    ids = {'oh': np.arange(1, 1 + n_oh)}
    if controlled:
        ids['ctrl'] = np.array([0])
    got = list(to_cirq(bloq.measured_ops(ids, ctx)))
    regs = {'oh': np.array(cirq.LineQubit.range(1, 1 + n_oh))}
    if controlled:
        regs['ctrl'] = np.array([cirq.LineQubit(0)])
    ref = list(cirq.flatten_op_tree(bloq.decompose_from_registers(
        context=cirq.DecompositionContext(cirq.SimpleQubitManager()), **regs)))
    assert len(got) == len(ref)
    for a, b_ in zip(got, ref):
        if isinstance(a, cirq.ClassicallyControlledOperation):
            assert a.without_classical_controls() == b_.without_classical_controls()
            ka = sorted(names[str(k)] for k in next(iter(a.classical_controls)).keys)
            kb = sorted(str(k) for k in next(iter(b_.classical_controls)).keys)
            assert ka == kb
        else:
            assert a == b_


def test_lazy_conditions():
    """A condition's keys are not computed until read."""
    calls = []
    cond = ParityCondition(key_fn=lambda: calls.append(1) or ['a', 'b'], label='f(3)')
    assert str(cond) == 'f(3)' and not calls
    assert len(cond.keys) == 2 and calls == [1]


def test_counts_at_full_size():
    """At n = 2^20 with the PD-T widths (29, 28): block sizes 2^7 and 2^10 as the reference
    implementation chooses them, and 4 times the AND counts of the lookup and the uncompute equal
    the T counts it charges for them (61716 T and 8176 T, Qualtran 0.7's QROAMClean and its
    adjoint)."""
    b, widths = 20, (29, 28)
    spec = default_lookup_spec(b, widths, ((0,) * 2 ** b, (0,) * 2 ** b))
    assert (spec.log_block, spec.log_block_unc) == (7, 10)
    assert 4 * paper_counts(BloqStream(SelectSwapLookup(spec)))['AND'] == 61716
    assert 4 * paper_counts(BloqStream(LookupUncompute(spec)))['AND'] == 8176

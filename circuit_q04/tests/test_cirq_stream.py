"""The cirq form of the gate stream: simulated in cirq, it gives the same states as the stream
simulator (which tests/test_stream.py checks against Qualtran's tensor contraction), with And and
And-dagger kept as Qualtran's And gate or written out with measurements; parity conditions on
measurement outcomes; and the stream is produced lazily."""
import itertools

import cirq
import numpy as np
import pytest
from qualtran import Adjoint, BloqBuilder, QUInt
from qualtran.bloqs.arithmetic import Add, LessThanEqual
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition

from ftfr_q04.arith import BitonicSort, CAdd, Comparator, HalfLessThan, Square
from ftfr_q04.cirq_stream import CirqStream, to_cirq
from ftfr_q04.catalog import ETA, MIDDLE
from ftfr_q04.gates import (BasisPhase, CHadamardExact, FanHadamard, RegisterCSwap, RouteToZero,
                            ZeroReflection)
from ftfr_q04.history import HistoryPrep, KeySort, StepFlags, TimeRegister, make_selector
from ftfr_q04.phasegrad import PhaseGradAdd, WindowAdd
from ftfr_q04.stream import Parity, StreamContext
from ftfr_q04.testing import assert_close, stream_matrix


def half_less_than_round_trip(n):
    bb = BloqBuilder()
    a = bb.add_register('a', n)
    b = bb.add_register('b', n)
    a, b, c, t = bb.add(HalfLessThan(n), a=a, b=b)
    a, b = bb.add(HalfLessThan(n, uncompute=True), a=a, b=b, c=c, target=t)
    return bb.finalize(a=a, b=b)


CASES = {
    'CH': CHadamardExact(),
    'FanHadamard(2)': FanHadamard(2),
    'BasisPhase(3)': BasisPhase(3, 0.37, (1, 0, 1)),
    'ZeroReflection(3)': ZeroReflection(3),
    'RegisterCSwap(2)': RegisterCSwap(2),
    'RouteToZero(1,1,2)': RouteToZero(1, 1, 2),
    'WindowAdd(3,cin)': WindowAdd(3, (0, 1), carry_in=True),
    'PhaseGradAdd(2,3,ctrl)': PhaseGradAdd(2, 3, controlled=True),
    'Adjoint(PhaseGradAdd(2,3))': Adjoint(PhaseGradAdd(2, 3)),
    'Square(2)': Square(2),
    'CAdd(2)': CAdd(2),
    'HalfLessThan(2) round trip': half_less_than_round_trip(2),
    'Comparator(2)': Comparator(2),
    'qualtran Add(2)': Add(QUInt(2)),
    'qualtran LessThanEqual(2,2)': LessThanEqual(2, 2),
    'qualtran PrepareUniformSuperposition(3)': PrepareUniformSuperposition(3),
    'qualtran PhaseGradientState(3)': PhaseGradientState(3),
    'AliasSampler(selector M=2)': make_selector(2, ETA, 0.05),
    'StepFlags(2, 2)': StepFlags(2, 2),
    'TimeRegister(2)': TimeRegister(2),
    'KeySort(M=2, b_t=1)': KeySort(2, 1),
    'HistoryPrep(M=1, b_t=1)': HistoryPrep(1, 1, ETA, 0.05),
    'Lookup then LookupUncompute': MIDDLE['Lookup then LookupUncompute'](),
    'ControlledRowInverse(2, dense)': MIDDLE['ControlledRowInverse(2, dense)'](),
    'InequalityTest(3)': MIDDLE['InequalityTest(3)'](),
}


def cirq_column(bloq, k, expand_and, seed):
    """Simulate the cirq stream of `bloq` on basis input k (left registers, big-endian) and return
    the state of the right registers; every other qubit must end in |0>."""
    ctx = StreamContext()
    lefts, rights = list(bloq.signature.lefts()), list(bloq.signature.rights())
    ids = {r.name: np.array(ctx.alloc.allocate(r.total_bits())).reshape(r.shape + (r.bitsize,))
           for r in lefts}
    cs = CirqStream(bloq, ctx, inputs=ids, expand_and=expand_and)
    in_q = [cirq.LineQubit(int(i)) for r in lefts for i in np.ravel(ids[r.name])]
    prep = [cirq.I(q) for q in in_q]
    prep += [cirq.X(q) for j, q in enumerate(in_q) if (k >> (len(in_q) - 1 - j)) & 1]
    ops = list(cs)
    out_q = [q for r in rights for q in cs.output_qubits(r.name)]
    circuit = cirq.Circuit(prep + ops + [cirq.I(q) for q in out_q])
    order = sorted(circuit.all_qubits())
    psi = cirq.Simulator(seed=seed).simulate(circuit, qubit_order=order).final_state_vector
    psi = psi.reshape([2] * len(order))
    pos = {q: i for i, q in enumerate(order)}
    others = [pos[q] for q in order if q not in out_q]
    idx = [slice(None)] * len(order)
    for i in others:
        idx[i] = 0
    sub = psi[tuple(idx)]                            # axes: the output qubits, in `order`
    assert abs(np.linalg.norm(sub) - 1) < 1e-6, 'a qubit outside the outputs is not |0>'
    kept = [q for q in order if q in out_q]
    sub = np.transpose(sub, [kept.index(q) for q in out_q])
    return sub.reshape(-1)


@pytest.mark.parametrize('expand_and', [False, True], ids=['And kept', 'And expanded'])
@pytest.mark.parametrize('name', list(CASES), ids=str)
def test_cirq_stream_matches_stream_simulator(name, expand_and):
    bloq = CASES[name]
    n_in = sum(r.total_bits() for r in bloq.signature.lefts())
    cols = list(range(2 ** n_in)) if n_in <= 5 else [0, 1, 2 ** n_in - 1, 0b101101 % 2 ** n_in]
    for ci, k in enumerate(cols):
        want = stream_matrix(bloq, columns=[k], measure_ands=True, seed=ci)[:, 0]
        got = cirq_column(bloq, k, expand_and, seed=ci)
        assert_close(got, want, 1e-6, f'{name} input {k}')


def test_parity_conditions():
    """X on q2 if m0 ^ m1, on q3 if not (m0 ^ m1), on q4 if m1."""
    ops = [('H', (0,), None), ('MZ', (0,), 0), ('H', (1,), None), ('MZ', (1,), 1),
           ('X', (2,), Parity([0, 1])), ('X', (3,), Parity([0, 1], invert=True)),
           ('X', (4,), Parity([1]))]
    q = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(list(to_cirq(ops)) + [cirq.measure(q[i], key=f'out{i}')
                                                 for i in (2, 3, 4)])
    res = cirq.Simulator(seed=5).run(circuit, repetitions=64)
    m0, m1 = res.measurements['m0'][:, 0], res.measurements['m1'][:, 0]
    assert len(set(zip(m0, m1))) == 4
    assert np.all(res.measurements['out2'][:, 0] == m0 ^ m1)
    assert np.all(res.measurements['out3'][:, 0] == 1 - (m0 ^ m1))
    assert np.all(res.measurements['out4'][:, 0] == m1)


def test_measured_qubit_is_reset():
    ops = [('H', (0,), None), ('MZ', (0,), 0), ('CNOT', (0, 1), None)]
    circuit = cirq.Circuit(list(to_cirq(ops)) + [cirq.measure(cirq.LineQubit(1), key='q1')])
    res = cirq.Simulator(seed=2).run(circuit, repetitions=32)
    assert not res.measurements['q1'].any()


def test_stream_is_lazy():
    """The first gates come out before the walk has gone through the bloq."""
    cs = CirqStream(BitonicSort(16, 8))
    first = list(itertools.islice(iter(cs), 10))
    assert len(first) == 10 and all(isinstance(op, cirq.Operation) for op in first)
    assert cs.stream.outputs is None


@pytest.mark.parametrize('expand_and', [False, True], ids=['And kept', 'And expanded'])
def test_operations_equal_cirq_own(expand_and):
    """Operations built without cirq's argument check equal those cirq builds with it."""
    for op in CirqStream(Comparator(3), expand_and=expand_and):
        if isinstance(op, cirq.GateOperation):
            ref = op.gate.on(*op.qubits)
            assert op == ref and hash(op) == hash(ref) and repr(op) == repr(ref)


def test_expanded_and_counts():
    """Expanded, an And is 4 T and an And-dagger one measurement and one conditioned CZ."""
    ops = list(CirqStream(CAdd(3), expand_and=True))
    t = sum(1 for op in ops if op.gate in (cirq.T, cirq.T ** -1))
    meas = sum(1 for op in ops if cirq.is_measurement(op))
    cz_if = sum(1 for op in ops if isinstance(op, cirq.ClassicallyControlledOperation))
    assert (t, meas, cz_if) == (4 * 5, 5, 5)

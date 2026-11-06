"""No decomposition is large: at register sizes at or above those of the n = 2^20 cells, every
decompose_bloq() reachable from each primitive, and from the Grover iterate of each cell, has at
most MAX_LEVEL bloq instances, so a compiler walking the bloqs depth first holds a bounded amount
of circuit at a time."""
import pytest
from qualtran import Adjoint, QUInt
from qualtran.bloqs.arithmetic import Add, AddK, LessThanConstant, LessThanEqual
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition

from ftfr_q04.arith import BitonicSort, CAdd, Comparator, GreaterThan, HalfLessThan, Square
from ftfr_q04.catalog import ETA
from ftfr_q04.gates import (BasisPhase, CHadamardExact, FanHadamard, RegisterCSwap, RouteToZero,
                            ZeroReflection)
from ftfr_q04.history import HistoryPrep
from ftfr_q04.phasegrad import PhaseGradAdd, WindowAdd
from ftfr_q04.stream import largest_level

MAX_LEVEL = 10_000

# the cells use phase-gradient registers of up to 44 qubits, times of up to 23 bits, 20-bit
# coordinates and selectors of up to 63 qubits
FULL_SIZE = {
    'CHadamardExact': lambda: CHadamardExact(),
    'FanHadamard(64)': lambda: FanHadamard(64),
    'BasisPhase(64)': lambda: BasisPhase(64, 0.3),
    'ZeroReflection(128)': lambda: ZeroReflection(128),
    'RegisterCSwap(64)': lambda: RegisterCSwap(64),
    'RouteToZero(9 bits, width 32, 512 slots)': lambda: RouteToZero(9, 32, 512),
    'WindowAdd(64)': lambda: WindowAdd(64, tuple(range(64)), carry_in=True),
    'PhaseGradAdd(44 into 64, controlled)': lambda: PhaseGradAdd(44, 64, right_shift=4,
                                                                 controlled=True),
    'Square(32)': lambda: Square(32),
    'Adjoint(Square(32))': lambda: Adjoint(Square(32)),
    'CAdd(64)': lambda: CAdd(64),
    'HalfLessThan(64)': lambda: HalfLessThan(64),
    'HalfLessThan(64) uncompute': lambda: HalfLessThan(64, uncompute=True),
    'GreaterThan(64, 64)': lambda: GreaterThan(64, 64),
    'Comparator(64)': lambda: Comparator(64),
    'BitonicSort(16 keys, 32 bits)': lambda: BitonicSort(16, 32),
    'qualtran Add(64)': lambda: Add(QUInt(64)),
    'qualtran AddK(64)': lambda: AddK(64, 12345),
    'qualtran LessThanConstant(64)': lambda: LessThanConstant(64, 12345),
    'qualtran LessThanEqual(64, 64)': lambda: LessThanEqual(64, 64),
    'qualtran PrepareUniformSuperposition(10^6)': lambda: PrepareUniformSuperposition(10 ** 6),
    'qualtran PhaseGradientState(64)': lambda: PhaseGradientState(64),
    # checkpoint 2 (the cells have M <= 4, b_t <= 23 and selectors of mu <= 28 bits)
    'HistoryPrep(M=4, b_t=24, mu=30)': lambda: HistoryPrep(4, 24, ETA, 2e-10),
    'Adjoint(HistoryPrep(M=4, b_t=24, mu=30))': lambda: Adjoint(HistoryPrep(4, 24, ETA, 2e-10)),
    # checkpoint 3 without tables (the lookups are checked kind by kind below)
    'MiddleBlock(PD closed form, M=4, b=20, b_t=22)': lambda: _middle_pd_f(),
}


def _middle_pd_f():
    from ftfr_q04.access import make_access
    from ftfr_q04.middle import MiddleBlock
    return MiddleBlock(4, 20, 22, 43, 'dense', make_access('PD', 'F', 20, 21, 41))


@pytest.mark.parametrize('name', list(FULL_SIZE), ids=str)
def test_largest_decomposition_is_bounded(name):
    size, where = largest_level(FULL_SIZE[name]())
    assert size <= MAX_LEVEL, f'{where}: {size} bloq instances in one decomposition'


def _is_lookup(b):
    from ftfr_q04.lookup import LookupUncompute, SelectSwapLookup
    return isinstance(b, (SelectSwapLookup, LookupUncompute))


def _cells():
    from ftfr_q04.block import load_cells
    return load_cells()


@pytest.mark.parametrize('cell', _cells(), ids=lambda c: c.label)
def test_cell_iterates_are_bounded(cell):
    """The Grover iterate of each of the nine cells at n = 2^20, through the reflection (on up to
    385 qubits), the block, P and C-X down to the leaves; the table lookups are checked kind by kind
    in the next test."""
    from ftfr_q04.block import GroverIterate
    size, where = largest_level(GroverIterate(cell), skip=_is_lookup)
    assert size <= MAX_LEVEL, f'{where}: {size} bloq instances in one decomposition'


@pytest.fixture(scope='module')
def full_spec():
    """The lookup of the PD-T eps = 1e-6 cell: 2^20 entries of 29 + 28 bits, 2^7 slots, a one-hot
    register of 2^10 qubits (the contents do not change the structure)."""
    from ftfr_q04.lookup import LookupSpec
    zeros = (0,) * 2 ** 20
    return LookupSpec(20, (29, 28), (zeros, zeros), 7, 10)


def test_lookup_decompositions_are_bounded(full_spec):
    """Every kind of bloq in the lookup tree at full size (walking all 10^6 of them is not needed:
    the bloqs of one kind differ only in their data).  The measuring bloqs are counted in cirq
    operations of their decomposition."""
    import cirq
    from qualtran._infra.gate_with_registers import get_named_qubits
    from ftfr_q04.lookup import (FanOut, LoadLeaf, LoadNode, LookupUncompute, MeasureX,
                                 OneHotEncode, PhaseFixup, PhaseLeaf, PhaseNode, SelectSwapLookup,
                                 UnaryLoad)
    sp = full_spec
    W = sp.W
    kinds = [SelectSwapLookup(sp), UnaryLoad(sp), LoadNode(sp, 11, 0, 0), LoadNode(sp, 0, 5),
             LoadLeaf(sp, 3, 1), FanOut(W, 2 ** W - 1), RouteToZero(7, W, 128),
             LookupUncompute(sp, 'k'), OneHotEncode(10), Adjoint(OneHotEncode(10)),
             PhaseFixup(sp, 'k'), PhaseNode(sp, 'k', 8, 0)]
    for b in kinds:
        n = len(b.decompose_bloq().bloq_instances)
        assert n <= MAX_LEVEL, f'{b}: {n}'
    ctx = cirq.DecompositionContext(cirq.SimpleQubitManager())
    for b in (PhaseLeaf(sp, 'k', 3, 0), MeasureX(W, 'k', 0)):
        ops = list(cirq.flatten_op_tree(b.decompose_from_registers(
            context=ctx, **get_named_qubits(b.signature))))
        assert len(ops) <= MAX_LEVEL, f'{b}: {len(ops)}'

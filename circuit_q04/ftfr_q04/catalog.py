"""Small instances of every bloq, for compiler probes (tools/probe.py) and tests."""
import math

from qualtran import Adjoint
from qualtran.bloqs.arithmetic import Add, AddK, LessThanConstant, LessThanEqual
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition

from .arith import BitonicSort, CAdd, Comparator, GreaterThan, HalfLessThan, Square
from .gates import (BasisPhase, CHadamardExact, FanHadamard, RegisterCSwap, RouteToZero,
                    ZeroReflection)
from .history import AliasSampler, HistoryPrep, KeySort, StepFlags, TimeRegister, make_selector
from .phasegrad import PhaseGradAdd, WindowAdd

ETA = 0.125 * (1 - math.exp(-1.0))           # eta of every cell: r (1 - e^{-mu T}), r = 1/8, mu T = 1

# checkpoint 1: primitives (name -> factory)
PRIMITIVES = {
    'CHadamardExact': lambda: CHadamardExact(),
    'FanHadamard(3)': lambda: FanHadamard(3),
    'BasisPhase(4, 0.3, |0110>)': lambda: BasisPhase(4, 0.3, (0, 1, 1, 0)),
    'ZeroReflection(5)': lambda: ZeroReflection(5),
    'RegisterCSwap(3)': lambda: RegisterCSwap(3),
    'RouteToZero(sel 2 bits, width 2, r 3)': lambda: RouteToZero(2, 2, 3),
    'WindowAdd(6, positions 1,3,5, minus_top)': lambda: WindowAdd(6, (1, 3, 5), minus_top=True),
    'PhaseGradAdd(4 into 6, controlled)': lambda: PhaseGradAdd(4, 6, right_shift=1, controlled=True),
    'Adjoint(PhaseGradAdd(5, 5))': lambda: Adjoint(PhaseGradAdd(5, 5)),
    'Square(4)': lambda: Square(4),
    'Adjoint(Square(3))': lambda: Adjoint(Square(3)),
    'CAdd(4)': lambda: CAdd(4),
    'Adjoint(CAdd(4))': lambda: Adjoint(CAdd(4)),
    'HalfLessThan(4)': lambda: HalfLessThan(4),
    'HalfLessThan(4) uncompute': lambda: HalfLessThan(4, uncompute=True),
    'GreaterThan(3, 3)': lambda: GreaterThan(3, 3),
    'Comparator(3)': lambda: Comparator(3),
    'BitonicSort(4 keys, 3 bits)': lambda: BitonicSort(4, 3),
    # Qualtran 0.4's own bloqs, used as they are (same gates as the reference implementation)
    'qualtran Add(QUInt(4))': lambda: Add(__import__('qualtran').QUInt(4)),
    'qualtran AddK(5, 1)': lambda: AddK(5, 1),
    'qualtran AddK(4, 1, controlled)': lambda: AddK(4, 1, cvs=(1,)),
    'qualtran LessThanConstant(3, 2)': lambda: LessThanConstant(3, 2),
    'qualtran LessThanEqual(3, 3)': lambda: LessThanEqual(3, 3),
    'qualtran PrepareUniformSuperposition(3)': lambda: PrepareUniformSuperposition(3),
    'qualtran PhaseGradientState(5)': lambda: PhaseGradientState(5),
}

# checkpoint 2: the history preparation P and its parts
HISTORY = {
    'AliasSampler(selector M=3)': lambda: make_selector(3, ETA, 0.01),
    'AliasSampler(5 outcomes, alternates)': lambda: AliasSampler.from_probabilities(
        [1, 3, 2, 5, 4], precision=0.01),
    'qualtran QROM(selector M=3)': lambda: make_selector(3, ETA, 0.01).qrom,
    'StepFlags(2 bits, M=3)': lambda: StepFlags(2, 3),
    'TimeRegister(4)': lambda: TimeRegister(4),
    'KeySort(M=3, b_t=2)': lambda: KeySort(3, 2),
    'HistoryPrep(M=3, b_t=2)': lambda: HistoryPrep(3, 2, ETA, 0.05),
    'Adjoint(HistoryPrep(M=3, b_t=2))': lambda: Adjoint(HistoryPrep(3, 2, ETA, 0.05)),
}

def _small_spec():
    from .lookup import LookupSpec
    cols = ((3, 1, 0, 2, 1, 3, 2, 0), (1, 0, 1, 1, 0, 0, 1, 0))
    return LookupSpec(3, (2, 1), cols, 1, 2)


def _lookup_round_trip():
    from qualtran import BloqBuilder, QUInt, Register
    from .lookup import LookupUncompute, SelectSwapLookup
    spec = _small_spec()
    bb = BloqBuilder()
    sel = bb.add_register(Register('selection', QUInt(spec.index_bits)))
    out = bb.add_d(SelectSwapLookup(spec), selection=sel)
    sel = bb.add(LookupUncompute(spec), **out)
    return bb.finalize(selection=sel)


def _middle(family, access, b=2, M=2, b_t=1):
    from .access import make_access
    from .middle import MiddleBlock
    acc = make_access(family, access, b, b + 2, 2 * b + 2) if family in ('PD', 'PS') else None
    skeleton = 'dense' if family in ('dense', 'PD') else 'sparse'
    w_phi = max(b + b_t + 1, acc.phase_bits if acc is not None else 0)
    return MiddleBlock(M, b, b_t, w_phi, skeleton, acc)


def _checkpoint3():
    from .access import InequalityTest, RowPhaseF, ValueOracleF, make_access
    from .lookup import FanOut, LookupUncompute, OneHotEncode, SelectSwapLookup, UnaryLoad
    from .middle import Contraction, VertexPhase
    from .rows import AChi, ContractionPhase, ControlledRowInverse, RowPrepDense
    return {
        'FanOut(4, 0b1011)': lambda: FanOut(4, 0b1011),
        'UnaryLoad(8 entries, 2 slots)': lambda: UnaryLoad(_small_spec()),
        'SelectSwapLookup(8 entries, 2 slots)': lambda: SelectSwapLookup(_small_spec()),
        'OneHotEncode(2)': lambda: OneHotEncode(2),
        'LookupUncompute(8 entries)': lambda: LookupUncompute(_small_spec()),
        'Lookup then LookupUncompute': _lookup_round_trip,
        'AChi(2, controlled)': lambda: AChi(2, True),
        'RowPrepDense(2)': lambda: RowPrepDense(2),
        'ControlledRowInverse(2, dense)': lambda: ControlledRowInverse(2, 'dense'),
        'ControlledRowInverse(2, sparse)': lambda: ControlledRowInverse(2, 'sparse'),
        'ContractionPhase(2, dense)': lambda: ContractionPhase(2, 'dense'),
        'ValueOracleF(2)': lambda: ValueOracleF(2),
        'InequalityTest(3)': lambda: InequalityTest(3),
        'RowPhaseF(2)': lambda: RowPhaseF(2),
        'RowAccess(PD, closed form, b=2)': lambda: make_access('PD', 'F', 2, 3, 5),
        'RowAccess(PD, tables, b=2)': lambda: make_access('PD', 'T', 2, 4, 6),
        'RowAccess(PS, tables, b=2)': lambda: make_access('PS', 'T', 2, 4, 0),
        'VertexPhase(2, 2)': lambda: VertexPhase(2, 2),
        'Contraction(r=2, dense, b=2)': lambda: Contraction(2, 2, 'dense'),
        'MiddleBlock(dense, M=2, b=2)': lambda: _middle('dense', 'flat'),
        'MiddleBlock(sparse, M=2, b=2)': lambda: _middle('sparse', 'flat'),
        'MiddleBlock(PD tables, M=2, b=2)': lambda: _middle('PD', 'T'),
        'MiddleBlock(PS closed form, M=1, b=2)': lambda: _middle('PS', 'F', M=1),
    }


# checkpoint 3: lookups, rows, row access, the middle block
MIDDLE = _checkpoint3()


def _checkpoint4():
    from .block import Block, GroverIterate, HadamardTest, ReflectZero, small_cell
    dense = small_cell('dense', 'flat', 2, 1, 2, sel_precision=0.05)
    tables = small_cell('PD', 'T', 2, 2, 1, sel_precision=0.05)
    return {
        'Block(dense, b=2, M=1)': lambda: Block(dense),
        'HadamardTest(dense, b=2, M=1)': lambda: HadamardTest(dense),
        'ReflectZero(dense, b=2, M=1)': lambda: ReflectZero(dense),
        'GroverIterate(dense, b=2, M=1)': lambda: GroverIterate(dense),
        'GroverIterate(PD tables, b=2, M=2)': lambda: GroverIterate(tables),
    }


# checkpoint 4: the block, the Hadamard test, the reflection, the Grover iterate
BLOCK = _checkpoint4()

CATALOG = {**PRIMITIVES, **HISTORY, **MIDDLE, **BLOCK}

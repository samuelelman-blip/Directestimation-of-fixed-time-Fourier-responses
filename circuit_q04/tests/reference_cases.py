"""The Qualtran 0.4 builds of the reference cases in data/reference_counts.json
(tools/reference_counts.py)."""
import numpy as np
from qualtran import Adjoint, BoundedQUInt, QUInt, Register
from qualtran.bloqs.arithmetic import Add, AddK, LessThanConstant, LessThanEqual
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.rotations import PhaseGradientState
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition

from ftfr_q04.arith import BitonicSort, CAdd, Comparator, GreaterThan, HalfLessThan, Square
from ftfr_q04.gates import BasisPhase, CHadamardExact, FanHadamard, RouteToZero, ZeroReflection
from ftfr_q04.history import make_selector
from ftfr_q04.phasegrad import PhaseGradAdd, WindowAdd

ETA = 0.0790150699

CASES = {
    'CHadamardExact': lambda: CHadamardExact(),
    'FanHadamard(3)': lambda: FanHadamard(3),
    'BasisPhase(4,0.3,0110)': lambda: BasisPhase(4, 0.3, (0, 1, 1, 0)),
    'BasisPhase(1,0.3,0)': lambda: BasisPhase(1, 0.3, (0,)),
    'ZeroReflection(2)': lambda: ZeroReflection(2),
    'ZeroReflection(5)': lambda: ZeroReflection(5),
    'RouteToZero(2,2,3)': lambda: RouteToZero(2, 2, 3),
    'RouteToZero(2,3,4)^-1': lambda: RouteToZero(2, 3, 4, uncompute=True),
    'WindowAdd(5,012,cin)': lambda: WindowAdd(5, (0, 1, 2), carry_in=True),
    'WindowAdd(6,135,minus)': lambda: WindowAdd(6, (1, 3, 5), minus_top=True),
    'PhaseGradAdd(4,6)': lambda: PhaseGradAdd(4, 6),
    'PhaseGradAdd(4,6,s1,ctrl)': lambda: PhaseGradAdd(4, 6, right_shift=1, controlled=True),
    'PhaseGradAdd(6,7,s1,ctrl,kz1)': lambda: PhaseGradAdd(6, 7, right_shift=1, controlled=True,
                                                          known_zero=(1,)),
    'Adjoint(PhaseGradAdd(5,5))': lambda: Adjoint(PhaseGradAdd(5, 5)),
    'Square(4)': lambda: Square(4),
    'Adjoint(Square(3))': lambda: Adjoint(Square(3)),
    'Add(3)': lambda: Add(QUInt(3)),
    'Add(5)': lambda: Add(QUInt(5)),
    'CAdd(4)': lambda: CAdd(4),
    'Adjoint(CAdd(4))': lambda: Adjoint(CAdd(4)),
    'AddK(5,1)': lambda: AddK(5, 1),
    'Adjoint(AddK(5,1))': lambda: Adjoint(AddK(5, 1)),
    'C-AddK(4,1)': lambda: AddK(4, 1, cvs=(1,)),
    'Adjoint(C-AddK(4,1))': lambda: Adjoint(AddK(4, 1, cvs=(1,))),
    'LinearDepthHalfLessThan(4)': lambda: HalfLessThan(4),
    'Adjoint(LinearDepthHalfLessThan(4))': lambda: HalfLessThan(4, uncompute=True),
    'LessThanConstant(3,2)': lambda: LessThanConstant(3, 2),
    'LessThanConstant(3,1)': lambda: LessThanConstant(3, 1),
    'LessThanConstant(3,3)': lambda: LessThanConstant(3, 3),
    'LessThanEqual(3,3)': lambda: LessThanEqual(3, 3),
    'LessThanEqual(5,5)': lambda: LessThanEqual(5, 5),
    'GreaterThan(3,3)': lambda: GreaterThan(3, 3),
    'Comparator(3)': lambda: Comparator(3),
    'BitonicSort(2,3)': lambda: BitonicSort(2, 3),
    'BitonicSort(4,3)': lambda: BitonicSort(4, 3),
    'PrepareUniformSuperposition(3)': lambda: PrepareUniformSuperposition(3),
    'PrepareUniformSuperposition(5)': lambda: PrepareUniformSuperposition(5),
    'PrepareUniformSuperposition(2)': lambda: PrepareUniformSuperposition(2),
    'PhaseGradientState(5)': lambda: PhaseGradientState(5),
    # checkpoint 2: the selector and its QROM
    'QROM(Selector M=2)': lambda: make_selector(2, ETA, 0.05).qrom,
    'QROM(Selector M=3)': lambda: make_selector(3, ETA, 0.01).qrom,
    'Selector(M=2,eta=0.079,prec=0.05)': lambda: make_selector(2, ETA, 0.05),
    'Selector(M=3,eta=0.079,prec=0.01)': lambda: make_selector(3, ETA, 0.01),
}

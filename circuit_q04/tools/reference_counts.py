"""Gate counts of the reference implementation (../circuit), which this package must reproduce.

The reference implementation was written for Qualtran 0.7, so this script, alone in this package,
runs in that environment (not the Qualtran 0.4 venv):  python3 tools/reference_counts.py
It writes data/reference_counts.json: for each named case, the leaf counts of the reference gate
stream in the paper's conventions (ANDs with control value 0 carry their X gates; rotations within
1e-12 of a Clifford+T gate are that gate; global phases dropped) and the peak number of live
qubits; and data/reference_cells.json (see `cells`).  The tests build the same cases in Qualtran 0.4
(tests/reference_cases.py) and must reproduce these counts.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'circuit'))

from qualtran import Adjoint, QUInt  # noqa: E402
from qualtran.bloqs.arithmetic import (Add, AddK, BitonicSort, LessThanConstant,  # noqa: E402
                                       LinearDepthHalfLessThan)
from qualtran.bloqs.arithmetic.controlled_addition import CAdd  # noqa: E402
from qualtran.bloqs.arithmetic.sorting import Comparator  # noqa: E402
from qualtran.bloqs.arithmetic import GreaterThan, LessThanEqual  # noqa: E402
from qualtran.bloqs.rotations import PhaseGradientState  # noqa: E402
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition  # noqa: E402

from ftfr.arith import Square  # noqa: E402
from ftfr.gates import (BasisPhase, CHadamardExact, FanHadamard, RouteToZero,  # noqa: E402
                        ZeroReflection)
from ftfr.history import make_selector  # noqa: E402
from ftfr.phasegrad import PhaseGradAdd, WindowAdd  # noqa: E402
from ftfr.stream import BloqStream, count  # noqa: E402

CASES = {
    # ---- our primitives ------------------------------------------------------------------------
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
    # ---- Qualtran 0.7 arithmetic used by the circuit -------------------------------------------
    'Add(3)': lambda: Add(QUInt(3)),
    'Add(5)': lambda: Add(QUInt(5)),
    'Adjoint(CAdd(4))': lambda: CAdd(QUInt(4), QUInt(4), cv=1).adjoint(),
    'CAdd(4)': lambda: CAdd(QUInt(4), QUInt(4), cv=1),
    'AddK(5,1)': lambda: AddK(QUInt(5), 1),
    'Adjoint(AddK(5,1))': lambda: Adjoint(AddK(QUInt(5), 1)),
    'C-AddK(4,1)': lambda: AddK(QUInt(4), 1).controlled(),
    'Adjoint(C-AddK(4,1))': lambda: Adjoint(AddK(QUInt(4), 1).controlled()),
    'LinearDepthHalfLessThan(4)': lambda: LinearDepthHalfLessThan(QUInt(4)),
    'Adjoint(LinearDepthHalfLessThan(4))': lambda: Adjoint(LinearDepthHalfLessThan(QUInt(4))),
    'LessThanConstant(3,2)': lambda: LessThanConstant(3, 2),
    'LessThanConstant(3,1)': lambda: LessThanConstant(3, 1),
    'LessThanConstant(3,3)': lambda: LessThanConstant(3, 3),
    'Comparator(3)': lambda: Comparator(3),
    'BitonicSort(2,3)': lambda: BitonicSort(2, 3),
    'BitonicSort(4,3)': lambda: BitonicSort(4, 3),
    'PrepareUniformSuperposition(3)': lambda: PrepareUniformSuperposition(3),
    'PrepareUniformSuperposition(5)': lambda: PrepareUniformSuperposition(5),
    'PrepareUniformSuperposition(2)': lambda: PrepareUniformSuperposition(2),
    'PhaseGradientState(5)': lambda: PhaseGradientState(5),
    'LessThanEqual(3,3)': lambda: LessThanEqual(3, 3),
    'LessThanEqual(5,5)': lambda: LessThanEqual(5, 5),
    'GreaterThan(3,3)': lambda: GreaterThan(3, 3),
    'QROM(Selector M=2)': lambda: make_selector(2, 0.0790150699, 0.05).qrom_bloq,
    'QROM(Selector M=3)': lambda: make_selector(3, 0.0790150699, 0.01).qrom_bloq,
    'Selector(M=2,eta=0.079,prec=0.05)': lambda: make_selector(2, 0.0790150699, 0.05),
    'Selector(M=3,eta=0.079,prec=0.01)': lambda: make_selector(3, 0.0790150699, 0.01),
}


def counts07(bloq):
    bs = BloqStream(bloq)
    c = count(bs)
    out = {k: v for k, v in c.items() if k not in ('ALLOC', 'FREE', 'GPHASE')}
    return {'counts': out, 'peak': bs.ctx.alloc.peak}


def cells():
    """data/reference_cells.json: for each n = 2^20 cell (Tables IV and V, and PS-F), its
    parameters and the reference stream's counts and peaks of P, P^dagger, C-X, W, the reflection,
    the phase-gradient state and the Grover iterate (from ../circuit/data/cells.json and
    stream_counts.json, written by ../circuit/tools/count_cells.py), with ALLOC, FREE and global
    phases left out."""
    import math
    data = os.path.join(HERE, '..', '..', 'circuit', 'data')
    cells_json = json.load(open(os.path.join(data, 'cells.json')))
    counts = json.load(open(os.path.join(data, 'stream_counts.json')))
    eta = 0.125 * (1 - math.exp(-1.0))
    out = {}
    for c in cells_json:
        acc = {'flat': '', 'F': '-F', 'T': '-T'}[c['access']]
        label = f"{c['family']}{acc} eps={c['eps']:g}"
        sc = counts[label]
        parts = {k: dict(counts={g: n for g, n in v['counts'].items()
                                 if g not in ('ALLOC', 'FREE', 'GPHASE')},
                         peak=v['peak'], T=v['T']) for k, v in sc['parts'].items()}
        sel = make_selector(c['M'], eta, c['sel_precision'])
        out[label] = dict(family=c['family'], access=c['access'], b=c['b'], eps=c['eps'], M=c['M'],
                          b_t=c['b_t'], w_phi=c['w_phi'], bv=c['bv'], bphi=c['bphi'],
                          sel_precision=c['sel_precision'], eta=eta,
                          alias=dict(alt=[int(x) for x in sel.alt],
                                     keep=[int(x) for x in sel.keep], mu=int(sel.mu)),
                          ts_per_rot=c['ts_per_rot'], nq=c['nq'], parts=parts, T_Q=sc['T_Q'],
                          peak_Q=sc['peak_Q'], T_total=sc['T_total'])
    path = os.path.join(HERE, '..', 'data', 'reference_cells.json')
    json.dump(out, open(path, 'w'), indent=1, sort_keys=True)
    print('wrote', os.path.abspath(path))


def main():
    cells()
    res = {}
    for name, fn in CASES.items():
        bloq = fn()
        res[name] = counts07(bloq)
        if name.startswith('Selector'):
            from ftfr.history import selector_weights
            res[name]['alias'] = dict(alt=[int(x) for x in bloq.alt], keep=[int(x) for x in bloq.keep],
                                      mu=int(bloq.mu),
                                      sum=float(bloq.sum_of_unnormalized_probabilities))
        print(f'{name:40s} {res[name]}')
    path = os.path.join(HERE, '..', 'data', 'reference_counts.json')
    json.dump(res, open(path, 'w'), indent=1, sort_keys=True)
    print('wrote', os.path.abspath(path))


if __name__ == '__main__':
    main()

"""pyLIQTR's resource estimate (Qualtran's call graph and T-complexity annotations) gives the same T
and rotation counts as walking each bloq down to its leaf gates, with one documented exception.

For a bloq without an explicit T-complexity annotation, pyLIQTR asks Qualtran's
`_from_directly_countable` before looking at its call graph, and that counts a gate on at most two
qubits whose unitary is a Clifford as one Clifford, whatever its decomposition.  Qualtran 0.4's
LessThanConstant(1, 1) is such a gate: x < 1 is NOT x, so the unitary is X, CNOT, X.  Its decomposition, which the compiler walks and which the reference
implementation and the cost model count, spends one And on it (4 T).  It is the step flag m >= 1
of a history with M = 1, so pyLIQTR reports 4 T fewer for P of such a history, and so 4 fewer per
block and 8 fewer per Grover iterate.  P^dagger holds Adjoint(LessThanConstant(1, 1)) instead,
which pyLIQTR counts through its call graph (4 T), as the stream does.
"""
import pytest
from qualtran.bloqs.arithmetic import LessThanConstant

from ftfr_q04.catalog import CATALOG
from ftfr_q04.stream import BloqStream, paper_counts, t_count
from ftfr_q04.testing import pyliqtr_view

pyliqtr = pytest.importorskip('pyLIQTR.utils.resource_analysis')


def _rotations(c):
    return c['RZ'] + c['RY']


# catalog entries with a history of M = 1, and the number of step flags m >= 1 computed in each (one
# in the P of every block)
STEP_FLAGS_M1 = {'Block(dense, b=2, M=1)': 1, 'HadamardTest(dense, b=2, M=1)': 1,
                 'GroverIterate(dense, b=2, M=1)': 2}


@pytest.mark.parametrize('name', list(CATALOG), ids=str)
def test_pyliqtr_estimate_equals_stream(name):
    bloq = CATALOG[name]()
    pc, direct = pyliqtr_view(bloq)
    est = pyliqtr.estimate_resources(bloq, profile=True)
    assert est['T'] == t_count(pc) + sum(tc.t for _, tc in direct)
    assert est['Rotations'] == _rotations(pc) + sum(tc.rotations for _, tc in direct)
    assert len(direct) == STEP_FLAGS_M1.get(name, 0)
    for sub, tc in direct:                  # each is the step flag, 4 T in the stream, 0 in pyLIQTR
        assert sub == LessThanConstant(1, 1) and tc.t == 0
        assert t_count(paper_counts(BloqStream(sub))) == 4

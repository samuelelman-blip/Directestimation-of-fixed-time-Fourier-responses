"""Reference values for the streamed blocks, from the paper's reference simulator (a numpy/scipy
state-vector simulation of one block that sums the histories directly).  The simulator is not part
of this repository: it is loaded if a copy is placed at
../../implementation_guide/reference/reference_block.py, and otherwise the tests that need it are
skipped (the amplitudes it produced are recorded in data/error_checks.json)."""
import math
import os
import sys

import numpy as np
import pytest

_REF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'implementation_guide',
                                    'reference'))
if os.path.exists(os.path.join(_REF, 'reference_block.py')):
    sys.path.insert(0, _REF)
    import reference_block as R  # noqa: E402
else:                                                   # pragma: no cover
    R = None

from ftfr_q04.access import profile_w, table_columns  # noqa: E402

needs_reference = pytest.mark.skipif(R is None, reason='reference simulator not found')


def family_for(cell):
    """The reference Family with the G1 the circuit block-encodes (tables: the rounded data)."""
    b = cell.b
    if cell.family == 'dense':
        return R.dense_family(b)
    if cell.family == 'PD' and cell.access == 'F':
        return R.profiled_family(b)
    n, nu, lam, z0, d = R._common(b, 1.0, cell.r_couple)
    rmu = cell.r_couple
    P = R.shift_matrix(n)
    if cell.access == 'T':
        cols = table_columns(b, cell.bv, cell.bpsi if cell.family == 'PD' else None)
        w = np.array(cols[0]) / 2 ** cell.bv
        psi = (2 * math.pi * np.array(cols[1]) / 2 ** cell.bpsi) if cell.family == 'PD' else np.zeros(n)
    elif cell.access == 'F':
        w, psi = profile_w(b), np.zeros(n)
    else:
        w, psi = np.ones(n), np.zeros(n)
    if cell.family in ('sparse', 'PS'):
        G1 = rmu * (w[:, None] * P)
    else:
        G1 = (rmu / nu) * (w * np.exp(1j * psi))[:, None] * (np.ones((n, n)) / math.sqrt(n) + P)
    G1 = G1.astype(complex)
    return R.Family(cell.family, b, n, 1.0, 1.0, rmu, nu, lam, G1, z0, d,
                    gamma=float(np.max(np.linalg.norm(G1, axis=1))), rho=1.0, C1=1.0,
                    w=w, psi=psi)


def reference_amplitude(cell, q_tilde=None):
    """<0|W|0> from the reference's history sum, with the selector weights q~ if given."""
    fam = family_for(cell)
    amp, per_order = R.structural_amplitude(fam, cell.M, cell.b_t, block=fam.G1 / fam.gamma)
    if q_tilde is None:
        return amp
    _, W, Xi = R.normalisation(fam, cell.M)
    q = fam.C1 * W / Xi
    return complex(np.sum(per_order * np.asarray(q_tilde) / q))

"""Rows, row access, vertex phases and the middle block C-X, simulated from their leaf gates against
their specifications; at the nine n = 2^20 cells C-X has exactly the gate counts and peak qubit
count of the reference implementation (the three cells with tables only with FTFR_SLOW=1, about
4 min)."""
import json
import math
import os

import numpy as np
import pytest
from qualtran import BloqBuilder, QBit, QUInt, Register

from ftfr_q04.access import make_access, profile_psi, profile_w, table_columns
from ftfr_q04.middle import MiddleBlock, VertexPhase
from ftfr_q04.rows import ContractionPhase, ControlledRowInverse, RowPrepDense, row_angles
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts
from ftfr_q04.testing import assert_close, prepare_phase_gradient, stream_matrix

CELLS = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_cells.json')))
SLOW = bool(os.environ.get('FTFR_SLOW'))


def shift_matrix(n):
    """P_{j l} = [l = j - 1 mod n]."""
    P = np.zeros((n, n))
    for j in range(n):
        P[j, (j - 1) % n] = 1
    return P


def phase_gradient_vector(w):
    return np.exp(-2j * math.pi * np.arange(2 ** w) / 2 ** w) / 2 ** (w / 2)


def test_vertex_phase():
    b, bt = 2, 2
    bloq = VertexPhase(b, bt)
    bphi = b + bt + 1
    phi = phase_gradient_vector(bphi)

    def prep(sim, ids):
        prepare_phase_gradient(sim, np.ravel(ids['phase_grad']))

    for j in range(2 ** b):
        for t in range(2 ** bt):
            col = ((j << bt) | t) << bphi
            got = stream_matrix(bloq, columns=[col], prepare=prep)[:, 0]
            want = np.zeros(2 ** (b + bt + bphi), dtype=complex)
            want[col:col + 2 ** bphi] = phi * np.exp(2j * math.pi * j * (2 * t + 1) / 2 ** bphi)
            assert_close(got, want, 1e-12, f'j={j} t={t}')


@pytest.mark.parametrize('b', [1, 2, 3])
def test_row_prep_dense(b):
    n = 2 ** b
    phi, alpha = row_angles(n)
    nu = math.sqrt(2 + 2 / math.sqrt(n))
    for j in range(n):
        col = j << (1 + b)
        got = stream_matrix(RowPrepDense(b), columns=[col])[:, 0]
        want = np.zeros(2 ** (2 * b + 1), dtype=complex)
        for l in range(n):
            amp = (1 / math.sqrt(n) + (1 if l == (j - 1) % n else 0)) / nu
            want[(j << (1 + b)) | l] = np.exp(1j * alpha) * amp
        assert_close(got, want, 1e-12, f'R j={j}')


@pytest.mark.parametrize('skeleton', ['dense', 'sparse'])
def test_contraction_core_block(skeleton):
    """Controlled R-dagger and the phase gate: the block to child 0 is i G1/gamma on ctrl = 1 and
    the identity on ctrl = 0."""
    b = 2
    n = 2 ** b
    G = ((np.ones((n, n)) / math.sqrt(n) + shift_matrix(n)) / math.sqrt(2 + 2 / math.sqrt(n))
         if skeleton == 'dense' else shift_matrix(n))
    bb = BloqBuilder()
    regs = dict(ctrl=bb.add_register(Register('ctrl', QBit())),
                j=bb.add_register(Register('j', QUInt(b))), l=bb.add_register(Register('l', QUInt(b))))
    if skeleton == 'dense':
        regs['a'] = bb.add_register(Register('a', QBit()))
    out = bb.add_d(ControlledRowInverse(b, skeleton), **regs)
    out['ctrl'] = bb.add(ContractionPhase(b, skeleton), ctrl=out['ctrl'])
    cbq = bb.finalize(**out)
    na = 1 if skeleton == 'dense' else 0
    for ctrl in (0, 1):
        for jj in range(n):
            for ll in range(n):
                col = ((((ctrl << b) | jj) << b) | ll) << na          # ctrl, j, l, a
                got = stream_matrix(cbq, columns=[col])[:, 0]
                zero = ((ctrl << b) | jj) << (b + na)
                if ctrl:
                    assert abs(got[zero] - 1j * G[jj, ll]) < 1e-12
                elif ll == 0:
                    assert abs(got[col] - 1) < 1e-12


def row_data(family, level, b, bv, bpsi):
    """w_j and psi_j that the row access encodes: exact (F) or from the rounded tables (T)."""
    if level == 'F':
        return profile_w(b), (profile_psi(b) if family == 'PD' else np.zeros(2 ** b))
    cols = table_columns(b, bv, bpsi if family == 'PD' else None)
    w = np.array(cols[0]) / 2 ** bv
    psi = 2 * math.pi * np.array(cols[1]) / 2 ** bpsi if family == 'PD' else np.zeros(2 ** b)
    return w, psi


@pytest.mark.parametrize('family,level', [('PD', 'F'), ('PS', 'F'), ('PD', 'T'), ('PS', 'T')])
def test_row_access_block(family, level):
    """With ctrl = 1 the block on m = 0, g = 0 is w_j e^{i psi_j}; with ctrl = 0 it is 1."""
    b = 2
    acc = make_access(family, level, b, b + 2, 2 * b + 2)
    w, psi = row_data(family, level, b, acc.bv, acc.bpsi if level == 'T' else 0)
    pb = acc.phase_bits
    phi = phase_gradient_vector(pb) if pb else np.ones(1)

    def prep(sim, ids):
        if pb:
            prepare_phase_gradient(sim, np.ravel(ids['phase_grad']))

    for ctrl in (0, 1):
        for jj in range(2 ** b):
            col = ((((ctrl << b) | jj) << acc.bv) << 1) << pb
            got = stream_matrix(acc, columns=[col], prepare=prep, measure_ands=True, seed=jj)[:, 0]
            blockv = got[col:col + 2 ** pb] if pb else got[col:col + 1]
            ratio = np.vdot(phi, blockv)
            want = w[jj] * np.exp(1j * psi[jj]) if ctrl else 1.0
            assert abs(ratio - want) < 1e-12, (ctrl, jj, ratio, want)


def middle_block(c):
    acc = (make_access(c['family'], c['access'], c['b'], c['bv'], c['bphi'])
           if c['family'] in ('PD', 'PS') else None)
    skeleton = 'dense' if c['family'] in ('dense', 'PD') else 'sparse'
    return MiddleBlock(c['M'], c['b'], c['b_t'], c['w_phi'], skeleton, acc)


@pytest.mark.parametrize('label', sorted(CELLS), ids=str)
def test_middle_block_counts_equal_reference(label):
    c = CELLS[label]
    if c['access'] == 'T' and not SLOW:
        pytest.skip('tabulated rows: set FTFR_SLOW=1 (about 4 minutes for the three cells)')
    ctx = StreamContext()
    got = {k: v for k, v in paper_counts(BloqStream(middle_block(c), ctx)).items()
           if k not in ('ALLOC', 'FREE')}
    assert got == c['parts']['X']['counts']
    assert ctx.alloc.peak == c['parts']['X']['peak']

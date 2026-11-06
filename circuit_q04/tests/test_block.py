"""Whole blocks streamed gate by gate against the reference simulator; the full-size counts of the
block, the reflection, the Grover iterate and the run against the cost model of the resource estimate
and against the reference implementation; the run plan.

The cells with closed-form or equal rows are streamed here (seconds each).  The cells with tables
take minutes, so by default their counts recorded by tools/count_cells.py (data/stream_counts.json)
are checked instead; set FTFR_SLOW=1 to stream them too (about 10 minutes, with the slow
Clifford+T blocks).
"""
import json
import os
from collections import Counter

import numpy as np
import pytest
from qualtran.bloqs.rotations import PhaseGradientState

from ftfr_q04.block import (Block, GroverIterate, ReflectZero, cell_model, load_cells, run_plan,
                            small_cell)
from ftfr_q04.history import TimeRegister
from ftfr_q04.simulate import selector_probabilities, simulate_block_amplitude
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts, t_count
from ftfr_q04.testing import assert_close, stream_matrix
from reference_bridge import R, family_for, needs_reference, reference_amplitude

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
CELLS = load_cells()
REF = json.load(open(os.path.join(DATA, 'reference_cells.json')))
SLOW = bool(os.environ.get('FTFR_SLOW'))

BLOCKS = [('dense', 'flat', 2, 1, 2), ('dense', 'flat', 2, 2, 2), ('dense', 'flat', 2, 3, 1),
          ('dense', 'flat', 3, 1, 2), ('sparse', 'flat', 2, 1, 2), ('sparse', 'flat', 3, 2, 2),
          ('sparse', 'flat', 2, 3, 1), ('PD', 'F', 2, 1, 2), ('PD', 'F', 2, 2, 1), ('PD', 'F', 3, 1, 1),
          ('PS', 'F', 2, 1, 2), ('PS', 'F', 3, 1, 1), ('PD', 'T', 2, 1, 1), ('PD', 'T', 2, 2, 1),
          ('PS', 'T', 2, 2, 1)]
IDS = ['-'.join(map(str, a)) for a in BLOCKS]


@needs_reference
@pytest.mark.parametrize('args', BLOCKS, ids=IDS)
def test_block_amplitude(args):
    """<0|W|0> of the streamed block equals the reference's history sum with the alias sampler's
    realised selector weights."""
    cell = small_cell(*args, sel_precision=0.05)
    amp, st = simulate_block_amplitude(cell)
    ref = reference_amplitude(cell, selector_probabilities(cell))
    assert abs(amp - ref) < 1e-12, (amp, ref)
    assert abs(st['norm'] - 1) < 1e-12


CT_RUNS = [(args, k + 1) for k, args in enumerate(BLOCKS)] + [(('PS', 'T', 2, 1, 2), 5)]
CT_QUICK = {('dense', 'flat', 2, 1, 2), ('sparse', 'flat', 2, 1, 2), ('PD', 'F', 2, 1, 2),
            ('PS', 'F', 2, 1, 2), ('PD', 'T', 2, 1, 1), ('PS', 'T', 2, 2, 1), ('PS', 'T', 2, 1, 2)}


@needs_reference
@pytest.mark.parametrize('args,seed', CT_RUNS, ids=['-'.join(map(str, a)) for a, _ in CT_RUNS])
def test_block_amplitude_clifford_t(args, seed):
    """The same, for every block above and one more, with every AND written out in Clifford+T
    gates and every uncompute measured with random outcomes (one block per family and row data by
    default, all sixteen with FTFR_SLOW=1)."""
    if args not in CT_QUICK and not SLOW:
        pytest.skip('larger blocks in Clifford+T: set FTFR_SLOW=1')
    cell = small_cell(*args, sel_precision=0.05)
    amp, _ = simulate_block_amplitude(cell, seed=seed, expand_ands=True)
    ref = reference_amplitude(cell, selector_probabilities(cell))
    assert abs(amp - ref) < 1e-12


@needs_reference
def test_exact_selector_limit():
    """With a precise alias sampler the realised weights approach q_m of Eq. (pomega)."""
    cell = small_cell('dense', 'flat', 2, 2, 1, sel_precision=1e-4)
    q = selector_probabilities(cell)
    fam = family_for(cell)
    _, W, Xi = R.normalisation(fam, 2)
    assert np.max(np.abs(q - fam.C1 * W / Xi)) < 1e-4


@needs_reference
@pytest.mark.parametrize('b_t', [1, 2, 3])
def test_time_register_equals_reference(b_t):
    m = stream_matrix(TimeRegister(b_t), columns=[0])[:, 0]
    ry_form, eq_form = R.product_state_amplitudes(b_t)
    assert_close(m, ry_form, 1e-12, 'time register')
    assert_close(ry_form, eq_form, 1e-12, 'Eq. (A1)')


def _stream(bloq, ts):
    ctx = StreamContext()
    c = paper_counts(BloqStream(bloq, ctx))
    return {k: v for k, v in c.items() if k not in ('ALLOC', 'FREE')}, ctx.alloc.peak, t_count(c, ts)


@pytest.mark.parametrize('cell', CELLS, ids=[c.label for c in CELLS])
def test_streamed_counts_equal_model_and_reference(cell):
    """W, the reflection, the phase-gradient state and the Grover iterate at n = 2^20: the
    reference implementation's gate counts and peaks, the model's T per block and per iterate,
    its qubit count and its run total."""
    if cell.access == 'T' and not SLOW:
        pytest.skip('tabulated rows take minutes to stream; set FTFR_SLOW=1')
    m, ref = cell_model(cell), REF[cell.label]['parts']
    ts = cell.ts_per_rot
    got = {}
    for name, bloq in (('W', Block(cell)), ('refl', ReflectZero(cell)),
                       ('pg', PhaseGradientState(cell.w_phi)), ('Q', GroverIterate(cell))):
        counts, peak, T = _stream(bloq, ts)
        assert counts == ref[name]['counts'], name
        assert peak == ref[name]['peak'], name
        got[name] = (T, peak)
    assert got['W'][0] == m['T_block']
    assert got['Q'][0] == m['T_Q'] and got['Q'][1] == m['qubits']
    assert got['pg'][0] == m['T_pg']
    assert cell.nq * got['Q'][0] + got['pg'][0] == m['T_total']


def test_recorded_counts_equal_model_and_reference():
    """The counts recorded by tools/count_cells.py, for every cell including the tabulated ones:
    the model's T per part, run total and qubits; the reference implementation's counts and peaks
    of every part; one iterate is two blocks and two reflections."""
    rec = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    assert sorted(rec) == sorted(c.label for c in CELLS)
    for cell in CELLS:
        m, r, ref = cell_model(cell), rec[cell.label], REF[cell.label]['parts']
        parts = r['parts']
        for key, part in parts.items():
            assert part['T'] == t_count(Counter(part['counts']), cell.ts_per_rot), (cell.label, key)
            assert part['counts'] == ref[key]['counts'], (cell.label, key)
            assert part['peak'] == ref[key]['peak'], (cell.label, key)
        if 'P' in parts:
            assert parts['P']['T'] == m['T_P'], cell.label
        assert parts['X']['T'] == m['T_X'], cell.label
        assert r['T_W'] == m['T_block'], cell.label
        assert r['T_Q'] == m['T_Q'], cell.label
        assert r['peak_Q'] == m['qubits'], cell.label
        assert parts['pg']['T'] == m['T_pg'], cell.label
        assert r['T_total'] == m['T_total'], cell.label
        assert r['T_Q'] == 2 * r['T_W'] + 2 * parts['refl']['T'], cell.label


def test_block_counts_by_gate_type():
    """The T count of one block follows from its counts by gate type; every AND is uncomputed by
    measurement."""
    rec = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    for cell in CELLS:
        c = Counter(rec[cell.label]['parts']['W']['counts'])
        rotations = c['RZ'] + c['RY']
        assert rec[cell.label]['T_W'] == 4 * c['AND'] + c['T'] + c['TDG'] + cell.ts_per_rot * rotations
        assert c['ANDDG'] == c['AND'], cell.label           # every AND uncomputed by measurement


def test_recorded_peaks_in_qualtran_04_order():
    """The peaks in Qualtran 0.4's own decomposition order (tools/count_cells.py --q04-order) equal
    the greedy ones, except at PS-T, where C-X, W and Q need one more qubit."""
    rec = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    for label, r in rec.items():
        for key in ('P', 'X', 'W', 'Q'):
            part = r['parts'][key]
            extra = 1 if label == 'PS-T eps=1e-06' and key != 'P' else 0
            assert part['peak_q04_order'] == part['peak'] + extra, (label, key)


# The gate table of the resource estimate (one row per cell; the Clifford counts of the cells with
# tables to two figures).  "qubits" is the peak of the Grover iterate.
TABLE_V_COLUMNS = ('M', 'b_t', 'ANDs', 'T', 'rotations', 'T per rotation', 'Cliffords',
                   'measurements', 'T per block', 'qubits')
TABLE_V = {
    'dense eps=0.01': (1, 7, 621, 240, 17, 57, 2658, 621, 3693, 180),
    'dense eps=1e-06': (4, 21, 5604, 720, 188, 107, 22252, 5604, 43252, 558),
    'PD-F eps=0.01': (1, 8, 1602, 240, 19, 61, 6345, 1602, 7807, 261),
    'PD-F eps=1e-06': (4, 22, 9564, 720, 196, 108, 37276, 9564, 60144, 743),
    'PD-T eps=0.01': (1, 10, 14832, 240, 23, 70, 1.5e7, 22768, 61178, 8093),
    'PD-T eps=1e-06': (4, 23, 76224, 720, 204, 116, 1.0e8, 105408, 329280, 7788),
    'sparse eps=1e-06': (1, 21, 1208, 120, 42, 99, 4495, 1208, 9110, 272),
    'PS-T eps=1e-06': (1, 23, 14855, 120, 46, 108, 1.3e7, 22279, 64508, 7651),
}
CLIFFORDS = ('X', 'Y', 'Z', 'H', 'S', 'SDG', 'CNOT', 'CZ', 'CZ_IF', 'Z_IF')


def test_block_counts_equal_table_v():
    """Every column of the gate table equals the streamed counts of one block W (conditioned CZ gates
    count as Cliffords; measurements are the AND uncomputes and the X-basis measurements of the
    lookup uncomputes)."""
    rec = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    for label, row in TABLE_V.items():
        r = rec[label]
        c = Counter(r['parts']['W']['counts'])
        assert set(c) <= set(CLIFFORDS) | {'AND', 'ANDDG', 'T', 'TDG', 'RZ', 'RY', 'MX'}, label
        got = (r['M'], r['b_t'], c['AND'], c['T'] + c['TDG'], c['RZ'] + c['RY'], r['ts_per_rot'],
               sum(c[k] for k in CLIFFORDS), c['ANDDG'] + c['MX'], r['T_W'], r['peak_Q'])
        for name, g, p in zip(TABLE_V_COLUMNS, got, row):
            if isinstance(p, float):
                assert float(f'{g:.2g}') == p, (label, name, g, p)
            else:
                assert g == p, (label, name, g, p)


def test_run_plan():
    """A run is the phase-gradient preparation once and the iterate `repeats` times."""
    rec = json.load(open(os.path.join(DATA, 'stream_counts.json')))
    for cell in CELLS:
        plan = run_plan(cell, rec[cell.label])
        assert plan['repeats'] == cell.nq
        assert isinstance(plan['iterate'], GroverIterate) and plan['iterate'].cell == cell
        assert isinstance(plan['prep'], PhaseGradientState) and plan['prep'].bitsize == cell.w_phi
        assert plan['T_total'] == plan['T_prep'] + plan['repeats'] * plan['T_iterate']
        assert plan['streamed']['T_Q'] == plan['T_iterate']
        assert plan['streamed']['peak_Q'] == plan['qubits']

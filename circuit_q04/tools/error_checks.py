"""The estimator's errors at small n and as n grows, for tools/plot_errors.py.

    python tools/error_checks.py                # every part (about 40 minutes)
    python tools/error_checks.py gate cells     # some parts

Writes data/error_checks.json; each part is stored as soon as it is done, and parts already in the
file are kept unless named.  The estimate of a run is the scalar the circuit's amplitude encodes,
Xi_M <0|W|0> (amplitude estimation then reads it to within its own allowance); its error is
measured against the exact response g(u(T)) of the full quadratic equation.

    gate        the streamed circuit (Qualtran 0.4), simulated gate by gate at n = 4, 8 and 16:
                its amplitude against the reference simulator's, and its estimate against g and
                against the calculated error for its parameters.  Also seven runs with every AND
                written out in Clifford+T gates and every uncompute measured at random.
    time        the error of the time grid against b_t = 1..12 (reference engine), with the
                Appendix A bound
    truncation  |g - g_M| for M = 0..6 at n = 4..2^20, with the two certified bounds
    selector    the error of the alias sampler's weights against its precision, with its bound
    scatter     measured against calculated error for 576 small configurations (reference engine)
    cells       the nine costed cells' circuit parameters at n = 4..2^20: measured errors and the
                calculated error budget

The reference engine is the reference simulator's dynamic programme over time cells
(../implementation_guide/reference/reference_block.py), which the gate-level runs reproduce to
within 1e-14; the responses at large n come from ftfr_q04.accuracy (O(n) per step).  Against n the
circuit is not simulated: the error of its estimate is formed from the exact responses, the realised
selector weights and, for M = 1, the time grid in closed form (for M = 4 the grid's error, below
1e-13 at the cells' b_t, is left out).
"""
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from ftfr_q04.accuracy import (instance, instance_for, order1, orders, response,  # noqa: E402
                               selector_bound, selector_realised, tail_bounds, time_bound,
                               truncation_bound, weights)
from ftfr_q04.block import load_cells, small_cell  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'error_checks.json')
ETA = 0.125 * (1 - math.exp(-1.0))
SIZES = list(range(2, 21, 2))                       # b = log2 n for the runs against n


def c2(z):
    return [float(np.real(z)), float(np.imag(z))]


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(data):
    json.dump(data, open(OUT, 'w'), indent=1, sort_keys=True)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def reference():
    from reference_bridge import R, family_for
    if R is None:
        raise RuntimeError('the reference simulator (../implementation_guide/reference) is needed')
    return R, family_for


def per_order(cell, b_t=None):
    """The reference engine: the order-m parts of <0|W|0> with the exact selector weights."""
    R, family_for = reference()
    fam = family_for(cell)
    _, po = R.structural_amplitude(fam, cell.M, cell.b_t if b_t is None else b_t,
                                   block=fam.G1 / fam.gamma)
    return po


def calculated(inst, M, b_t, precision):
    W, Xi = weights(inst, M)
    return dict(truncation=truncation_bound(inst, M), time=time_bound(inst, M, b_t),
                selector=selector_bound(Xi, M, precision), Xi=Xi)


# ------------------------------------------------------------------------------------------------
# gate level
# ------------------------------------------------------------------------------------------------

GATE = [('dense', 'flat', 2, 1, 2), ('dense', 'flat', 2, 2, 2), ('dense', 'flat', 2, 3, 1),
        ('dense', 'flat', 3, 1, 2), ('sparse', 'flat', 2, 1, 2), ('sparse', 'flat', 3, 2, 2),
        ('sparse', 'flat', 2, 3, 1), ('PD', 'F', 2, 1, 2), ('PD', 'F', 2, 2, 1),
        ('PD', 'F', 3, 1, 1), ('PS', 'F', 2, 1, 2), ('PS', 'F', 3, 1, 1), ('PD', 'T', 2, 1, 1),
        ('PD', 'T', 2, 2, 1), ('PS', 'T', 2, 2, 1),
        # larger time registers and n = 16
        ('dense', 'flat', 2, 1, 4), ('dense', 'flat', 2, 1, 5), ('dense', 'flat', 2, 2, 3),
        ('dense', 'flat', 3, 1, 4), ('PD', 'F', 2, 1, 4), ('sparse', 'flat', 3, 1, 4),
        ('sparse', 'flat', 4, 1, 2), ('dense', 'flat', 4, 1, 2)]
GATE_CT = [(('dense', 'flat', 2, 1, 2), 1), (('sparse', 'flat', 2, 1, 2), 5), (('PD', 'F', 2, 1, 2), 8),
           (('PS', 'F', 2, 1, 2), 11), (('PD', 'T', 2, 1, 1), 13), (('PS', 'T', 2, 2, 1), 15),
           (('PS', 'T', 2, 1, 2), 5)]
GATE_PRECISION = 0.05
# finer selectors (the alias sampler's state grows as 1/precision, so only small blocks)
GATE_FINE = [(('dense', 'flat', 2, 1, 2), 0.01), (('dense', 'flat', 2, 1, 2), 0.002),
             (('dense', 'flat', 2, 1, 2), 0.0005), (('sparse', 'flat', 3, 1, 2), 0.002)]


def part_gate(data):
    from ftfr_q04.simulate import simulate_block_amplitude
    from reference_bridge import reference_amplitude
    rows = []
    runs = ([(a, None, GATE_PRECISION) for a in GATE] + [(a, None, p) for a, p in GATE_FINE]
            + [(a, s, GATE_PRECISION) for a, s in GATE_CT])
    for args, seed, precision in runs:
        cell = small_cell(*args, sel_precision=precision)
        inst = instance_for(cell)
        t0 = time.time()
        if seed is None:
            amp, st = simulate_block_amplitude(cell)
        else:
            amp, st = simulate_block_amplitude(cell, seed=seed, expand_ands=True)
        secs = time.time() - t0
        q_real = selector_realised(cell.M, cell.eta, precision)
        amp_ref = reference_amplitude(cell, q_real)
        po = per_order(cell)
        W, Xi = weights(inst, cell.M)
        h = orders(inst, cell.M)
        g = response(inst)
        est = Xi * amp
        calc = calculated(inst, cell.M, cell.b_t, precision)
        rows.append(dict(
            args=list(args), seed=seed, clifford_t=seed is not None, n=inst.n, M=cell.M,
            b_t=cell.b_t, precision=precision, ops=st['ops'], peak=st['peak'],
            seconds=round(secs, 1), amp=c2(amp), amp_reference=c2(amp_ref),
            amp_difference=abs(amp - amp_ref), estimate=c2(est), g=c2(g),
            error=abs(est - g), truncation=abs(g - h.sum()),
            time=abs(Xi * po.sum() - h.sum()), selector=abs(Xi * (amp_ref - po.sum())),
            calculated=calc, calculated_total=calc['truncation'] + calc['time'] + calc['selector']))
        log('gate', args, 'seed', seed, 'precision', precision, f'{secs:.1f}s', 'amp diff',
            f'{abs(amp - amp_ref):.1e}',
            'error', f'{abs(est - g):.2e}', 'calculated', f"{rows[-1]['calculated_total']:.2e}")
    data['gate'] = rows


# ------------------------------------------------------------------------------------------------
# reference-engine sweeps
# ------------------------------------------------------------------------------------------------

def part_time(data):
    rows = []
    for (fam, acc, b) in [('dense', 'flat', 2), ('PD', 'F', 2), ('dense', 'flat', 4)]:
        inst = instance(fam, acc, b)
        h = orders(inst, 3)
        for M in (1, 2, 3):
            W, Xi = weights(inst, M)
            for b_t in range(1, 13):
                cell = small_cell(fam, acc, b, M, b_t)
                po = per_order(cell)
                rows.append(dict(family=fam, access=acc, b=b, n=inst.n, M=M, b_t=b_t,
                                 error=abs(Xi * po.sum() - h[:M + 1].sum()),
                                 bound_mean=time_bound(inst, M, b_t),
                                 bound_worst=time_bound(inst, M, b_t, worst=True)))
            log('time', fam, acc, b, 'M', M, [f"{r['error']:.1e}" for r in rows[-12:]])
    data['time'] = rows


def responses(data, key, inst, M):
    """g and h_0..h_M for an instance, cached in data['responses']."""
    store = data.setdefault('responses', {}).setdefault(key, {})
    hit = store.get(str(inst.b))
    if hit is not None and len(hit['h']) >= M + 1:
        return complex(*hit['g']), np.array([complex(*z) for z in hit['h']])
    t0 = time.time()
    h = orders(inst, M)
    g = response(inst)
    store[str(inst.b)] = dict(g=c2(g), h=[c2(z) for z in h])
    log('responses', key, 'b', inst.b, 'M', M, f'{time.time() - t0:.1f}s')
    return g, h


def inst_key(family, access, bv=0, bpsi=0):
    return f'{family}-{access}' + (f'-bv{bv}-bpsi{bpsi}' if access == 'T' else '')


def part_truncation(data):
    rows = []
    for fam, acc in (('dense', 'flat'), ('PD', 'F'), ('sparse', 'flat')):
        for b in SIZES:
            inst = instance(fam, acc, b)
            g, h = responses(data, inst_key(fam, acc), inst, 6)
            for M in range(7):
                e22, e1 = tail_bounds(inst, M)
                rows.append(dict(family=fam, access=acc, b=b, n=inst.n, M=M,
                                 error=abs(g - h[:M + 1].sum()), bound_22=e22, bound_inf1=e1))
            save(data)
    data['truncation'] = rows


def part_selector(data):
    rows = []
    inst = instance('dense', 'flat', 2)
    precisions = [10 ** (-k / 2) for k in range(2, 13)]
    for M in (1, 2, 3, 4):
        cell = small_cell('dense', 'flat', 2, M, 14)
        po = per_order(cell)
        W, Xi = weights(inst, M)
        q = W / W.sum()
        for p in precisions:
            q_real = selector_realised(M, ETA, p)
            rows.append(dict(M=M, precision=p, n=inst.n, b_t=14,
                             error=abs(Xi * np.sum((q_real / q - 1) * po)),
                             bound=selector_bound(Xi, M, p),
                             max_weight_error=float(np.max(np.abs(q_real - q)))))
        log('selector M', M, [f"{r['error']:.1e}" for r in rows[-len(precisions):]])
    data['selector'] = rows


def part_scatter(data):
    rows = []
    fams = [('dense', 'flat'), ('PD', 'F'), ('sparse', 'flat'), ('PS', 'F'), ('PD', 'T'),
            ('PS', 'T')]
    for (fam, acc), b in itertools.product(fams, (2, 3)):
        inst0 = instance_for(small_cell(fam, acc, b, 1, 1))
        g = response(inst0)
        h = orders(inst0, 3)
        for M, b_t in itertools.product((1, 2, 3), (2, 4, 6, 8)):
            cell = small_cell(fam, acc, b, M, b_t)
            po = per_order(cell)
            W, Xi = weights(inst0, M)
            q = W / W.sum()
            for p in (1e-1, 1e-2, 1e-3, 1e-4):
                q_real = selector_realised(M, ETA, p)
                est = Xi * np.sum(q_real / q * po)
                calc = calculated(inst0, M, b_t, p)
                rows.append(dict(family=fam, access=acc, b=b, n=inst0.n, M=M, b_t=b_t,
                                 precision=p, error=abs(est - g),
                                 truncation=abs(g - h[:M + 1].sum()),
                                 time=abs(Xi * po.sum() - h[:M + 1].sum()),
                                 selector=abs(Xi * np.sum((q_real / q - 1) * po)),
                                 calculated=calc,
                                 calculated_total=calc['truncation'] + calc['time']
                                 + calc['selector']))
        log('scatter', fam, acc, b, len(rows))
    data['scatter'] = rows


# ------------------------------------------------------------------------------------------------
# the nine cells against n
# ------------------------------------------------------------------------------------------------

def part_cells(data):
    cells = load_cells()
    budget = json.load(open(os.path.join(ROOT, 'data', 'error_budget.json')))
    model = {c['family'] + c['access'] + f"{c['eps']:g}": c
             for c in json.load(open(os.path.join(ROOT, 'data', 'cells.json')))}
    out = {}
    for cell in cells:
        raw = model[cell.family + cell.access + f'{cell.eps:g}']
        rows = []
        for b in SIZES:
            inst = instance_for(cell, b)
            g, h = responses(data, inst_key(cell.family, cell.access, cell.bv, cell.bpsi), inst,
                             max(cell.M, 1))
            h = h[:cell.M + 1]
            W, Xi = weights(inst, cell.M)
            q = W / W.sum()
            q_real = selector_realised(cell.M, ETA, cell.sel_precision) if cell.M else np.ones(1)
            h_grid = h.copy()
            if cell.M == 1:                            # the time grid exactly (closed form)
                h_grid[1] = order1(inst, cell.b_t)
            est = complex(np.sum(q_real / q * h_grid))
            calc = calculated(inst, cell.M, cell.b_t, cell.sel_precision)
            calc['rotations'] = Xi * raw['n_rot_block'] * raw['eps_rot']
            rows.append(dict(
                b=b, n=inst.n, g=c2(g), estimate=c2(est), error=abs(est - g),
                truncation=abs(g - h.sum()),
                selector=abs(complex(np.sum((q_real / q - 1) * h_grid))),
                time=abs(h_grid[1] - order1(inst)) if cell.M == 1 else None,
                calculated=calc))
        out[cell.label] = dict(M=cell.M, b_t=cell.b_t, eps=cell.eps,
                               sel_precision=cell.sel_precision, budget=budget[cell.label],
                               rows=rows)
        log('cells', cell.label, [f"{r['error']:.1e}" for r in rows])
        data['cells'] = out
        save(data)


PARTS = dict(gate=part_gate, time=part_time, truncation=part_truncation, selector=part_selector,
             scatter=part_scatter, cells=part_cells)


def main(names):
    data = load()
    for name in names or list(PARTS):
        t0 = time.time()
        PARTS[name](data)
        save(data)
        log('done', name, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])

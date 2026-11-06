"""The point beyond the nine cells: the dense family at n = 2^40 to absolute error 1e-8.

The nine costed cells (data/cells.json) are at n = 2^20.  The paper also quotes one point beyond
them: a complete run on the dense family with n = 2^40 coordinates to absolute error 1e-8 uses
3.6e14 T gates and 1,169 logical qubits.  This script streams the Grover iterate at that point
and forms the run total the way data/cells.json does (T of the phase-gradient state once, plus
nq times the T of one iterate), so the point can be checked from this package alone.

The circuit parameters are the cost model's choices, read off the nine cells:

- M, the number of interactions kept: the smallest whose certified truncation error (the
  (inf, 1) certificate of ftfr_q04.accuracy.truncation_bound, as in data/error_budget.json) is
  below eps.  At n = 2^40 and eps = 1e-8 that is M = 6; M = 5 certifies 1.19e-8.
- b_t, the bits of each time register: 28, at which the time-grid bound (mean order, as the
  cost model takes it) is 4.0e-10, four per cent of eps, as at the cells with eps = 1e-6.
- w_phi = b + b_t + 1 bits of phase gradient, as at every cell without row data.
- the selector's precision, 2^-38 (an alias sampler with mu = 36 bits), so that the selector
  error Xi_M (M + 1) 2^-38 is 1e-11.
- T per rotation 130, from the rule the nine cells follow, ts = 3 log2(1/eps_rot) + 10 with
  eps_rot = 2^-40 the precision of each rotation; the rotation error is Xi_M times the number of
  rotations in a block times eps_rot.
- the readout takes what is left of eps, and the number of iterates is nq = c Xi_M / eps_read
  with c = nq eps_read / Xi_M of the nine cells, which is 26.65 in each of them (the empirical
  constant of the amplitude estimation).

With these, the iterate has 300,912 T gates and 1,169 qubits, and the run 3.6e14 T gates.  Any of
the parameters can be overridden; the script first streams the dense eps = 1e-6 cell of
data/cells.json as a check of the path (88,560 T per iterate and 558 qubits, as recorded).

    python3 tools/headline.py                       # the paper's point; writes data/headline.json
    python3 tools/headline.py --b 30 --eps 1e-7     # another point (M chosen by the certificate,
                                                    # the rest as given or by the defaults)
    python3 tools/headline.py --b_t 29 --ts 134     # override a parameter
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qualtran.bloqs.rotations import PhaseGradientState  # noqa: E402

from ftfr_q04.block import Cell, GroverIterate, load_cells  # noqa: E402
from ftfr_q04.history import make_selector  # noqa: E402
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts, t_count  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
R, MU, T_FINAL = 0.125, 1.0, 1.0        # the instances of the resource estimate (ftfr_q04.accuracy)


def dense_bounds(b, M, b_t, sel_precision):
    """Xi_M and the truncation, time-grid and selector bounds of the dense family at n = 2^b, in
    closed form (ftfr_q04.accuracy builds the instance, which needs O(n) memory; these are the same
    expressions).  Truncation: the better of the (2, 2) and (inf, 1) certificates, as the cost
    model.  Time grid: Appendix A at the mean order.  Selector: Xi_M (M + 1) precision."""
    n = 2.0 ** b
    tau = (1 - math.exp(-MU * T_FINAL)) / MU
    pre = math.exp(-MU * T_FINAL)
    eta = R * MU * tau                                            # gamma tau, gamma = r mu
    nu = math.sqrt(2 + 2 / math.sqrt(n))
    eta1 = (R * MU / nu) * (math.sqrt(n) + 1) * tau / math.sqrt(n)  # gamma_1 tau / sqrt(n)
    W = [pre * eta ** m for m in range(M + 1)]
    Xi = sum(W)
    trunc = min(pre * eta ** (M + 1) / (1 - eta), pre * eta1 ** (M + 1) / (1 - eta1))
    L = 2 * math.pi * MU * (n - 1) / n
    mean_m = sum(m * w for m, w in enumerate(W)) / Xi
    tgrid = mean_m * L * T_FINAL * 2.0 ** (-(b_t + 1)) * Xi
    sel = Xi * (M + 1) * sel_precision
    return Xi, trunc, tgrid, sel


def smallest_M(b, eps):
    for M in range(1, 64):
        if dense_bounds(b, M, 1, 0.0)[1] < eps:
            return M
    raise ValueError('no M certifies eps = %g' % eps)


def stream(bloq, ts):
    ctx = StreamContext()
    c = paper_counts(BloqStream(bloq, ctx))
    return c, ctx.alloc.peak, t_count(c, ts)


def amplitude_constant():
    """nq eps_read / Xi_M over the nine cells (26.65 in each)."""
    budget = json.load(open(os.path.join(DATA, 'error_budget.json')))
    ratios = []
    for c in json.load(open(os.path.join(DATA, 'cells.json'))):
        acc = {'flat': '', 'F': '-F', 'T': '-T'}[c['access']]
        b = budget[f"{c['family']}{acc} eps={c['eps']:g}"]
        ratios.append(c['nq'] * b['readout'] / c['Xi_M'])
    return sum(ratios) / len(ratios), min(ratios), max(ratios)


def check_recorded_cell():
    cell = [c for c in load_cells() if c.label == 'dense eps=1e-06'][0]
    _, peak, TQ = stream(GroverIterate(cell), cell.ts_per_rot)
    print(f'check: {cell.label} at n = 2^{cell.b}: {TQ:,} T per iterate, {peak} qubits '
          f'(recorded 88,560 and 558)')
    assert (TQ, peak) == (88560, 558)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--b', type=int, default=40, help='n = 2^b (default 40)')
    ap.add_argument('--eps', type=float, default=1e-8, help='absolute error (default 1e-8)')
    ap.add_argument('--M', type=int, help='interactions kept (default: the smallest certified)')
    ap.add_argument('--b_t', type=int, default=28, help='time-register bits (default 28)')
    ap.add_argument('--sel_bits', type=int, default=38,
                    help='selector precision 2^-sel_bits (default 38)')
    ap.add_argument('--ts', type=int, default=130, help='T per rotation (default 130)')
    ap.add_argument('--out', default=os.path.join(DATA, 'headline.json'),
                    help='where to write the result (default data/headline.json; "-" for none)')
    ap.add_argument('--no-check', action='store_true', help='skip the n = 2^20 check')
    a = ap.parse_args()

    if not a.no_check:
        check_recorded_cell()

    M = a.M or smallest_M(a.b, a.eps)
    sel_precision = 2.0 ** (-a.sel_bits)
    eps_rot = 2.0 ** (-(a.ts - 10) / 3)
    w_phi = a.b + a.b_t + 1
    cell = Cell('dense', 'flat', a.b, M, a.b_t, w_phi, sel_precision, ts_per_rot=a.ts, eps=a.eps)

    cQ, peak, TQ = stream(GroverIterate(cell), a.ts)
    cpg, _, Tpg = stream(PhaseGradientState(w_phi), a.ts)
    rotations = cQ['RZ'] + cQ['RY']                  # per iterate; a block has half of them
    Xi, trunc, tgrid, sel = dense_bounds(a.b, M, a.b_t, sel_precision)
    rot = Xi * (rotations // 2) * eps_rot
    read = a.eps - trunc - tgrid - sel - rot
    if read <= 0:
        sys.exit('the truncation, time-grid, selector and rotation errors already exceed eps')
    c, c_min, c_max = amplitude_constant()
    nq = math.ceil(c * Xi / read)
    total = Tpg + nq * TQ

    out = dict(
        family='dense', access='flat', b=a.b, n=2 ** a.b, eps=a.eps, M=M, b_t=a.b_t, w_phi=w_phi,
        sel_precision=sel_precision, selector_bits=make_selector(M, cell.eta, sel_precision).mu,
        ts_per_rot=a.ts, eps_rot=eps_rot, Xi_M=Xi,
        budget=dict(truncation=trunc, time=tgrid, selector=sel, rotations=rot, readout=read),
        amplitude_constant=dict(mean=c, min=c_min, max=c_max),
        iterate=dict(T=TQ, qubits=peak, rotations=rotations,
                     counts={k: v for k, v in sorted(cQ.items()) if k not in ('ALLOC', 'FREE')}),
        phase_gradient_T=Tpg, nq=nq, T_total=total,
        note='T counts include the rotations at ts_per_rot T each; qubits is the peak of the '
             'iterate in the greedy order, as data/cells.json',
    )
    print(f'dense family, n = 2^{a.b}, eps = {a.eps:g}: M = {M}, b_t = {a.b_t}, w_phi = {w_phi}, '
          f'selector 2^-{a.sel_bits} (mu = {out["selector_bits"]}), {a.ts} T per rotation')
    print(f'  Xi_M = {Xi:.6f}; truncation {trunc:.3e}, time grid {tgrid:.3e}, selector {sel:.3e}, '
          f'rotations {rot:.3e}, readout {read:.3e}')
    print(f'  iterate: {TQ:,} T ({rotations} rotations), {peak} qubits; phase gradient {Tpg:,} T')
    print(f'  nq = {nq:,} iterates (c = {c:.4f}); run total {total:.4e} T')
    if a.out != '-':
        json.dump(out, open(a.out, 'w'), indent=1)
        print(f'  written to {os.path.relpath(a.out)}')


if __name__ == '__main__':
    main()

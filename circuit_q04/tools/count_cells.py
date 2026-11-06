"""Stream the nine costed cells of the resource estimate at n = 2^20 and count their gates
exactly.

For each cell: P, P^dagger, C-X, one block W, the reflection, the phase-gradient state and one
Grover iterate, streamed gate by gate; T counts with the cell's T per rotation; the peak number of
live qubits.  Writes data/stream_counts.json, which tests/test_block.py checks against the cost
model (data/cells.json) and against the reference implementation (data/reference_cells.json).  The
cells with tables take minutes each.

    python3 tools/count_cells.py                 # every cell
    python3 tools/count_cells.py "PD-T eps=0.01" # some cells, by label
    python3 tools/count_cells.py --q04-order     # add the peaks of P, C-X, W and Q in Qualtran
                                                 # 0.4's own decomposition order
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qualtran import Adjoint  # noqa: E402
from qualtran.bloqs.rotations import PhaseGradientState  # noqa: E402

from ftfr_q04.block import Block, GroverIterate, ReflectZero, block_parts, load_cells  # noqa: E402
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts, t_count  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'stream_counts.json')


def run(bloq, ts_per_rot):
    t0 = time.time()
    ctx = StreamContext()
    c = paper_counts(BloqStream(bloq, ctx))
    counts = {k: v for k, v in sorted(c.items()) if k not in ('ALLOC', 'FREE')}
    return dict(counts=counts, peak=ctx.alloc.peak, T=t_count(c, ts_per_rot),
                seconds=round(time.time() - t0, 1))


def peak_q04_order(bloq):
    """The peak number of live qubits when each decomposition runs in Qualtran 0.4's
    iter_bloqnections order (every allocation of a decomposition at its start)."""
    ctx = StreamContext(greedy=False)
    for _ in BloqStream(bloq, ctx):
        pass
    return ctx.alloc.peak


def add_q04_peaks(selected=None):
    results = json.load(open(OUT))
    for cell in load_cells():
        if selected and cell.label not in selected:
            continue
        parts = results[cell.label]['parts']
        prep, mid = block_parts(cell)
        todo = [('X', mid), ('W', Block(cell)), ('Q', GroverIterate(cell))]
        if prep is not None:
            todo.insert(0, ('P', prep))
        for key, bloq in todo:
            if 'peak_q04_order' not in parts[key]:
                parts[key]['peak_q04_order'] = peak_q04_order(bloq)
        print(cell.label, {k: (parts[k]['peak'], parts[k]['peak_q04_order']) for k, _ in todo},
              flush=True)
        json.dump(results, open(OUT, 'w'), indent=1, sort_keys=True)


def main(selected=None):
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for cell in load_cells():
        if selected and cell.label not in selected:
            continue
        ts = cell.ts_per_rot
        prep, mid = block_parts(cell)
        parts = {}
        if prep is not None:
            parts['P'] = run(prep, ts)
            parts['Pdg'] = run(Adjoint(prep), ts)
        parts['X'] = run(mid, ts)
        parts['W'] = run(Block(cell), ts)
        parts['refl'] = run(ReflectZero(cell), ts)
        parts['pg'] = run(PhaseGradientState(cell.w_phi), ts)
        parts['Q'] = run(GroverIterate(cell), ts)
        res = dict(label=cell.label, M=cell.M, b_t=cell.b_t, w_phi=cell.w_phi, ts_per_rot=ts,
                   nq=cell.nq, parts=parts, T_W=parts['W']['T'], T_Q=parts['Q']['T'],
                   peak_Q=parts['Q']['peak'], T_total=cell.nq * parts['Q']['T'] + parts['pg']['T'])
        results[cell.label] = res
        print(cell.label, 'T_W', res['T_W'], 'T_Q', res['T_Q'], 'peak', res['peak_Q'],
              'T_total', f"{res['T_total']:.4e}",
              'seconds', {k: v['seconds'] for k, v in parts.items()}, flush=True)
        json.dump(results, open(OUT, 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--q04-order' in args:
        args.remove('--q04-order')
        add_q04_peaks(args or None)
    else:
        main(args or None)

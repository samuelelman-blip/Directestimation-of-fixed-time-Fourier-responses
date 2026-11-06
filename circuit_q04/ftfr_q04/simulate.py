"""Simulating streamed blocks at small n: the block amplitude <0|W|0> and the selector weights."""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from .block import Block, Cell
from .history import make_selector
from .stream import BloqStream, Op, Parity, SparseSim, StreamContext
from .testing import output_distribution, prepare_phase_gradient


def expand_and(ops: Iterable[Op], ctx: Optional[StreamContext] = None) -> Iterator[Op]:
    """AND and ANDDG written out exactly as Qualtran's And.decompose_from_registers writes them:
    X on each control whose value is 0, around both; AND as Clifford+T gates (4 T); ANDDG as an
    X measurement of the target and a CZ on the controls conditioned on the outcome."""
    ctx = ctx or StreamContext()
    for op in ops:
        name = op[0]
        if name not in ('AND', 'ANDDG'):
            yield op
            continue
        c1, c2, t = op[1]
        cv = op[2] if isinstance(op[2], tuple) else (1, 1)
        flips = [c for c, v in zip((c1, c2), cv) if v == 0]
        for c in flips:
            yield ('X', (c,), None)
        if name == 'AND':
            for g, qs in (('H', (t,)), ('T', (t,)), ('CNOT', (c1, t)), ('CNOT', (c2, t)),
                          ('CNOT', (t, c1)), ('CNOT', (t, c2)), ('TDG', (c1,)), ('TDG', (c2,)),
                          ('T', (t,)), ('CNOT', (t, c1)), ('CNOT', (t, c2)), ('H', (t,)),
                          ('S', (t,))):
                yield (g, qs, None)
        else:
            key = ctx.new_cbit()
            yield ('MX', (t,), key)
            yield ('CZ', (c1, c2), Parity([key]))
        for c in flips:
            yield ('X', (c,), None)


def simulate_block_amplitude(cell: Cell, seed: int = 0, expand_ands: bool = False
                             ) -> Tuple[complex, Dict]:
    """<0|W|0>: every register |0>, anc set to 1, the phase gradient prepared; stream the block;
    undo the phase gradient and anc; return the amplitude of the all-zero state.  With
    `expand_ands` every AND is written out in Clifford+T gates and every uncompute measured, with
    random outcomes drawn from `seed`."""
    blk = Block(cell)
    ctx = StreamContext()
    sim = SparseSim(seed=seed)
    ids = {}
    for reg in blk.signature.lefts():
        ids[reg.name] = np.array(ctx.alloc.allocate(reg.total_bits()),
                                 dtype=np.int64).reshape(reg.shape + (reg.bitsize,))
        for q in np.ravel(ids[reg.name]):
            sim.apply(('ALLOC', (int(q),), None))
    anc = int(np.ravel(ids['anc'])[0])
    sim.apply(('X', (anc,), None))
    prepare_phase_gradient(sim, np.ravel(ids['phase_grad']))
    bs = BloqStream(blk, ctx, inputs=ids)
    ops = iter(bs)
    if expand_ands:
        ops = expand_and(ops, ctx)
    n_ops = 0
    for op in ops:
        sim.apply(op)
        n_ops += 1
    out = bs.outputs
    prepare_phase_gradient(sim, np.ravel(out['phase_grad']), inverse=True)
    sim.apply(('X', (int(np.ravel(out['anc'])[0]),), None))
    stats = dict(ops=n_ops, peak=ctx.alloc.peak, norm=sim.norm())
    return sim.amplitude_all_zero(), stats


def selector_probabilities(cell: Cell) -> np.ndarray:
    """q~_m: the alias sampler's realised probabilities of m = 0..M (junk traced out)."""
    sel = make_selector(cell.M, cell.eta, cell.sel_precision)
    dist = output_distribution(sel, ['selection'], measure_ands=False)
    return np.array([dist.get((m,), 0.0) for m in range(cell.M + 1)])

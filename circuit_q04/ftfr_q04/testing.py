"""Helpers for checking walked circuits against Qualtran 0.4's tensor simulation and against
specifications on basis states."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from qualtran import Bloq

from .stream import BloqStream, SparseSim, StreamContext


def register_qubits(regs, ids: Dict[str, np.ndarray]) -> List[int]:
    out: List[int] = []
    for reg in regs:
        out.extend(int(x) for x in np.ravel(ids[reg.name]))
    return out


def _start(bloq: Bloq, k: int, seed: int, measure_ands: bool, prepare=None):
    lefts = list(bloq.signature.lefts())
    n_in = sum(r.total_bits() for r in lefts)
    ctx = StreamContext()
    ids = {}
    for reg in lefts:
        ids[reg.name] = np.array(ctx.alloc.allocate(reg.total_bits()),
                                 dtype=np.int64).reshape(reg.shape + (reg.bitsize,))
    in_q = register_qubits(lefts, ids)
    sim = SparseSim(seed=seed, measure_ands=measure_ands)
    for j, q in enumerate(in_q):
        sim.apply(('ALLOC', (q,), None))
        if (k >> (n_in - 1 - j)) & 1:
            sim.apply(('X', (q,), None))
    if prepare is not None:
        prepare(sim, ids)
    bs = BloqStream(bloq, ctx, inputs=ids)
    return sim, bs


def stream_matrix(bloq: Bloq, columns: Optional[Sequence[int]] = None, *, measure_ands=False,
                  seed: int = 0, prepare=None, finish=None) -> np.ndarray:
    """The matrix of `bloq` from simulating its leaves, in Qualtran's tensor convention: rows index
    the right registers, columns the left registers, signature order, each register big-endian.
    `columns` restricts the input basis states.  `prepare(sim, in_ids)` acts on the inputs before
    the bloq and `finish(sim, out_ids)` on the outputs after it (used for phase-gradient states)."""
    lefts = list(bloq.signature.lefts())
    rights = list(bloq.signature.rights())
    n_in = sum(r.total_bits() for r in lefts)
    n_out = sum(r.total_bits() for r in rights)
    cols = range(2 ** n_in) if columns is None else columns
    mat = np.zeros((2 ** n_out, len(cols)), dtype=complex)
    for ci, k in enumerate(cols):
        sim, bs = _start(bloq, k, seed + ci, measure_ands, prepare)
        sim.run(iter(bs))
        if finish is not None:
            finish(sim, bs.outputs)
        out_q = register_qubits(rights, bs.outputs)
        mat[:, ci] = sim.vector(out_q)
    return mat


def stream_basis_map(bloq: Bloq, k: int, seed: int = 0, measure_ands: bool = False):
    """Simulate on the basis input k (left registers, signature order, big-endian) and return
    (output basis value over the right registers, amplitude); the output must be a basis state and
    every other qubit |0>."""
    rights = list(bloq.signature.rights())
    sim, bs = _start(bloq, k, seed, measure_ands)
    sim.run(iter(bs))
    out_q = register_qubits(rights, bs.outputs)
    if sim.amp.size != 1:
        raise AssertionError(f'output is not a basis state ({sim.amp.size} terms)')
    for q in sim.slot:
        if q not in out_q and int(sim.bit(q)[0]):
            raise AssertionError('a qubit outside the outputs is not |0>')
    v = 0
    for q in out_q:
        v = (v << 1) | int(sim.bit(q)[0])
    return v, complex(sim.amp[0])


def output_distribution(bloq: Bloq, names: Sequence[str], k: int = 0, seed: int = 0,
                        measure_ands: bool = True) -> Dict[tuple, float]:
    """Simulate `bloq` on basis input k and return the marginal distribution of the right
    registers `names`: {(value of each register, ...): probability}.  A shaped register's value is
    the tuple of its entries; each entry is read as Qualtran reads it, most significant qubit
    first.  All other registers and qubits are traced out."""
    sim, bs = _start(bloq, k, seed, measure_ands)
    sim.run(iter(bs))
    probs = np.abs(sim.amp) ** 2
    cols = []
    for name in names:
        reg = bloq.signature.get_right(name)
        ids = np.asarray(bs.outputs[name]).reshape(-1, reg.bitsize)
        vals = []
        for row in ids:
            v = np.zeros(sim.amp.size, dtype=np.int64)
            for q in row:
                v = (v << 1) | sim.bit(int(q)).astype(np.int64)
            vals.append(v)
        cols.append((bool(reg.shape), vals))
    out: Dict[tuple, float] = {}
    for j in range(sim.amp.size):
        key = tuple(tuple(int(v[j]) for v in vals) if shaped else int(vals[0][j])
                    for shaped, vals in cols)
        out[key] = out.get(key, 0.0) + float(probs[j])
    return out


def prepare_phase_gradient(sim: SparseSim, qubits_big_endian, inverse: bool = False):
    """|phi> = 2^{-w/2} sum_k e^{-2 pi i k / 2^w} |k> on the given qubits (first = most significant):
    H then diag(1, e^{-2 pi i 2^(w-1-m) / 2^w}) on qubit m; `inverse` undoes it."""
    qs = [int(q) for q in qubits_big_endian]
    w = len(qs)
    steps = [(q, -2 * 2 ** (w - 1 - m) / 2 ** w) for m, q in enumerate(qs)]
    if inverse:
        for q, t in reversed(steps):
            sim.apply(('ZPOW', (q,), -t))
            sim.apply(('H', (q,), None))
    else:
        for q, t in steps:
            sim.apply(('H', (q,), None))
            sim.apply(('ZPOW', (q,), t))


def tensor_matrix(bloq: Bloq) -> np.ndarray:
    """Qualtran 0.4's tensor simulation as a 2-D array [out, in]."""
    u = bloq.tensor_contract()
    n_in = sum(r.total_bits() for r in bloq.signature.lefts())
    n_out = sum(r.total_bits() for r in bloq.signature.rights())
    return np.asarray(u).reshape(2 ** n_out, 2 ** n_in)


def assert_close(a, b, atol=1e-10, what=''):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0
    if err > atol:
        raise AssertionError(f'{what}: max deviation {err:.3e}')
    return err


def bits_to_int(bits_lsb_first) -> int:
    return sum(int(b) << i for i, b in enumerate(bits_lsb_first))


def pyliqtr_view(bloq: Bloq):
    """The leaf counts of `bloq` as pyLIQTR's estimate sees them.  For a bloq without an explicit
    T-complexity annotation, pyLIQTR asks Qualtran's `_from_directly_countable` before its call
    graph, and that counts, for example, a gate on at most two qubits whose unitary is a Clifford as
    one Clifford, whatever its decomposition.  This walks `bloq` to its leaves, except that a
    sub-bloq which pyLIQTR counts directly, with a count that differs from its leaves, is not
    walked.  Returns the paper counts of the leaves walked and a list of (sub-bloq, pyLIQTR's
    TComplexity of it) for the sub-bloqs not walked, one entry per instance."""
    from qualtran.cirq_interop.t_complexity_protocol import (_from_directly_countable,
                                                             _from_explicit_annotation)
    from .stream import paper_counts, t_count, walk

    direct, seen = [], {}

    def directly_counted(sub):
        if sub not in seen:
            seen[sub] = None
            if not getattr(sub, 'measured', False) and _from_explicit_annotation(sub) is None:
                tc = _from_directly_countable(sub)
                if tc is not None:
                    c = paper_counts(BloqStream(sub))
                    if (t_count(c), c['RZ'] + c['RY']) != (tc.t, tc.rotations):
                        seen[sub] = tc
        return seen[sub]

    def hook(sub, inq):
        tc = directly_counted(sub)
        if tc is not None:                  # a cirq gate: every register is left and right
            direct.append((sub, tc))
            return inq
        out = yield from walk(sub, inq, ctx)
        return out

    ctx = StreamContext(compiled=False, sub_hook=hook)
    return paper_counts(BloqStream(bloq, ctx)), direct

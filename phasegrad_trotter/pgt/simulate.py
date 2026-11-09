"""Exact checks of a compiled layer.

Two simulators.

`check_layer` runs the circuit on every system basis state x with the gradient register in
|Psi_b> = 2^(-b/2) sum_y exp(-2 pi i y / 2^b) |y>.  Every operation maps a basis state to a phase
times a basis state (an AND is a Toffoli onto a fresh qubit, and its uncompute is exact on the
states where the target equals the AND, which is asserted), so the run carries the 2^b basis
states of the gradient register in parallel as numpy arrays, one boolean array per qubit.  The
check is that every ancilla returns to 0, x is unchanged, and the gradient register ends in
exp(2 pi i (f(x) + C) / 2^b) |Psi_b> exactly (to rounding).  This is the whole unitary on the
system and the catalyst, because the circuit is a phase times a permutation of basis states.

`statevector` is a plain state-vector simulation with the measurement-based uncompute done as a
measurement: an X-basis measurement of the target, a CZ on the controls if the outcome is 1, and
a reset.  It is for small circuits (up to about 22 qubits) and checks that the measured
uncompute is what `check_layer` assumes.
"""
from __future__ import annotations

import cmath
import math
from typing import Dict, Optional, Sequence

import numpy as np

from .circuit import Layer

_W8 = cmath.exp(1j * math.pi / 4)


def gradient_state(b: int) -> np.ndarray:
    y = np.arange(2 ** b)
    return np.exp(-2j * math.pi * y / 2 ** b) / math.sqrt(2 ** b)


def run_basis(layer: Layer, x: Sequence[int]) -> np.ndarray:
    """The gradient register's final state for the system in basis state x."""
    b = layer.b
    Y = 2 ** b
    y = np.arange(Y)
    bits: Dict[int, np.ndarray] = {}
    for j, q in enumerate(layer.system):
        bits[q] = np.full(Y, bool(x[j]))
    for k, q in enumerate(layer.gradient):
        bits[q] = ((y >> k) & 1).astype(bool)
    amp = gradient_state(b).astype(complex)
    for op in layer.ops:
        k = op[0]
        if k == 'alloc':
            assert op[1] not in bits, f'qubit {op[1]} allocated twice'
            bits[op[1]] = np.zeros(Y, dtype=bool)
        elif k == 'free':
            assert not bits[op[1]].any(), f'qubit {op[1]} freed dirty'
            del bits[op[1]]
        elif k == 'x':
            bits[op[1]] = ~bits[op[1]]
        elif k == 'cx':
            bits[op[2]] = bits[op[2]] ^ bits[op[1]]
        elif k == 'and':
            assert not bits[op[3]].any()
            bits[op[3]] = bits[op[1]] & bits[op[2]]
        elif k == 'anddg':
            assert np.array_equal(bits[op[3]], bits[op[1]] & bits[op[2]]), 'bad uncompute'
            bits[op[3]] = np.zeros(Y, dtype=bool)
        elif k == 'z':
            amp = np.where(bits[op[1]], -amp, amp)
        elif k == 'cz':
            amp = np.where(bits[op[1]] & bits[op[2]], -amp, amp)
        elif k == 's':
            amp = np.where(bits[op[1]], 1j * amp, amp)
        elif k == 't':
            amp = np.where(bits[op[1]], _W8 * amp, amp)
        elif k == 'tdg':
            amp = np.where(bits[op[1]], np.conj(_W8) * amp, amp)
        else:
            raise ValueError(op)
    extra = set(bits) - set(layer.system) - set(layer.gradient)
    assert not extra, f'ancillas left allocated: {sorted(extra)}'
    for j, q in enumerate(layer.system):
        assert (bits[q] == bool(x[j])).all(), 'system bit changed'
    yout = np.zeros(Y, dtype=np.int64)
    for kk, q in enumerate(layer.gradient):
        yout |= bits[q].astype(np.int64) << kk
    assert len(set(yout.tolist())) == Y, 'not a permutation'
    out = np.zeros(Y, dtype=complex)
    out[yout] = amp
    return out


def check_layer(layer: Layer, f, xs=None, atol: float = 1e-9) -> float:
    """Run every basis state (or those in xs) and return the largest deviation from
    exp(2 pi i (f(x) + C) / 2^b) |Psi_b>."""
    n = len(layer.system)
    b = layer.b
    psi = gradient_state(b)
    if xs is None:
        xs = [[(i >> j) & 1 for j in range(n)] for i in range(2 ** n)]
    worst = 0.0
    for x in xs:
        out = run_basis(layer, x)
        want = np.exp(2j * math.pi * (f(x) + layer.constant) / 2 ** b) * psi
        worst = max(worst, float(np.abs(out - want).max()))
    assert worst < atol, worst
    return worst


def statevector(layer: Layer, init: np.ndarray, rng: Optional[np.random.Generator] = None,
                n_qubits: Optional[int] = None) -> np.ndarray:
    """Run the circuit on the state init (system qubits, then gradient register) with every
    ancilla starting in |0>, the uncomputes done by measurement; returns the final state of the
    system and gradient register.  Qubit q is bit q of the basis index."""
    rng = rng or np.random.default_rng(0)
    nq = n_qubits or layer.peak
    ns = len(layer.system) + layer.b
    psi = np.zeros(2 ** nq, dtype=complex)
    psi[:2 ** ns] = init
    idx = np.arange(2 ** nq)

    def bit(q):
        return ((idx >> q) & 1).astype(bool)

    def flip(state, q, mask=None):
        j = idx ^ (1 << q)
        new = state[j]
        if mask is None:
            return new
        return np.where(mask[j], new, state)       # mask is evaluated on the source index

    for op in layer.ops:
        k = op[0]
        if k in ('alloc', 'free'):
            if k == 'free':
                assert np.abs(psi[bit(op[1])]).max() < 1e-9, 'freed qubit not |0>'
            continue
        if k == 'x':
            psi = flip(psi, op[1])
        elif k == 'cx':
            psi = flip(psi, op[2], bit(op[1]))
        elif k == 'and':
            psi = flip(psi, op[3], bit(op[1]) & bit(op[2]))
        elif k == 'anddg':
            a, bb, t = op[1:]
            # X-basis measurement of t: H on t, then a Z measurement
            tb = bit(t)
            # amplitude of outcome m on the t=0 branch after H: (psi[t=0] + (-1)^m psi[t=1]) / sqrt2
            s0 = np.where(~tb, (psi + psi[idx ^ (1 << t)]) / math.sqrt(2), 0)
            s1 = np.where(~tb, (psi - psi[idx ^ (1 << t)]) / math.sqrt(2), 0)
            pr1 = float(np.vdot(s1, s1).real)
            if rng.random() < pr1:
                psi = s1 / math.sqrt(pr1)
                psi = np.where(bit(a) & bit(bb), -psi, psi)      # the CZ fix-up
            else:
                psi = s0 / math.sqrt(1 - pr1)
        elif k == 'z':
            psi = np.where(bit(op[1]), -psi, psi)
        elif k == 'cz':
            psi = np.where(bit(op[1]) & bit(op[2]), -psi, psi)
        elif k == 's':
            psi = np.where(bit(op[1]), 1j * psi, psi)
        elif k == 't':
            psi = np.where(bit(op[1]), _W8 * psi, psi)
        elif k == 'tdg':
            psi = np.where(bit(op[1]), np.conj(_W8) * psi, psi)
        else:
            raise ValueError(op)
    assert np.abs(psi[2 ** ns:]).max() < 1e-9, 'ancillas not returned to |0>'
    return psi[:2 ** ns]

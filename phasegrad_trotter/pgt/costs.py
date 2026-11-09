"""Error budgets and costs: the phase-gradient layer against rotations synthesised one by one.

Errors are diamond distances, per rotation, as in the synthesis literature.  On the grid of
spacing Delta = pi / 2^b:

  nearest rounding     |theta' - theta| <= Delta / 2, a unitary error 2 sin|theta' - theta|,
                       so b_det(eps) is the least b with 2 sin(Delta / 2) <= eps;
  randomised rounding  the two neighbours mixed so that the coherent error cancels, leaving Z
                       dephasing of probability <= sin^2(Delta / 2) and an error 2 sin^2(Delta/2),
                       so b_rand(eps) is the least b with 2 sin^2(Delta / 2) <= eps.

Synthesis: gridsynth (Ross and Selinger's algorithm, pygridsynth 2.0) at operator-norm accuracy
eps / 2, which is diamond accuracy eps; and the mixed-fallback formula 0.53 log2(1/eps) + 4.86
expected T gates, the angle-independent Clifford+T figure that Bothe et al. (arXiv:2605.31544)
quote for the method of Kliuchnikov et al. (Quantum 7, 1208 (2023)).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .circuit import Circ, _Compute, compile_layer
from .digits import digits_mod
from . import bothe
from .digits import grid, mixing_probability, naf_weight, round_nearest, sparse_pair, wrap

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data')


def b_det(eps: float) -> int:
    b = 4
    while 2 * math.sin(grid(b) / 2) > eps:
        b += 1
    return b


def b_rand(eps: float) -> int:
    b = 4
    while 2 * math.sin(grid(b) / 2) ** 2 > eps:
        b += 1
    return b


def t_mixed_fallback(eps: float) -> float:
    return 0.53 * math.log2(1 / eps) + 4.86


def gridsynth_t(theta: float, eps: float) -> int:
    """T gates of gridsynth's approximation of exp(-i theta Z) = Rz(2 theta) to diamond eps."""
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates
    mpmath.mp.dps = max(30, int(3 * math.log10(1 / eps)) + 20)
    gates = gridsynth_gates(theta=mpmath.mpf(2 * theta), epsilon=mpmath.mpf(eps / 2))
    return sum(1 for g in str(gates) if g == 'T')


_GS: Optional[Dict] = None


def gridsynth_fit() -> Tuple[float, float]:
    """(a, c): the least-squares fit T = a log2(1/eps) + c to data/gridsynth.json."""
    global _GS
    if _GS is None:
        with open(os.path.join(DATA, 'gridsynth.json')) as fh:
            _GS = json.load(fh)
    return _GS['fit']['a'], _GS['fit']['c']


def t_gridsynth(eps: float) -> float:
    a, c = gridsynth_fit()
    return a * math.log2(1 / eps) + c


# --- choosing the integers -------------------------------------------------------------------

def choose_K(theta: float, eps: float, mode: str, rng: np.random.Generator) -> Tuple[int, int]:
    """(K, b) for one rotation exp(-i theta Z) at diamond error eps.

    mode 'det': nearest point on the b_det grid; 'rand': randomised rounding between the two
    neighbours on the b_rand grid; 'sparse': randomised rounding onto the sparse pair of
    digits.sparse_pair on the b_rand grid (dephasing <= eps / 2)."""
    if mode == 'det':
        b = b_det(eps)
        return round_nearest(theta, b), b
    b = b_rand(eps)
    if mode == 'rand':
        K_lo, K_hi, q = mixing_probability(theta, b)
    elif mode == 'sparse':
        K_lo, K_hi, q, _ = sparse_pair(theta, b, eps / 2)
    else:
        raise ValueError(mode)
    return (K_hi if rng.random() < q else K_lo), b


def _digits(K, n_members: int, b: int) -> int:
    """Signed digits a class contributes (an estimate for a shared-angle class: its digits times
    the bits of its Hamming weight)."""
    if isinstance(K, int):
        return len(digits_mod(K, b)) * (n_members.bit_length() if n_members > 1 else 1)
    return sum(len(digits_mod(k, b)) for k in K)


def batches(classes: Sequence, b: int, batch: Optional[int] = None,
            batch_digits: Optional[int] = None) -> List[list]:
    """Split a layer into batches: at most `batch` classes each, or, with `batch_digits`, as many
    classes as fit in that many signed digits (at least one), so that the ancillas, which grow
    with the digits, stay bounded while rotations with no digits cost nothing."""
    classes = list(classes)
    if batch_digits is None:
        batch = batch or max(1, len(classes))
        return [classes[i:i + batch] for i in range(0, len(classes), batch)]
    out, cur, n = [], [], 0
    for K, m in classes:
        d = _digits(K, len(m), b)
        if cur and n + d > batch_digits:
            out.append(cur)
            cur, n = [], 0
        cur.append((K, m))
        n += d
    if cur:
        out.append(cur)
    return out


def layer_cost(classes: Sequence, b: int, batch: Optional[int] = None,
               batch_digits: Optional[int] = None) -> Dict[str, int]:
    """Counts of compile_layer over the batches of `batches`.  Each batch is its own compute,
    gradient addition and uncompute."""
    out = {'and_compute': 0, 'and_gradient': 0, 't_direct': 0, 'peak_ancilla': 0, 'digits': 0,
           'batches': 0}
    for chunk in batches(classes, b, batch, batch_digits):
        qubits = sorted({q for _, m in chunk for q in m})
        remap = {q: j for j, q in enumerate(qubits)}
        lay = compile_layer([(K, [remap[q] for q in m]) for K, m in chunk], b, len(qubits))
        out['and_compute'] += lay.n_and_compute
        out['and_gradient'] += lay.n_and_gradient
        out['t_direct'] += lay.n_t_direct
        out['digits'] += lay.digits
        out['batches'] += 1
        out['peak_ancilla'] = max(out['peak_ancilla'], lay.peak - len(qubits) - b)
    out['t'] = 4 * (out['and_compute'] + out['and_gradient']) + out['t_direct']
    return out


def hw_ands(n: int) -> int:
    """ANDs to add n bits of equal weight into a binary number with the full and half adders of
    circuit.py: the Hamming-weight tree used both here and in the synthesis baseline."""
    c = Circ(n_fixed=n)
    comp = _Compute(c)
    comp.compress({0: list(range(n))}, last=10 ** 6)
    return comp.n_and


# --- the example: a Heisenberg chain in a random field -----------------------------------------

def heisenberg_layers(J: Sequence[float], h: Sequence[float], dt: float, classes: bool):
    """The commuting layers of one second-order step of
        H = sum_i J_i (XX + YY + ZZ)_{i,i+1} + sum_i h_i Z_i
    on an open chain of n = len(h) spins, with the half-steps of XX merged between steps:
    XX(dt), YY(dt/2), [ZZ + Z](dt), YY(dt/2).  Each layer is a list of (kind, theta, bits): with
    `classes` (all J_i equal) the bonds of a layer form one equal-angle class; otherwise every
    bond is its own rotation.  Bits 0..n-2 are the bond parities, n-1..2n-2 the spins."""
    n = len(h)
    bonds = list(range(n - 1))
    spins = list(range(n - 1, 2 * n - 1))

    def bond_layer(f):
        if classes:
            return [('class', J[0] * dt * f, bonds)]
        return [('single', J[i] * dt * f, [bonds[i]]) for i in range(n - 1)]

    field = [('single', h[i] * dt, [spins[i]]) for i in range(n)]
    return [bond_layer(1), bond_layer(0.5), bond_layer(1) + field, bond_layer(0.5)]


def expected_digits(theta: float, eps: float) -> float:
    """Expected NAF weight of the sparse-pair rounding of theta at error eps (b_rand grid)."""
    b = b_rand(eps)
    K_lo, K_hi, q, _ = sparse_pair(theta, b, eps / 2)
    return (1 - q) * naf_weight(wrap(K_lo, b)) + q * naf_weight(wrap(K_hi, b))


SYNTH = ('gridsynth', 'fallback', 'bothe', 'bothe_af')


def synth_cost(theta: float, eps: float, method: str) -> float:
    if method == 'gridsynth':
        return t_gridsynth(eps)
    if method == 'fallback':
        return t_mixed_fallback(eps)
    if method == 'bothe':
        return bothe.t_fallback(theta, eps)
    if method == 'bothe_af':
        return bothe.t_ancilla_free(theta, eps)
    raise ValueError(method)


def heisenberg_step_cost(J: Sequence[float], h: Sequence[float], dt: float, eps_step: float,
                         method: str, rng: np.random.Generator, batch: Optional[int] = None,
                         classes: bool = False, batch_digits: Optional[int] = None):
    """T gates of one step at total diamond error eps_step.

    The approximated rotations are the single rotations and, for an equal-angle class of n - 1
    bonds, one rotation per bit of its Hamming weight (m = bit length of n - 1), the bit of
    weight 2^k by 2^k theta; R of them in all, each given eps_step / R.

    method  'gridsynth', 'fallback' (mixed fallback, angle-independent), 'bothe' (the
            angle-dependent fallback formula of Bothe et al., Eq. 15) or 'bothe_af' (their
            ancilla-free formula, Eq. 12): every rotation synthesised, the classes summed first
            by the same Hamming-weight tree;
            'det', 'rand' or 'sparse': the phase-gradient layers of circuit.py, batches of at
            most `batch` rotations or classes;
            'hybrid': each rotation goes to whichever of 'bothe' and 'sparse' has the lower
            expected T count (decided before any rounding is drawn).
    """
    n = len(h)
    m = (n - 1).bit_length()
    layers = heisenberg_layers(J, h, dt, classes)
    R = sum(m if kind == 'class' else 1 for layer in layers for kind, _, _ in layer)
    eps = eps_step / R
    tree = 4 * hw_ands(n - 1)
    if method in SYNTH:
        t = 0.0
        for layer in layers:
            for kind, theta, bits in layer:
                if kind == 'class':
                    t += tree + sum(synth_cost((2 ** k) * theta, eps, method) for k in range(m))
                else:
                    t += synth_cost(theta, eps, method)
        return {'t': t, 'rotations': R, 'eps_rot': eps, 't_synth': t, 'peak_ancilla': 0}
    mode = 'sparse' if method == 'hybrid' else method
    b = b_det(eps) if mode == 'det' else b_rand(eps)

    def to_gradient(theta):
        if method != 'hybrid':
            return True
        return 4 * expected_digits(theta, eps) <= bothe.t_fallback(theta, eps)

    total = {'t': 0.0, 'and_compute': 0, 'and_gradient': 0, 't_direct': 0, 'peak_ancilla': 0,
             't_synth': 0.0}
    for layer in layers:
        cl = []
        for kind, theta, bits in layer:
            if kind == 'class':
                Ks = []
                for k in range(m):
                    a = (2 ** k) * theta
                    if to_gradient(a):
                        Ks.append(choose_K(a, eps, mode, rng)[0])
                    else:
                        Ks.append(0)
                        total['t_synth'] += bothe.t_fallback(a, eps)
                if any(Ks):
                    cl.append((Ks, bits))
                else:
                    total['t_synth'] += tree
            elif to_gradient(theta):
                cl.append((choose_K(theta, eps, mode, rng)[0], bits))
            else:
                total['t_synth'] += bothe.t_fallback(theta, eps)
        if cl:
            c = layer_cost(cl, b, batch, batch_digits)
            for key in ('t', 'and_compute', 'and_gradient', 't_direct'):
                total[key] += c[key]
            total['peak_ancilla'] = max(total['peak_ancilla'], c['peak_ancilla'])
    total['t'] += total['t_synth']
    total['rotations'] = R
    total['eps_rot'] = eps
    return total

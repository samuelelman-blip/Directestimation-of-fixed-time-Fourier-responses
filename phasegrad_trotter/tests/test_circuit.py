"""The compiled layer, simulated exactly, and its counts."""
import math

import numpy as np
import pytest

from pgt.circuit import compile_layer, compile_rotations
from pgt.digits import digits_mod
from pgt.simulate import check_layer, gradient_state, statevector


def _f_single(Ks):
    return lambda x: sum(K * xi for K, xi in zip(Ks, x))


@pytest.mark.parametrize('seed', range(60))
def test_rotations_exact(seed):
    rng = np.random.default_rng(seed)
    L = int(rng.integers(1, 8))
    b = int(rng.integers(4, 11))
    Ks = [int(k) for k in rng.integers(-2 ** (b + 1), 2 ** (b + 1), size=L)]
    lay = compile_rotations(Ks, b)
    check_layer(lay, _f_single(Ks))


@pytest.mark.parametrize('seed', range(40))
def test_classes_exact(seed):
    rng = np.random.default_rng(100 + seed)
    b = int(rng.integers(4, 9))
    n = int(rng.integers(2, 9))
    m = n.bit_length()
    per_bit = [int(k) for k in rng.integers(-2 ** (b - 1), 2 ** (b - 1), size=m)]
    K = int(rng.integers(-2 ** (b - 1), 2 ** (b - 1)))
    half = n // 2
    classes = [(K, list(range(half))), (per_bit, list(range(half, n)))]
    singles = [(int(rng.integers(-40, 40)), [j]) for j in range(n)]
    lay = compile_layer(classes + singles, b, n)

    def f(x):
        W1 = sum(x[:half])
        W2 = sum(x[half:])
        return (K * W1 + sum(per_bit[k] * ((W2 >> k) & 1) for k in range(m))
                + sum(k * x[g[0]] for k, g in singles))
    check_layer(lay, f)


@pytest.mark.parametrize('Ks,b', [([5, -3], 5), ([7, 2, -6], 5), ([11], 6), ([3, 3, 1], 4),
                                  ([9, -13], 6)])
def test_statevector_measured_uncompute(Ks, b):
    """A full state-vector run, the uncomputes done by X measurements with random outcomes."""
    lay = compile_rotations(Ks, b)
    L = len(Ks)
    rng = np.random.default_rng(7)
    a = rng.normal(size=2 ** L) + 1j * rng.normal(size=2 ** L)
    a /= np.linalg.norm(a)
    g = gradient_state(b)
    ph = np.array([np.exp(2j * math.pi * (sum(K * ((x >> j) & 1) for j, K in enumerate(Ks))
                                          + lay.constant) / 2 ** b) for x in range(2 ** L)])
    want = np.kron(g, a * ph)
    for seed in range(4):
        out = statevector(lay, np.kron(g, a), rng=np.random.default_rng(seed))
        assert np.abs(out - want).max() < 1e-12


@pytest.mark.parametrize('seed', range(30))
def test_count_bounds(seed):
    """ANDs of the compute part <= digits at positions <= b-4; of the addition <= b-2; T gates
    outside ANDs only on column b-3."""
    rng = np.random.default_rng(200 + seed)
    b = int(rng.integers(6, 40))
    L = int(rng.integers(1, 300))
    Ks = [int(k) for k in rng.integers(-2 ** (b - 1), 2 ** (b - 1), size=L)]
    lay = compile_rotations(Ks, b)
    digits = sum(1 for K in Ks for p, _ in digits_mod(K, b) if p <= b - 4)
    top = sum(1 for K in Ks for p, _ in digits_mod(K, b) if p == b - 3)
    assert lay.digits == digits
    assert lay.n_and_compute <= digits
    assert lay.n_and_gradient <= b - 2
    assert lay.t_count <= 4 * (digits + b - 2) + top + lay.n_and_compute
    assert lay.counts.get('and', 0) == lay.counts.get('anddg', 0) == lay.n_and


@pytest.mark.parametrize('b', [6, 10, 20, 32])
def test_single_rotation_cost(b):
    """One rotation: no compression; the addition costs b - 2 - p0 ANDs, p0 the lowest digit."""
    rng = np.random.default_rng(b)
    for _ in range(50):
        K = int(rng.integers(1, 2 ** (b - 3)))
        lay = compile_rotations([K], b)
        dg = [p for p, _ in digits_mod(K, b) if p <= b - 4]
        assert lay.n_and_compute == 0
        assert lay.n_and_gradient == (b - 2 - min(dg) if dg else 0)

"""The grid, the signed digits and the two roundings."""
import math

import numpy as np
import pytest

from pgt.digits import (_min_weight, dephasing, digits_mod, grid, mixing_probability, naf,
                        naf_weight, round_nearest, sparse_nearest, sparse_pair, wrap)


@pytest.mark.parametrize('K', list(range(-300, 300)) + [2 ** 40 + 12345, -(3 ** 25)])
def test_naf(K):
    d = naf(K)
    assert sum(s * 2 ** p for p, s in d) == K
    ps = [p for p, _ in d]
    assert all(b - a >= 2 for a, b in zip(ps, ps[1:])), 'adjacent nonzero digits'


def test_naf_is_lightest():
    # no signed-digit expansion of K is lighter than its NAF: compare with the interval search
    for lo in range(-60, 60):
        for hi in range(lo, lo + 10):
            w, K = _min_weight(lo, hi)
            assert lo <= K <= hi
            assert w == naf_weight(K) == min(naf_weight(k) for k in range(lo, hi + 1))


@pytest.mark.parametrize('b', [4, 5, 8, 13])
def test_digits_mod(b):
    for K in range(-2 ** (b + 1), 2 ** (b + 1)):
        s = sum(d * 2 ** p for p, d in digits_mod(K, b))
        assert (s - K) % 2 ** b == 0
        assert all(p < b for p, _ in digits_mod(K, b))
        assert -2 ** (b - 1) < wrap(K, b) <= 2 ** (b - 1)


def test_nearest():
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = int(rng.integers(4, 40))
        th = float(rng.uniform(-math.pi, math.pi))
        assert abs(round_nearest(th, b) * grid(b) - th) <= grid(b) / 2 * (1 + 1e-12)


def _channel(pairs):
    """Superoperator (on vec(rho), 2x2) of the mixture sum_k w_k exp(-i a_k Z) . exp(i a_k Z)."""
    S = np.zeros((4, 4), dtype=complex)
    for w, a in pairs:
        U = np.diag([np.exp(-1j * a), np.exp(1j * a)])
        S += w * np.kron(U, U.conj())
    return S


def test_mixing_is_rotation_then_dephasing():
    rng = np.random.default_rng(2)
    for _ in range(300):
        b = int(rng.integers(4, 30))
        th = float(rng.uniform(-1.5, 1.5))
        K0, K1, q = mixing_probability(th, b)
        D = grid(b)
        S = _channel([(1 - q, K0 * D), (q, K1 * D)])
        p = dephasing(th - K0 * D, K1 * D - th)
        U = np.diag([np.exp(-1j * th), np.exp(1j * th)])
        Zs = np.kron(np.diag([1, -1]), np.diag([1, -1]))
        want = np.kron(U, U.conj()) @ ((1 - p) * np.eye(4) + p * Zs)
        assert np.abs(S - want).max() < 1e-12
        assert 0 <= p <= math.sin(D / 2) ** 2 * (1 + 1e-9)


def test_sparse_pair():
    rng = np.random.default_rng(3)
    for eps in (1e-6, 1e-10):
        from pgt.costs import b_rand
        b = b_rand(eps)
        for _ in range(150):
            th = float(10 ** rng.uniform(-8, 0)) * float(rng.choice([-1, 1]))
            Kl, Kh, q, p = sparse_pair(th, b, eps / 2)
            D = grid(b)
            assert Kl * D <= th <= Kh * D
            assert p <= eps / 2 * (1 + 1e-9)
            S = _channel([(1 - q, Kl * D), (q, Kh * D)])
            U = np.diag([np.exp(-1j * th), np.exp(1j * th)])
            Zs = np.kron(np.diag([1, -1]), np.diag([1, -1]))
            assert np.abs(S - np.kron(U, U.conj()) @ ((1 - p) * np.eye(4) + p * Zs)).max() < 1e-12
            K0, K1, q0 = mixing_probability(th, b)
            adj = (1 - q0) * naf_weight(K0) + q0 * naf_weight(K1)
            assert (1 - q) * naf_weight(Kl) + q * naf_weight(Kh) <= adj + 1e-9


def test_sparse_nearest():
    rng = np.random.default_rng(4)
    for _ in range(100):
        b = int(rng.integers(10, 30))
        th = float(rng.uniform(-1, 1))
        tol = grid(b) * float(rng.uniform(0.5, 20))
        K = sparse_nearest(th, b, tol)
        assert abs(K * grid(b) - th) <= tol
        assert naf_weight(K) <= naf_weight(round_nearest(th, b))

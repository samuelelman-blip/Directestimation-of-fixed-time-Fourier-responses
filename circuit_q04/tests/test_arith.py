"""Exact gadgets, additions, comparisons and sorting in Qualtran 0.4, against their specifications."""
import itertools
import math

import numpy as np
import pytest
from qualtran import Adjoint, BloqBuilder

from ftfr_q04.arith import (BitonicSort, CAdd, Comparator, GreaterThan, HalfLessThan,
                            OutOfPlaceAdder, Square)
from ftfr_q04.gates import (BasisPhase, CHadamardExact, FanHadamard, RegisterCSwap, RouteToZero,
                            ZeroReflection)
from ftfr_q04.phasegrad import PhaseGradAdd, WindowAdd
from ftfr_q04.stream import BloqStream, count, leaf_report
from ftfr_q04.testing import assert_close, stream_basis_map, stream_matrix, tensor_matrix


def perm_matrix(n_in, n_out, fn):
    m = np.zeros((2 ** n_out, 2 ** n_in))
    for k in range(2 ** n_in):
        m[fn(k), k] = 1
    return m


def test_chadamard():
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    ch = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), h]])
    assert_close(stream_matrix(CHadamardExact()), ch, 1e-12, 'CH')
    c = count(BloqStream(CHadamardExact()))
    assert c['T'] + c['TDG'] == 2 and c['ZPOW'] == 0


def test_fan_hadamard():
    assert_close(stream_matrix(FanHadamard(2)), tensor_matrix(FanHadamard(2).decompose_bloq()),
                 1e-12, 'fan')


@pytest.mark.parametrize('n,vals', [(1, (0,)), (1, (1,)), (3, (0, 0, 0)), (3, (1, 0, 1)),
                                    (4, (0, 1, 1, 0))])
def test_basis_phase(n, vals):
    t = 0.37
    want = np.eye(2 ** n, dtype=complex)
    k = int(''.join(map(str, vals)), 2)
    want[k, k] = np.exp(1j * math.pi * t)
    assert_close(stream_matrix(BasisPhase(n, t, vals)), want, 1e-12, 'basis phase')
    c = count(BloqStream(BasisPhase(n, t, vals)))
    assert c['AND'] == max(n - 1, 0) and c['ZPOW'] == 1


@pytest.mark.parametrize('n', [1, 2, 3, 5])
def test_zero_reflection(n):
    want = np.eye(2 ** n)
    want[0, 0] = -1
    assert_close(stream_matrix(ZeroReflection(n)), want, 1e-12, 'reflection')
    assert count(BloqStream(ZeroReflection(n)))['AND'] == max(n - 2, 0)


@pytest.mark.parametrize('w', [1, 2, 3])
def test_register_cswap(w):
    def fn(k):
        c = k >> (2 * w)
        x = (k >> w) & (2 ** w - 1)
        y = k & (2 ** w - 1)
        if c:
            x, y = y, x
        return (c << 2 * w) | (x << w) | y
    assert_close(stream_matrix(RegisterCSwap(w)), perm_matrix(2 * w + 1, 2 * w + 1, fn), 1e-12,
                 'cswap')
    c = count(BloqStream(RegisterCSwap(w)))
    assert c['AND'] == w and c['TOFFOLI'] == 0


@pytest.mark.parametrize('sel_bits,r', [(1, 2), (2, 3), (2, 4)])
def test_route_to_zero(sel_bits, r):
    w = 2
    bloq = RouteToZero(sel_bits, w, r)
    for sel in range(r):
        for vals in itertools.product(range(2 ** w), repeat=r):
            k = sel
            for v in vals:
                k = (k << w) | v
            out, amp = stream_basis_map(bloq, k)
            assert abs(amp - 1) < 1e-12
            assert (out >> (w * (r - 1))) & (2 ** w - 1) == vals[sel]
            assert (out >> (r * w)) == sel
    assert count(BloqStream(bloq))['AND'] == (r - 1) * w
    bb = BloqBuilder()
    s = bb.add_register(bloq.signature[0])
    t = bb.add_register(bloq.signature[1])
    s, t = bb.add(bloq, sel=s, targets=t)
    s, t = bb.add(bloq.adjoint(), sel=s, targets=t)
    cb = bb.finalize(sel=s, targets=t)
    n = sel_bits + r * w
    for k in [0, 5, 2 ** n - 1, 37 % 2 ** n]:
        if (k >> (r * w)) < r:
            assert stream_basis_map(cb, k) == (k, 1)


@pytest.mark.parametrize('W,positions,cin', [(1, (0,), False), (2, (0,), True),
                                             (3, (0, 1, 2), False), (4, (0, 1), True),
                                             (4, (0, 2, 3), True), (5, (1, 2), False),
                                             (5, (0, 1, 2, 3, 4), True)])
def test_window_add(W, positions, cin):
    bloq = WindowAdd(W, positions, carry_in=cin)
    L = len(positions)
    n = L + W + int(cin)

    def fn(k):
        bits = [(k >> (n - 1 - i)) & 1 for i in range(n)]
        xv = bits[:L]
        wv = int(''.join(map(str, bits[L:L + W])), 2)
        cv = bits[L + W] if cin else 0
        add = sum(b << p for b, p in zip(xv, sorted(positions, reverse=True)))
        out_w = (wv + add + cv) % (2 ** W)
        out = k & ~(((1 << W) - 1) << int(cin))
        return out | (out_w << int(cin))

    assert_close(stream_matrix(bloq), perm_matrix(n, n, fn), 1e-12, 'window add')


def phase_gradient_prep(name, w):
    def prep(sim, ids):
        qs = list(np.ravel(ids[name]))          # big-endian: qs[m] has weight 2^(w-1-m)
        for m, q in enumerate(qs):
            sim.apply(('H', (q,), None))
            sim.apply(('ZPOW', (q,), -2 * 2 ** (w - 1 - m) / 2 ** w))
    return prep


def phase_gradient_vector(w):
    return np.exp(-2j * math.pi * np.arange(2 ** w) / 2 ** w) / 2 ** (w / 2)


@pytest.mark.parametrize('xb,w,s,ctrl,kz', [(3, 3, 0, False, ()), (2, 4, 0, False, ()),
                                            (2, 4, 1, False, ()), (3, 4, 1, True, ()),
                                            (3, 3, 0, True, ()), (4, 5, 1, True, (1,)),
                                            (1, 3, 0, True, ()), (2, 2, 0, False, ())])
def test_phase_grad_add(xb, w, s, ctrl, kz):
    bloq = PhaseGradAdd(xb, w, right_shift=s, controlled=ctrl, known_zero=kz)
    k = w - xb - s
    nc = int(ctrl)
    phi = phase_gradient_vector(w)
    for cx in range(2 ** (nc + xb)):
        x = cx & (2 ** xb - 1)
        c = (cx >> xb) if ctrl else 1
        if any((x >> j) & 1 for j in kz):
            continue
        got = stream_matrix(bloq, columns=[cx << w], prepare=phase_gradient_prep('phase_grad', w))[:, 0]
        want = np.zeros(2 ** (nc + xb + w), dtype=complex)
        want[cx << w:(cx + 1) << w] = phi * np.exp(2j * math.pi * c * x * 2 ** k / 2 ** w)
        assert_close(got, want, 1e-12, f'{bloq} x={x} c={c}')


def test_phase_grad_subtract():
    w, xb = 4, 3
    bloq = Adjoint(PhaseGradAdd(xb, w))
    phi = phase_gradient_vector(w)
    for x in range(2 ** xb):
        got = stream_matrix(bloq, columns=[x << w], prepare=phase_gradient_prep('phase_grad', w))[:, 0]
        want = np.zeros(2 ** (xb + w), dtype=complex)
        want[x << w:(x + 1) << w] = phi * np.exp(-2j * math.pi * x * 2 ** (w - xb) / 2 ** w)
        assert_close(got, want, 1e-12, 'subtract')


@pytest.mark.parametrize('b', [1, 2, 3, 4])
def test_square(b):
    m = stream_matrix(Square(b))
    want = np.zeros((2 ** (3 * b), 2 ** b))
    for a in range(2 ** b):
        want[(a << 2 * b) | a * a, a] = 1
    assert_close(m, want, 1e-12, 'square')
    assert count(BloqStream(Square(b)))['AND'] == b * b - b
    assert count(BloqStream(Adjoint(Square(b))))['ANDDG'] == b * b - b


@pytest.mark.parametrize('n,cv', [(2, 1), (3, 1), (3, 0)])
def test_cadd(n, cv):
    for c, a, b in itertools.product(range(2), range(2 ** n), range(2 ** n)):
        k = (c << 2 * n) | (a << n) | b
        out, amp = stream_basis_map(CAdd(n, cv), k)
        bb_ = (b + a) % 2 ** n if c == cv else b
        assert out == (c << 2 * n) | (a << n) | bb_ and abs(amp - 1) < 1e-12
        out, amp = stream_basis_map(Adjoint(CAdd(n, cv)), k)
        bb_ = (b - a) % 2 ** n if c == cv else b
        assert out == (c << 2 * n) | (a << n) | bb_ and abs(amp - 1) < 1e-12


@pytest.mark.parametrize('n', [2, 3])
def test_out_of_place_adder(n):
    for a, b in itertools.product(range(2 ** n), repeat=2):
        out, amp = stream_basis_map(OutOfPlaceAdder(n), (a << n) | b)
        assert out == (a << 2 * n) | (b << n) | ((a + b) % 2 ** n) and abs(amp - 1) < 1e-12


@pytest.mark.parametrize('n', [2, 3])
def test_half_less_than(n):
    for a, b in itertools.product(range(2 ** n), repeat=2):
        out, amp = stream_basis_map(HalfLessThan(n), (a << n) | b)
        c = (a - b) % 2 ** (n + 1)
        assert out == (a << (2 * n + 2)) | (b << (n + 2)) | (c << 1) | int(a < b)
        assert abs(amp - 1) < 1e-12
    # compute then measured uncompute is the identity (random X-measurement outcomes)
    bb = BloqBuilder()
    a = bb.add_register('a', n)
    b = bb.add_register('b', n)
    a, b, c, t = bb.add(HalfLessThan(n), a=a, b=b)
    a, b = bb.add(HalfLessThan(n, uncompute=True), a=a, b=b, c=c, target=t)
    cb = bb.finalize(a=a, b=b)
    for seed in range(3):
        m = stream_matrix(cb, measure_ands=True, seed=seed)
        assert_close(m, np.eye(2 ** (2 * n)), 1e-12, 'round trip')


def test_greater_than_and_comparator():
    n = 3
    for a, b in itertools.product(range(2 ** n), repeat=2):
        out, _ = stream_basis_map(GreaterThan(n, n), (a << (n + 1)) | (b << 1))
        assert out == (a << (n + 1)) | (b << 1) | int(a > b)
        out, _ = stream_basis_map(Comparator(n), (a << n) | b)
        lo, hi = min(a, b), max(a, b)
        assert out == (lo << (n + 1)) | (hi << 1) | int(a > b)


@pytest.mark.parametrize('k,n', [(2, 2), (4, 2)])
def test_bitonic_sort(k, n):
    bloq = BitonicSort(k, n)
    rng = np.random.default_rng(1)
    for _ in range(12):
        vals = list(rng.integers(0, 2 ** n, size=k))
        kin = 0
        for v in vals:
            kin = (kin << n) | int(v)
        out, amp = stream_basis_map(bloq, kin)
        keys = [(out >> (bloq.num_comparisons + n * (k - 1 - i))) & (2 ** n - 1) for i in range(k)]
        assert keys == sorted(vals) and abs(amp - 1) < 1e-12


@pytest.mark.parametrize('bloq', [CHadamardExact(), FanHadamard(3), BasisPhase(4, 0.3),
                                  ZeroReflection(5), RegisterCSwap(3), RouteToZero(2, 2, 3),
                                  WindowAdd(5, (0, 1, 2), minus_top=True), PhaseGradAdd(4, 6, controlled=True),
                                  Square(4), CAdd(3), HalfLessThan(3), Comparator(3), BitonicSort(4, 3)],
                         ids=str)
def test_leaves_only(bloq):
    """Each bloq decomposes, through decompose_bloq(), to the compiler's leaf gates only, and never
    to a Toffoli."""
    rep = leaf_report(bloq)
    assert rep['leaves']['TOFFOLI'] == 0

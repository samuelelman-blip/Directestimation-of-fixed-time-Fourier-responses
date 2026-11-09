"""Angles as integers on the phase-gradient grid, and their signed-digit expansions.

A b-bit phase-gradient register turns the addition of an integer K into the phase
exp(2 pi i K / 2^b).  The rotation exp(-i theta Z) on a bit x is exp(-i theta) exp(2 i theta x), so
it is the addition of K x with

    theta = pi K / 2^b,

that is, the Z angle lives on a grid of spacing Delta = pi / 2^b.  This module puts angles on that
grid (to the nearest point, or at random between the two neighbouring points so that the error
averages away to second order) and writes the integers in non-adjacent form, the signed-digit
expansion with the fewest nonzero digits.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np


def grid(b: int) -> float:
    """The Z-angle spacing Delta = pi / 2^b of a b-bit phase-gradient register."""
    return math.pi / 2 ** b


def wrap(K: int, b: int) -> int:
    """K mod 2^b as the representative in (-2^(b-1), 2^(b-1)]."""
    K %= 2 ** b
    return K - 2 ** b if K > 2 ** (b - 1) else K


def naf(K: int) -> List[Tuple[int, int]]:
    """The non-adjacent form of K: [(position, digit)], digit in {+1, -1}, no two adjacent."""
    out = []
    p = 0
    while K:
        if K & 1:
            d = 2 - (K & 3)            # +1 if K = 1 mod 4, -1 if K = 3 mod 4
            out.append((p, d))
            K -= d
        K >>= 1
        p += 1
    return out


def naf_weight(K: int) -> int:
    return len(naf(K))


def digits_mod(K: int, b: int) -> List[Tuple[int, int]]:
    """Signed digits of K mod 2^b, positions 0..b-1, from the NAF of its representative in
    (-2^(b-1), 2^(b-1)]; a digit at position b (weight 2^b) is a whole turn and is dropped."""
    return [(p, d) for p, d in naf(wrap(K, b)) if p < b]


def round_nearest(theta: float, b: int) -> int:
    """The grid point nearest theta: K = round(theta 2^b / pi)."""
    return int(round(theta / grid(b)))


def mixing_probability(theta: float, b: int) -> Tuple[int, int, float]:
    """The two grid points around theta and the probability of the upper one.

    With u = theta - K_lo Delta and v = K_hi Delta - theta, choosing K_hi with probability
    q = sin 2u / (sin 2u + sin 2v) makes the averaged channel exp(-i theta Z) followed by Z
    dephasing of probability sin u sin v / cos(u - v) <= sin^2(Delta / 2) (see errors.py).
    """
    D = grid(b)
    K_lo = math.floor(theta / D)
    u = theta - K_lo * D
    v = D - u
    if u <= 0.0:
        return K_lo, K_lo + 1, 0.0
    q = math.sin(2 * u) / (math.sin(2 * u) + math.sin(2 * v))
    return K_lo, K_lo + 1, q


def round_random(theta: float, b: int, rng: np.random.Generator) -> int:
    """One draw of the randomised rounding of mixing_probability."""
    K_lo, K_hi, q = mixing_probability(theta, b)
    return K_hi if rng.random() < q else K_lo


@lru_cache(maxsize=None)
def _min_weight(lo: int, hi: int) -> Tuple[int, int]:
    """(weight, K): the least NAF weight of an integer K in [lo, hi], and one such K."""
    if lo > hi:
        return (10 ** 9, 0)
    if lo <= 0 <= hi:
        return (0, 0)
    if hi < 0:
        w, K = _min_weight(-hi, -lo)
        return (w, -K)
    if lo <= 1:
        return (1, 1)                  # 0 is not in [lo, hi], so 1 is, and nothing is lighter
    best = (10 ** 9, 0)
    # K even: K = 2m
    w, m = _min_weight(-(-lo // 2), hi // 2)
    if w < best[0]:
        best = (w, 2 * m)
    # K odd: K = 2m + d with d = +1 or -1
    for d in (1, -1):
        w, m = _min_weight(-(-(lo - d) // 2), (hi - d) // 2)
        if w + 1 < best[0]:
            best = (w + 1, 2 * m + d)
    return best


def sparsest_in(lo: int, hi: int) -> int:
    """An integer in [lo, hi] of least NAF weight."""
    w, K = _min_weight(lo, hi)
    if w >= 10 ** 9:
        raise ValueError('empty interval')
    return K


def dephasing(u: float, v: float) -> float:
    """The Z-dephasing probability left by mixing exp(-i(theta - u)Z) and exp(-i(theta + v)Z)
    with the probability that cancels the coherent error: sin u sin v / cos(u - v)."""
    if u <= 0.0 or v <= 0.0:
        return 0.0
    return math.sin(u) * math.sin(v) / math.cos(u - v)


def _closest_below(t: int, w: int, floor_lo: int) -> int:
    """The largest K <= t with NAF weight <= w (K >= floor_lo is assumed to exist)."""
    lo, hi = floor_lo, t                       # weight(K in [lo, t]) <= w holds for lo = floor_lo
    while lo < hi:                             # find the largest lo with min weight on [lo, t] <= w
        mid = (lo + hi + 1) // 2
        if _min_weight(mid, t)[0] <= w:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _closest_above(t: int, w: int, ceil_hi: int) -> int:
    return -_closest_below(-t, w, -ceil_hi)


def _v_max(u: float, p_max: float) -> float:
    """The largest v in (0, pi/2 - u] with dephasing(u, v) <= p_max (dephasing grows with v)."""
    lo, hi = 0.0, math.pi / 2 - u
    if dephasing(u, hi) <= p_max:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dephasing(u, mid) <= p_max:
            lo = mid
        else:
            hi = mid
    return lo


def _floor_le(t: int, w: int, lo: int):
    """The largest K in [lo, t] with NAF weight <= w, or None."""
    if lo > t or _min_weight(lo, t)[0] > w:
        return None
    return _closest_below(t, w, lo)


def sparse_pair(theta: float, b: int, p_max: float) -> Tuple[int, int, float, float]:
    """Randomised rounding onto a sparse pair: grid points K_lo < theta/Delta < K_hi, and the
    probability q of K_hi, with dephasing <= p_max and small expected NAF weight
    (1 - q) w(K_lo) + q w(K_hi).  Returns (K_lo, K_hi, q, p).

    The candidates are, for every weight w, the nearest point of weight <= w on each side, and
    for each of those the farthest point of weight <= w on the other side that keeps the
    dephasing below p_max.  The adjacent pair is among them, so the result is never worse than
    mixing_probability when sin^2(Delta / 2) <= p_max.
    """
    D = grid(b)
    t = theta / D
    tl = math.floor(t)
    if tl == t:
        return tl, tl, 0.0, 0.0
    span = 2 ** b
    wmax = naf_weight(tl) + naf_weight(tl + 1) + 1
    lows = {_closest_below(tl, w, -span) for w in range(wmax + 1)}
    highs = {_closest_above(tl + 1, w, span) for w in range(wmax + 1)}
    pairs = set()
    for Kl in lows:
        u = theta - Kl * D
        if u >= math.pi / 2:
            continue
        hmax = math.floor((theta + _v_max(u, p_max)) / D)
        for w in range(wmax + 1):
            Kh = _floor_le(hmax, w, tl + 1)
            if Kh is not None:
                pairs.add((Kl, Kh))
    for Kh in highs:
        v = Kh * D - theta
        if v >= math.pi / 2:
            continue
        lmin = math.ceil((theta - _v_max(v, p_max)) / D)
        for w in range(wmax + 1):
            m = _floor_le(-lmin, w, -tl)
            if m is not None:
                pairs.add((-m, Kh))
    best = None
    for Kl, Kh in pairs:
        u, v = theta - Kl * D, Kh * D - theta
        if u + v > math.pi / 2:
            continue
        p = dephasing(u, v)
        if p > p_max * (1 + 1e-12):
            continue
        q = math.sin(2 * u) / (math.sin(2 * u) + math.sin(2 * v))
        cost = (1 - q) * naf_weight(wrap(Kl, b)) + q * naf_weight(wrap(Kh, b))
        key = (cost, Kh - Kl)
        if best is None or key < best[0]:
            best = (key, Kl, Kh, q, p)
    if best is None:
        raise ValueError('no pair meets p_max: refine the grid')
    return best[1], best[2], best[3], best[4]


def sparse_nearest(theta: float, b: int, tol: float) -> int:
    """The grid point of least NAF weight within tol of theta (in angle units)."""
    D = grid(b)
    lo = math.ceil((theta - tol) / D)
    hi = math.floor((theta + tol) / D)
    if lo > hi:
        raise ValueError('no grid point within tol: refine the grid')
    return sparse_in_wrapped(lo, hi, b)


def sparse_in_wrapped(lo: int, hi: int, b: int) -> int:
    return sparsest_in(lo, hi)

"""The error budgets and the cost model."""
import math

import numpy as np

from pgt.costs import (b_det, b_rand, gridsynth_fit, heisenberg_step_cost, hw_ands,
                       t_gridsynth)
from pgt.digits import grid


def test_budgets():
    for e in range(2, 16):
        eps = 10.0 ** -e
        b = b_det(eps)
        assert 2 * math.sin(grid(b) / 2) <= eps < 2 * math.sin(grid(b - 1) / 2)
        b = b_rand(eps)
        assert 2 * math.sin(grid(b) / 2) ** 2 <= eps < 2 * math.sin(grid(b - 1) / 2) ** 2


def test_hw_ands():
    # the tree adds n bits with n - (number of output bits) full adders plus some half adders
    for n in range(1, 200):
        a = hw_ands(n)
        assert n - n.bit_length() <= a <= n


def test_gridsynth_fit():
    a, c = gridsynth_fit()
    assert 2.9 < a < 3.1
    assert abs(t_gridsynth(1e-10) - (a * math.log2(1e10) + c)) < 1e-9


def test_heisenberg_runs():
    rng = np.random.default_rng(0)
    n = 9
    h = rng.uniform(-1, 1, n)
    J = rng.uniform(0.5, 1.5, n - 1)
    for method in ('gridsynth', 'fallback', 'bothe', 'bothe_af', 'det', 'rand', 'sparse',
                   'hybrid'):
        c = heisenberg_step_cost(J, h, 0.01, 1e-8, method, rng, batch=8)
        assert c['rotations'] == 4 * (n - 1) + n and c['t'] > 0
    c = heisenberg_step_cost([1.0] * (n - 1), h, 0.01, 1e-8, 'rand', rng, classes=True)
    assert c['rotations'] == 4 * (n - 1).bit_length() + n

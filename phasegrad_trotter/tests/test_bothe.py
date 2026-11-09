"""The transcribed cost formulas of Bothe et al. (arXiv:2605.31544)."""
import math

import numpy as np

from pgt import bothe


def test_tables():
    assert len(bothe._TANALPHA) == len(bothe._AVG_T) == len(bothe._PHI) == 56
    # the first two over-rotations of the staircase are S^dagger and T^dagger
    assert abs(bothe._PHI[0] - math.pi / 4) < 1e-15 and abs(bothe._PHI[1] - math.pi / 8) < 1e-15


def test_large_angles_return_to_kliuchnikov():
    for eps in (1e-6, 1e-10, 1e-14):
        for th in (0.1, 0.3, 1.0, 2.5):
            assert bothe.t_fallback(th, eps) == bothe.mixed_fallback(eps)
            assert bothe.t_ancilla_free(th, eps) == bothe.mixed_diagonal(eps)
    assert abs(bothe.mixed_fallback(1e-10) - (0.53 * math.log2(1e10) + 4.86)) < 1e-12


def test_small_angles_cost_little():
    rng = np.random.default_rng(0)
    for eps in (1e-6, 1e-10):
        for th in rng.uniform(0.01, 0.1, 50) * math.sqrt(eps):
            assert bothe.t_fallback(th, eps) < 1
            assert bothe.t_fallback(th, eps) <= bothe.t_ancilla_free(th, eps)
            # of order theta^2 / eps times a logarithm (their Eq. 66)
            assert bothe.t_ancilla_free(th, eps) < 4 * math.log2(1 / eps) * th ** 2 / eps


def test_symmetries():
    for th in (1e-4, 3e-3, 0.05):
        for eps in (1e-8, 1e-12):
            a = bothe.t_fallback(th, eps)
            assert bothe.t_fallback(-th, eps) == a
            assert abs(bothe.t_fallback(th + math.pi / 4, eps) - a) < 1e-9
    assert bothe.t_fallback(0.0, 1e-8) == 0.0

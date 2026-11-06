"""The history preparation P and its parts, simulated from their leaf gates, against their
specifications; P then P-dagger is the identity; at the parameters of the nine n = 2^20 cells P and
P-dagger have exactly the gate counts and peak qubit counts of the reference implementation."""
import itertools
import json
import math
import os

import numpy as np
import pytest
from qualtran import Adjoint, BloqBuilder

from ftfr_q04.history import (AliasSampler, HistoryPrep, KeySort, StepFlags, TimeRegister,
                              make_selector, selector_weights, sort_size, time_angles)
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts
from ftfr_q04.testing import assert_close, output_distribution, stream_basis_map, stream_matrix

ETA = 0.125 * (1 - math.exp(-1.0))
CELLS = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_cells.json')))


def alias_probabilities(sampler: AliasSampler) -> np.ndarray:
    """p_j = (1/N) [keep_j / 2^mu + sum over l with alt_l = j of (1 - keep_l / 2^mu)]."""
    N, mu = sampler.N, sampler.mu
    p = np.zeros(N)
    for l in range(N):
        kept = sampler.keep[l] / 2 ** mu
        p[l] += kept / N
        p[sampler.alt[l]] += (1 - kept) / N
    return p


def time_weights(b_t, mu_T=1.0):
    w = np.ones(2 ** b_t)
    i = np.arange(2 ** b_t)
    for k, th in enumerate(time_angles(b_t, mu_T)):
        w *= np.where((i >> k) & 1, math.sin(th / 2) ** 2, math.cos(th / 2) ** 2)
    return w


@pytest.mark.parametrize('M,prec', [(1, 0.05), (2, 0.05), (3, 0.01), (4, 0.02)])
def test_alias_sampler_distribution(M, prec):
    """The selection register ends with the alias tables' probabilities, which are within the
    requested precision of the target weights q_m."""
    sel = make_selector(M, ETA, prec)
    dist = output_distribution(sel, ['selection'])
    got = np.array([dist.get((j,), 0.0) for j in range(sel.N)])
    assert_close(got, alias_probabilities(sel), 1e-12, 'alias probabilities')
    assert np.max(np.abs(got - selector_weights(M, ETA))) <= prec


def test_alias_sampler_with_alternates():
    """Tables with non-trivial alternates (the selector's are all 0)."""
    sampler = AliasSampler.from_probabilities([1, 3, 2, 5, 4], precision=0.01)
    assert len(set(sampler.alt)) > 1
    dist = output_distribution(sampler, ['selection'], seed=3)
    got = np.array([dist.get((j,), 0.0) for j in range(5)])
    assert_close(got, alias_probabilities(sampler), 1e-12, 'alias probabilities')
    assert np.max(np.abs(got - np.array([1, 3, 2, 5, 4]) / 15)) <= 0.01


@pytest.mark.parametrize('s,M', [(1, 1), (2, 2), (2, 3), (3, 4)])
def test_step_flags(s, M):
    for m in range(2 ** s):
        k = m << M
        out, amp = stream_basis_map(StepFlags(s, M), k)
        flags = [int(m >= r) for r in range(1, M + 1)]
        want = m
        for f in flags:
            want = (want << 1) | f
        assert out == want and abs(amp - 1) < 1e-12


@pytest.mark.parametrize('b_t', [1, 2, 3])
def test_time_register(b_t):
    """Product of R_y(theta_k) with tan(theta_k/2) = exp(-mu Delta 2^(k-1)): p_i proportional to
    exp(-mu t_i), t_i = (i + 1/2) T / 2^b_t."""
    v = stream_matrix(TimeRegister(b_t), columns=[0])[:, 0]
    assert np.max(np.abs(v.imag)) < 1e-14 and np.all(v.real > 0)
    p = np.abs(v) ** 2
    t = (np.arange(2 ** b_t) + 0.5) / 2 ** b_t
    assert_close(p, np.exp(-t) / np.exp(-t).sum(), 1e-12, 'time weights')


@pytest.mark.parametrize('M,b_t', [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1)])
def test_key_sort(M, b_t):
    """Step r ends with the r-th largest key (flag, time), flag most significant."""
    rng = np.random.default_rng(M * 10 + b_t)
    bloq = KeySort(M, b_t)
    n_rec = bloq.signature.get_right('record').shape[0]
    for _ in range(10):
        f = rng.integers(0, 2, size=M)
        t = rng.integers(0, 2 ** b_t, size=M)
        k = 0
        for x in f:
            k = (k << 1) | int(x)
        for x in t:
            k = (k << b_t) | int(x)
        out, amp = stream_basis_map(bloq, k)
        keys = sorted(((int(a) << b_t) | int(b) for a, b in zip(f, t)), reverse=True)
        out >>= n_rec
        t_out = [(out >> (b_t * (M - 1 - r))) & (2 ** b_t - 1) for r in range(M)]
        f_out = [(out >> (b_t * M + M - 1 - r)) & 1 for r in range(M)]
        assert [(a << b_t) | b for a, b in zip(f_out, t_out)] == keys and abs(amp - 1) < 1e-12


def classical_history_distribution(P: HistoryPrep):
    """The distribution of (sel, f, t, parents) that P should prepare: m from the alias tables,
    f_r = [m >= r], M independent times, the steps sorted on (f, t) in descending order, parent
    p_r uniform on 0..r-1."""
    M, b_t = P.M, P.b_t
    p_sel = alias_probabilities(P.selector)
    tw = time_weights(b_t, P.mu_T)
    parent_ranges = [range(r) for r in range(2, M + 1)]
    p_par = 1.0 / math.prod(range(2, M + 1))
    dist = {}
    for m in range(M + 1):
        flags = [int(m >= r) for r in range(1, M + 1)]
        for ts in itertools.product(range(2 ** b_t), repeat=M):
            p = p_sel[m] * math.prod(tw[t] for t in ts)
            keys = sorted(((f << b_t) | t for f, t in zip(flags, ts)), reverse=True)
            f_out = tuple(key >> b_t for key in keys)
            t_out = tuple(key & (2 ** b_t - 1) for key in keys)
            for par in itertools.product(*parent_ranges):
                key = (m, f_out, t_out) + tuple(par)
                dist[key] = dist.get(key, 0.0) + p * p_par
    return dist


@pytest.mark.parametrize('M,b_t', [(1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1)])
def test_history_prep_distribution(M, b_t):
    P = HistoryPrep(M, b_t, ETA, 0.05)
    names = ['sel', 'f', 't'] + [f'p{r}' for r in range(2, M + 1)]
    got = output_distribution(P, names, seed=M + b_t)
    want = classical_history_distribution(P)
    keys = set(got) | set(want)
    err = max(abs(got.get(k, 0.0) - want.get(k, 0.0)) for k in keys)
    assert err < 1e-12, err


@pytest.mark.parametrize('M,b_t', [(1, 2), (2, 2), (3, 1), (4, 1)])
def test_prep_then_inverse_is_identity(M, b_t):
    """P then P-dagger returns |0...0> exactly, with the uncomputes measured (random outcomes)."""
    P = HistoryPrep(M, b_t, ETA, 0.05)
    bb = BloqBuilder()
    soqs = {r.name: bb.add_register(r) for r in P.history_registers()}
    out = bb.add_d(P, **soqs)
    out = bb.add_d(Adjoint(P), **out)
    cb = bb.finalize(**out)
    for seed in range(3):
        out, amp = stream_basis_map(cb, 0, seed=seed, measure_ands=True)
        assert out == 0 and abs(amp - 1) < 1e-12


@pytest.mark.parametrize('label', sorted(CELLS), ids=str)
def test_counts_equal_reference_at_cells(label):
    """P and P-dagger at the parameters of an n = 2^20 cell (P does not depend on n): the
    reference implementation's alias tables, gate counts and peak qubit counts; the peak is the same in
    Qualtran 0.4's decomposition order."""
    c = CELLS[label]
    P = HistoryPrep(c['M'], c['b_t'], c['eta'], c['sel_precision'])
    a = c['alias']
    assert (list(P.selector.alt), list(P.selector.keep), P.selector.mu) == (a['alt'], a['keep'],
                                                                            a['mu'])
    for part, bloq in (('P', P), ('Pdg', Adjoint(P))):
        ctx = StreamContext()
        got = {k: v for k, v in paper_counts(BloqStream(bloq, ctx)).items()
               if k not in ('ALLOC', 'FREE')}
        assert got == c['parts'][part]['counts'], part
        assert ctx.alloc.peak == c['parts'][part]['peak'], part
        ctx04 = StreamContext(greedy=False)
        for _ in BloqStream(bloq, ctx04):
            pass
        assert ctx04.alloc.peak == ctx.alloc.peak, part

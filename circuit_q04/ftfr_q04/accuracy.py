"""The accuracy of the estimator: the problem instances at any n, their exact responses, the error
bounds of the paper and of the cost model, and the selector's realised weights.

Used by tools/error_checks.py (which writes data/error_checks.json for tools/plot_errors.py) and
checked in tests/test_accuracy.py against the reference simulator.

The instances are those of the paper's resource estimates with K = 1, mu = T = 1, r = 1/8:
lambda_j = -mu + 2 pi i mu j / n, z0 = 1/sqrt(n), d_j = e^{-2 pi i j / n}/sqrt(n), and
G1 = (r mu / nu_n) D (J_n/sqrt(n) + P_n) (dense skeleton) or G1 = r mu D P_n (sparse skeleton),
with D = diag(w_j e^{i psi_j}) the cell's row data (w = 1 and psi = 0 for the dense and sparse
families; the closed-form profiles for PD-F and PS-F; those profiles rounded to the table widths
for PD-T and PS-T, as the circuit's tables hold them; the paper's tabulated profiles are random,
which changes the data but neither the circuit nor its error budget).  G1 is applied in O(n), so the exact
response and the truncated responses can be integrated at n = 2^20.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .access import profile_psi, profile_w, table_columns
from .history import make_selector, selector_weights


@dataclass
class Instance:
    family: str            # 'dense', 'sparse', 'PD' or 'PS'
    access: str            # 'flat', 'F' or 'T'
    b: int
    w: np.ndarray
    psi: np.ndarray
    mu: float = 1.0
    T: float = 1.0
    r: float = 1 / 8
    n: int = field(init=False)

    def __post_init__(self):
        self.n = 2 ** self.b
        j = np.arange(self.n)
        self.lam = -self.mu + 2j * np.pi * self.mu * j / self.n
        self.z0 = np.ones(self.n, dtype=complex) / math.sqrt(self.n)
        self.d = np.exp(-2j * np.pi * j / self.n) / math.sqrt(self.n)
        self.D = self.w * np.exp(1j * self.psi)

    @property
    def skeleton(self) -> str:
        return 'dense' if self.family in ('dense', 'PD') else 'sparse'

    @property
    def nu(self) -> float:
        return math.sqrt(2 + 2 / math.sqrt(self.n))

    @property
    def gamma(self) -> float:
        """The largest Euclidean row norm of G1: r mu max_j w_j (each skeleton row has norm nu_n,
        or 1 for the sparse skeleton)."""
        return self.r * self.mu * float(np.max(self.w))

    @property
    def gamma1(self) -> float:
        """The largest row 1-norm of G1 (for the (inf, 1) certificate of the truncation)."""
        if self.skeleton == 'dense':
            return (self.r * self.mu / self.nu) * (math.sqrt(self.n) + 1) * float(np.max(self.w))
        return self.r * self.mu * float(np.max(self.w))

    @property
    def L(self) -> float:
        """Lipschitz constant of D(t), max_j |lambda_j + mu| (Appendix A)."""
        return 2 * math.pi * self.mu * (self.n - 1) / self.n

    @property
    def tau(self) -> float:
        return (1 - math.exp(-self.mu * self.T)) / self.mu

    def G(self, z: np.ndarray) -> np.ndarray:
        """G1 z in O(n); (P z)_j = z_{j-1}."""
        if self.skeleton == 'dense':
            s = z.sum() / math.sqrt(self.n)
            return (self.r * self.mu / self.nu) * self.D * (s + np.roll(z, 1))
        return self.r * self.mu * self.D * np.roll(z, 1)

    def GT(self, v: np.ndarray) -> np.ndarray:
        """G1^T v (transpose, not adjoint); (P^T x)_l = x_{l+1}."""
        x = self.D * v
        if self.skeleton == 'dense':
            return (self.r * self.mu / self.nu) * (x.sum() / math.sqrt(self.n) + np.roll(x, -1))
        return self.r * self.mu * np.roll(x, -1)

    def dense_G1(self) -> np.ndarray:
        """G1 as a matrix (small n only)."""
        n = self.n
        P = np.zeros((n, n))
        P[(np.arange(n) + 1) % n, np.arange(n)] = 1.0
        if self.skeleton == 'dense':
            S = np.ones((n, n)) / math.sqrt(n) + P
            return (self.r * self.mu / self.nu) * self.D[:, None] * S
        return self.r * self.mu * self.D[:, None] * P


def instance(family: str, access: str, b: int, bv: int = 0, bpsi: int = 0) -> Instance:
    n = 2 ** b
    if access == 'flat':
        w, psi = np.ones(n), np.zeros(n)
    elif access == 'F':
        w = profile_w(b)
        psi = profile_psi(b) if family == 'PD' else np.zeros(n)
    elif access == 'T':
        cols = table_columns(b, bv, bpsi if family == 'PD' else None)
        w = np.array(cols[0], dtype=float) / 2 ** bv
        psi = (2 * math.pi * np.array(cols[1], dtype=float) / 2 ** bpsi if family == 'PD'
               else np.zeros(n))
    else:
        raise ValueError(access)
    return Instance(family, access, b, w, psi)


def instance_for(cell, b: Optional[int] = None) -> Instance:
    """The instance a cell (ftfr_q04.block.Cell) block-encodes, at its own b or at another."""
    return instance(cell.family, cell.access, cell.b if b is None else b, cell.bv, cell.bpsi)


# ------------------------------------------------------------------------------------------------
# exact responses
# ------------------------------------------------------------------------------------------------

def orders(inst: Instance, M: int, rtol: float = 1e-13, atol: float = 1e-15) -> np.ndarray:
    """h_m = d . z^[m](T) for m = 0..M: the interaction-picture hierarchy, integrated with DOP853.
    z^[0](t) = e^{lambda t} z0, dz^[m]/dt = lambda z^[m] + i sum_{a+b=m-1} z^[a] o G1 z^[b]."""
    lam, z0, n, T = inst.lam, inst.z0, inst.n, inst.T
    out = [complex(inst.d @ (np.exp(lam * T) * z0))]
    if M == 0:
        return np.array(out)

    def rhs(t, y):
        Z = y.reshape(M, n)
        levels = [np.exp(lam * t) * z0] + [Z[k] for k in range(M)]
        GZ = [inst.G(v) for v in levels]
        res = np.empty_like(Z)
        for m in range(1, M + 1):
            acc = lam * levels[m]
            for a in range(m):
                acc = acc + 1j * levels[a] * GZ[m - 1 - a]
            res[m - 1] = acc
        return res.ravel()

    sol = solve_ivp(rhs, (0.0, T), np.zeros(M * n, dtype=complex), method='DOP853', rtol=rtol,
                    atol=atol, t_eval=[T])            # keep only the final state (n = 2^20)
    if not sol.success:
        raise RuntimeError(sol.message)
    Z = sol.y[:, -1].reshape(M, n)
    out += [complex(inst.d @ Z[k]) for k in range(M)]
    return np.array(out)


def response(inst: Instance, rtol: float = 1e-13, atol: float = 1e-15) -> complex:
    """g(u(T)) = d . z(T) for the full quadratic equation dz/dt = lambda z + i z o G1 z."""
    sol = solve_ivp(lambda t, z: inst.lam * z + 1j * z * inst.G(z), (0.0, inst.T), inst.z0,
                    method='DOP853', rtol=rtol, atol=atol, t_eval=[inst.T])
    if not sol.success:
        raise RuntimeError(sol.message)
    return complex(inst.d @ sol.y[:, -1])


def _cell_sum(s: np.ndarray, b_t: int, mu: float, T: float, tau: float) -> np.ndarray:
    """sum_c mass_c e^{s t_c} over the 2^b_t cells of [0, T] (midpoints t_c, exact masses
    mass_c = 2 sinh(mu Delta/2) e^{-mu t_c} / (mu tau)), as a geometric series."""
    N = 2 ** b_t
    Delta = T / N
    x = s - mu
    pref = 2 * math.sinh(mu * Delta / 2) / (mu * tau)
    return pref * np.exp(x * Delta / 2) * np.expm1(x * N * Delta) / np.expm1(x * Delta)


def _continuum_sum(s: np.ndarray, mu: float, T: float, tau: float) -> np.ndarray:
    """int_0^T (e^{-mu t}/tau) e^{s t} dt."""
    x = s - mu
    return np.expm1(x * T) / (x * tau)


def order1(inst: Instance, b_t: Optional[int] = None) -> complex:
    """h_1 as the circuit computes it with a b_t-bit time grid (b_t = None: the continuum limit,
    which equals the hierarchy's h_1).  h_1 = i W_1 sum_l a_l int (e^{-mu t}/tau) e^{(lambda_l+mu)t}
    dt, with a = u0 o (G1/gamma)^T root, root = d o u0 o e^{(lambda+mu)T} and u0 = z0 (rho = 1);
    on the grid the time integral becomes the sum over cells with their exact masses."""
    shift = inst.lam + inst.mu
    u0 = inst.z0
    root = inst.d * u0 * np.exp(shift * inst.T)
    a = u0 * inst.GT(root) / inst.gamma
    tau = inst.tau
    S = (_continuum_sum(shift, inst.mu, inst.T, tau) if b_t is None
         else _cell_sum(shift, b_t, inst.mu, inst.T, tau))
    W1 = math.exp(-inst.mu * inst.T) * inst.gamma * tau
    return complex(1j * W1 * np.sum(a * S))


# ------------------------------------------------------------------------------------------------
# normalisation and error bounds
# ------------------------------------------------------------------------------------------------

def weights(inst: Instance, M: int) -> Tuple[np.ndarray, float]:
    """W_m = e^{-mu T} rho^{1+m} (gamma tau)^m for m = 0..M (rho = C1 = 1) and Xi_M = sum W_m."""
    W = np.array([math.exp(-inst.mu * inst.T) * (inst.gamma * inst.tau) ** m
                  for m in range(M + 1)])
    return W, float(W.sum())


def tail_bounds(inst: Instance, M: int) -> Tuple[float, float]:
    """Theorem 1's truncation bound E_M, Euclidean (2, 2) and conjugate (inf, 1):
    e^{-mu T} eta^{M+1}/(1 - eta) with eta = gamma tau or gamma_1 tau / sqrt(n) (both prefactors
    are 1 for these instances)."""
    pre = math.exp(-inst.mu * inst.T)
    e22 = inst.gamma * inst.tau
    e1 = inst.gamma1 * inst.tau / math.sqrt(inst.n)
    return pre * e22 ** (M + 1) / (1 - e22), pre * e1 ** (M + 1) / (1 - e1)


def truncation_bound(inst: Instance, M: int) -> float:
    """The certified truncation error: the better of the two certificates (as the cost model)."""
    return min(tail_bounds(inst, M))


def mean_order(inst: Instance, M: int) -> float:
    W, _ = weights(inst, M)
    return float(np.dot(np.arange(M + 1), W) / W.sum())


def time_bound(inst: Instance, M: int, b_t: int, worst: bool = False) -> float:
    """Appendix A at the answer level: m L T 2^{-(b_t+1)} per order-m block, averaged with the
    selector weights (mean order, as the cost model) or with every block at order M (worst),
    times Xi_M."""
    if M == 0:
        return 0.0
    _, Xi = weights(inst, M)
    m = M if worst else mean_order(inst, M)
    return m * inst.L * inst.T * 2.0 ** (-(b_t + 1)) * Xi


def selector_realised(M: int, eta: float, precision: float) -> np.ndarray:
    """q~_m, the probabilities the alias sampler of the circuit prepares: a uniform index l over
    N = M + 1 outcomes is kept if keep_l > sigma for sigma uniform on 0..2^mu - 1, otherwise
    replaced by alt_l (ftfr_q04.history.AliasSampler)."""
    sel = make_selector(M, eta, precision)
    N, scale = sel.N, 2.0 ** sel.mu
    q = np.zeros(N)
    for l in range(N):
        q[l] += sel.keep[l] / scale / N
        q[sel.alt[l]] += (scale - sel.keep[l]) / scale / N
    return q


def selector_bound(Xi: float, M: int, precision: float) -> float:
    """Each realised weight is within `precision` of its target and each normalised history block
    has modulus at most 1, so the answer moves by at most Xi_M (M + 1) precision."""
    return Xi * (M + 1) * precision


def eta_of(inst: Instance) -> float:
    """eta = gamma rho tau, the ratio of consecutive selector weights."""
    return inst.gamma * inst.tau


def estimate(h: np.ndarray, q_real: np.ndarray, q_exact: np.ndarray) -> complex:
    """The scalar the circuit's amplitude encodes, Xi_M <0|W|0>: each order h_m reweighted by the
    realised selector weight, sum_m (q~_m / q_m) h_m."""
    return complex(np.sum(np.asarray(q_real) / np.asarray(q_exact) * np.asarray(h)))


__all__ = ['Instance', 'instance', 'instance_for', 'orders', 'response', 'order1', 'weights',
           'tail_bounds', 'truncation_bound', 'mean_order', 'time_bound', 'selector_realised',
           'selector_bound', 'eta_of', 'estimate', 'selector_weights']

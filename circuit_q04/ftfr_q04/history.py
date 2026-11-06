"""The history preparation P: selector, step flags, time registers, sort, parents (Qualtran 0.4).

The same bloqs, in the same order, as the reference implementation (../circuit/ftfr/history.py).  The
selector is our own `AliasSampler`, with the structure of Qualtran's StatePreparationAliasSampling
but hashable and with its controlled swap written as one AND per qubit pair; Qualtran 0.4's QROM,
LessThanEqual, LessThanConstant and PrepareUniformSuperposition are used as they are.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Dict, List, Sequence, Tuple

import numpy as np
from attrs import field, frozen

from qualtran import (Bloq, BloqBuilder, BoundedQUInt, QAny, QBit, QUInt, Register, Side, Signature,
                      SoquetT)
from qualtran.bloqs.arithmetic import LessThanConstant, LessThanEqual
from qualtran.bloqs.basic_gates import Hadamard, OnEach, XGate
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from .arith import BitonicSort
from .gates import RegisterCSwap, nbits
from .wires import Wires


def time_angles(b_t: int, mu_T: float = 1.0) -> List[float]:
    """theta_k for the time qubit of weight 2^k: tan(theta_k / 2) = exp(-mu Delta 2^(k-1))."""
    mu_delta = mu_T / 2 ** b_t
    return [2 * math.atan(math.exp(-mu_delta * 2.0 ** (k - 1))) for k in range(b_t)]


def selector_weights(M: int, eta: float) -> np.ndarray:
    """q_m = eta^m / sum_{m' <= M} eta^m' (K = 1)."""
    w = np.array([eta ** m for m in range(M + 1)])
    return w / w.sum()


def sort_size(M: int) -> int:
    """Bitonic networks need a power of two: k = 2^ceil(log2 M) keys (1 for M <= 1)."""
    return 1 if M <= 1 else 1 << (M - 1).bit_length()


def n_comparators(M: int) -> int:
    k = sort_size(M)
    if k <= 1:
        return 0
    lg = k.bit_length() - 1
    return (k // 2) * lg * (lg + 1) // 2


def parent_dtype(r: int):
    return QBit() if nbits(r) == 1 else QAny(nbits(r))


def _reg_dtype(n: int):
    return QBit() if n == 1 else QAny(n)


# ------------------------------------------------------------------------------------------------
# the selector
# ------------------------------------------------------------------------------------------------

@frozen
class AliasSampler(Bloq):
    r"""Coherent alias sampling of $N$ outcomes (Babbush et al. 2018, Sec. III.D, Fig. 11):
    $|0\rangle\mapsto\sum_\ell\sqrt{p_\ell}|\ell\rangle|\mathrm{junk}_\ell\rangle$ with
    $p_\ell$ the $\mu$-bit approximations of the alias tables `alt`, `keep`.

    In the order of Qualtran's StatePreparationAliasSampling: a uniform superposition over $N$ on
    `selection`; $H^{\otimes\mu}$ on `sigma_mu`; Qualtran's QROM loading `alt` and `keep` addressed
    by `selection`; `less_than_equal` $\mathrel{\oplus}=[\mathit{keep}\le\sigma]$ (Qualtran's
    LessThanEqual); `alt` and `selection` swapped if `less_than_equal`, one AND per qubit pair.
    The tables are tuples, so the bloq is hashable."""
    N: int
    alt: Tuple[int, ...] = field(converter=lambda v: tuple(int(x) for x in v))
    keep: Tuple[int, ...] = field(converter=lambda v: tuple(int(x) for x in v))
    mu: int

    @classmethod
    def from_probabilities(cls, probabilities: Sequence[float], precision: float) -> 'AliasSampler':
        """Tables for unnormalised probabilities w_l, each represented to within `precision` (as
        the reference implementation's StatePreparationAliasSampling.from_probabilities)."""
        total = float(sum(probabilities))
        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            [w / total for w in probabilities], epsilon=precision / total)
        return cls(len(probabilities), alt, keep, int(mu))

    @cached_property
    def s(self) -> int:
        return max(1, (self.N - 1).bit_length())

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('selection', BoundedQUInt(self.s, self.N)),
                          Register('sigma_mu', _reg_dtype(self.mu)),
                          Register('alt', _reg_dtype(self.s)),
                          Register('keep', _reg_dtype(self.mu)),
                          Register('less_than_equal', QBit())])

    @cached_property
    def qrom(self) -> QROM:
        return QROM((np.array(self.alt), np.array(self.keep)), (self.s,), (self.s, self.mu))

    def build_composite_bloq(self, bb: BloqBuilder, selection, sigma_mu, alt, keep,
                             less_than_equal) -> Dict[str, SoquetT]:
        w = Wires(bb)
        s, mu = self.s, self.mu
        sel = w.split(selection, s)
        sel = w.apply(PrepareUniformSuperposition(self.N), target=sel)['target']
        sig, alt_h, keep_h = w.split(sigma_mu, mu), w.split(alt, s), w.split(keep, mu)
        lte = w.put(less_than_equal)
        if mu:
            sig = w.apply(OnEach(mu, Hadamard()), q=sig)['q']
            out = w.apply(self.qrom, selection=sel, target0_=alt_h, target1_=keep_h)
            sel, alt_h, keep_h = out['selection'], out['target0_'], out['target1_']
            out = w.apply(LessThanEqual(mu, mu), x=keep_h, y=sig, target=lte)
            keep_h, sig, lte = out['x'], out['y'], out['target']
            out = w.apply(RegisterCSwap(s), ctrl=lte, x=alt_h, y=sel)
            lte, alt_h, sel = out['ctrl'], out['x'], out['y']
        return {'selection': w.join(sel, dtype=BoundedQUInt(s, self.N)),
                'sigma_mu': w.join(sig, dtype=_reg_dtype(mu)),
                'alt': w.join(alt_h, dtype=_reg_dtype(s)),
                'keep': w.join(keep_h, dtype=_reg_dtype(mu)),
                'less_than_equal': w.take(lte)}

    def __str__(self):
        return f'AliasSampler(N={self.N}, mu={self.mu})'


# ------------------------------------------------------------------------------------------------
# flags, times, sort
# ------------------------------------------------------------------------------------------------

@frozen
class TimeRegister(Bloq):
    r"""$|0\rangle\mapsto\sum_i\sqrt{p_i}|i\rangle$, $p_i\propto e^{-\mu t_i}$, $t_i=(i+\frac12)T/2^{b_t}$:
    one fixed $R_y(\theta_k)$ on the qubit of weight $2^k$."""
    b_t: int
    mu_T: float = 1.0

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(t=QUInt(self.b_t))

    def build_composite_bloq(self, bb: BloqBuilder, t) -> Dict[str, SoquetT]:
        w = Wires(bb)
        qs = w.split(t, self.b_t)
        for k, th in enumerate(time_angles(self.b_t, self.mu_T)):
            w.ry(qs[k], th)
        return {'t': w.join(qs)}

    def __str__(self):
        return f'Time({self.b_t})'


@frozen
class StepFlags(Bloq):
    r"""$f_r\mathrel{\oplus}=[m\ge r]$, $r=1,\dots,M$: LessThanConstant($s$, $r$) then $X$."""
    s: int
    M: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('sel', QUInt(self.s)), Register('f', QBit(), shape=(self.M,))])

    def build_composite_bloq(self, bb: BloqBuilder, sel, f) -> Dict[str, SoquetT]:
        f = np.array(f)
        for r in range(1, self.M + 1):
            sel, f[r - 1] = bb.add(LessThanConstant(self.s, r), x=sel, target=f[r - 1])
            f[r - 1] = bb.add(XGate(), q=f[r - 1])
        return {'sel': sel, 'f': f}

    def __str__(self):
        return 'Flags'


@frozen
class KeySort(Bloq):
    r"""Sort the steps on the key $(f_r,t_r)$, flag most significant, into descending order.

    Step $r$ goes to slot $k-r$ of $k=2^{\lceil\log_2M\rceil}$ keys; slots $0..k-M-1$ are fresh zero
    keys.  BitonicSort sorts ascending, so slot $k-r$ then holds the $r$-th largest key.  The
    padding is zero again afterwards (zero is the smallest key) and is freed.  The comparison record
    is a new register, cleared only by the inverse sort in $P^\dagger$."""
    M: int
    b_t: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('f', QBit(), shape=(self.M,)),
                          Register('t', QUInt(self.b_t), shape=(self.M,)),
                          Register('record', QBit(), shape=(n_comparators(self.M),), side=Side.RIGHT)])

    def build_composite_bloq(self, bb: BloqBuilder, f, t) -> Dict[str, SoquetT]:
        M, bt = self.M, self.b_t
        k = sort_size(M)
        w = Wires(bb)
        keys = [None] * k
        for r in range(1, M + 1):
            tq = w.split(t[r - 1], bt)
            keys[k - r] = w.join(tq + [w.put(f[r - 1])], dtype=QUInt(bt + 1))
        for slot in range(k - M):
            keys[slot] = bb.allocate(dtype=QUInt(bt + 1))
        keys, record = bb.add(BitonicSort(k, bt + 1), xs=np.array(keys))
        f_out, t_out = np.array(f), np.array(t)
        for r in range(1, M + 1):
            qs = w.split(keys[k - r], bt + 1)
            f_out[r - 1] = w.take(qs[bt])
            t_out[r - 1] = w.join(qs[:bt])
        for slot in range(k - M):
            bb.free(keys[slot])
        return {'f': f_out, 't': t_out, 'record': record}

    def __str__(self):
        return 'Sort'


# ------------------------------------------------------------------------------------------------
# P
# ------------------------------------------------------------------------------------------------

def make_selector(M: int, eta: float, precision: float) -> AliasSampler:
    return AliasSampler.from_probabilities(list(selector_weights(M, eta)), precision=precision)


@frozen
class HistoryPrep(Bloq):
    r"""$P$: selector (alias sampling), step flags, a time register on every step, the sort on
    (flag, time), uniform parents.  Every history register is a through-register that starts in
    $|0\rangle$; the comparison record is created here and consumed by $P^\dagger$."""
    M: int
    b_t: int
    eta: float
    sel_precision: float
    mu_T: float = 1.0

    @cached_property
    def selector(self) -> AliasSampler:
        return make_selector(self.M, self.eta, self.sel_precision)

    @cached_property
    def s(self) -> int:
        return max(1, nbits(self.M + 1))

    @cached_property
    def mu(self) -> int:
        return self.selector.mu

    def history_registers(self) -> List[Register]:
        sig = {r.name: r for r in self.selector.signature}
        regs = [Register('sel', QUInt(self.s)),
                Register('sigma', sig['sigma_mu'].dtype),
                Register('alt', sig['alt'].dtype),
                Register('keep', sig['keep'].dtype),
                Register('lte', QBit()),
                Register('f', QBit(), shape=(self.M,)),
                Register('t', QUInt(self.b_t), shape=(self.M,))]
        for r in range(2, self.M + 1):
            regs.append(Register(f'p{r}', parent_dtype(r)))
        return regs

    @cached_property
    def signature(self) -> Signature:
        regs = self.history_registers()
        if n_comparators(self.M):
            regs.append(Register('record', QBit(), shape=(n_comparators(self.M),), side=Side.RIGHT))
        return Signature(regs)

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
        w = Wires(bb)
        M = self.M
        sig = {r.name: r for r in self.selector.signature}
        sel = w.split(soqs['sel'], self.s)
        out = w.apply(self.selector, selection=sel, sigma_mu=w.split(soqs['sigma'], self.mu),
                      alt=w.split(soqs['alt'], self.s), keep=w.split(soqs['keep'], self.mu),
                      less_than_equal=w.put(soqs['lte']))
        sel_soq = w.join(out['selection'], dtype=QUInt(self.s))
        sel_soq, f = bb.add(StepFlags(self.s, M), sel=sel_soq, f=np.array(soqs['f']))
        t = np.array(soqs['t'])
        for r in range(M):
            t[r] = bb.add(TimeRegister(self.b_t, self.mu_T), t=t[r])
        res = {}
        if n_comparators(M):
            f, t, record = bb.add(KeySort(M, self.b_t), f=f, t=t)
            res['record'] = record
        for r in range(2, M + 1):
            res[f'p{r}'] = bb.add(PrepareUniformSuperposition(r), target=soqs[f'p{r}'])
        res.update(sel=sel_soq, sigma=w.join(out['sigma_mu'], dtype=sig['sigma_mu'].dtype),
                   alt=w.join(out['alt'], dtype=sig['alt'].dtype),
                   keep=w.join(out['keep'], dtype=sig['keep'].dtype),
                   lte=w.take(out['less_than_equal']), f=f, t=t)
        return res

    def __str__(self):
        return 'P'

"""T gates to run a Heisenberg chain in a random field for time t = 10 at a total synthesis
error eps_total in {1e-3, 1e-1}, against the Trotter step dt.

Synthesis: gridsynth; the angle-dependent formulas of Bothe et al. (fallback, Eq. 15, and
ancilla-free, Eq. 12; pgt/bothe.py), which fall back to the mixed fallback and mixed diagonal fits
of Kliuchnikov et al.  Gradient layers: nearest, randomised and sparse-pair rounding, and the
hybrid that sends each rotation to the cheaper of Bothe et al.'s fallback scheme and the sparse
gradient.  Two chains of n = 64 spins: random bonds J_i in [0.5, 1.5] (every angle distinct) and
uniform bonds J = 1, where each method uses the cheaper of Hamming-weight phasing and none.
Fields h_i uniform in [-1, 1].  Gradient costs are averaged over five draws of the roundings.
Writes data/trotter.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pgt.costs import DATA, heisenberg_step_cost  # noqa: E402

N = 64
T_TOTAL = 10.0
EPSS = (1e-3, 1e-1)
METHODS = ('gridsynth', 'bothe', 'bothe_af', 'det', 'rand', 'sparse', 'hybrid')
SYNTH = ('gridsynth', 'bothe', 'bothe_af')
BATCH_DIGITS = 256


def step(J, h, dt, eps_step, method, rng, classes):
    draws = 1 if method in SYNTH else 5
    cs = [heisenberg_step_cost(J, h, dt, eps_step, method, rng, None, classes, BATCH_DIGITS)
          for _ in range(draws)]
    return {'t_step': float(np.mean([c['t'] for c in cs])), 'eps_rot': cs[0]['eps_rot'],
            'rotations': cs[0]['rotations'],
            'peak_ancilla': max(c.get('peak_ancilla', 0) for c in cs)}


def main():
    rng = np.random.default_rng(64)
    h = rng.uniform(-1, 1, N)
    chains = {'random bonds': rng.uniform(0.5, 1.5, N - 1), 'uniform bonds': np.ones(N - 1)}
    rows = []
    for eps in EPSS:
        for name, J in chains.items():
            for dt in 10.0 ** np.arange(-0.5, -6.01, -0.25):
                r = T_TOTAL / dt
                for method in METHODS:
                    routes = [False, True] if name == 'uniform bonds' else [False]
                    best = min((dict(step(J, h, dt, eps / r, method, rng, c), hw=c)
                                for c in routes), key=lambda x: x['t_step'])
                    rows.append(dict(best, chain=name, eps_total=eps, dt=float(dt), steps=r,
                                     method=method, t_total=best['t_step'] * r))
                print(eps, name, round(dt, 7), [round(x['t_step']) for x in rows[-len(METHODS):]])
    with open(os.path.join(DATA, 'trotter.json'), 'w') as fh:
        json.dump({'n': N, 't': T_TOTAL, 'eps_totals': EPSS, 'batch_digits': BATCH_DIGITS,
                   'rows': rows}, fh,
                  indent=1)


if __name__ == '__main__':
    main()

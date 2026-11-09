"""T gates to run a Heisenberg chain in a random field for time t = 10 with a total synthesis
error of 1e-3, against the Trotter step dt: every rotation synthesised (gridsynth, or the mixed
fallback formula) or every layer added into a phase gradient (nearest, randomised or sparse
rounding).  Two chains of n = 64 spins: random bonds J_i in [0.5, 1.5] (every angle distinct),
and uniform bonds J = 1 (the bonds of a layer share an angle: Hamming-weight phasing on both
sides).  Fields h_i uniform in [-1, 1].  The cost of a step is averaged over five draws of the
randomised roundings.  Writes data/trotter.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pgt.costs import DATA, heisenberg_step_cost  # noqa: E402

N = 64
T_TOTAL = 10.0
EPS = 1e-3
BATCH = 64


def main():
    rng = np.random.default_rng(64)
    h = rng.uniform(-1, 1, N)
    chains = {'random bonds': (rng.uniform(0.5, 1.5, N - 1), False),
              'uniform bonds': (np.ones(N - 1), True)}
    rows = []
    for name, (J, classes) in chains.items():
        for dt in 10.0 ** np.arange(-0.5, -5.01, -0.25):
            r = T_TOTAL / dt
            eps_step = EPS / r
            for method in ('gridsynth', 'fallback', 'det', 'rand', 'sparse'):
                draws = 1 if method in ('gridsynth', 'fallback') else 5
                cs = [heisenberg_step_cost(J, h, dt, eps_step, method, rng, BATCH, classes)
                      for _ in range(draws)]
                step = float(np.mean([c['t'] for c in cs]))
                rows.append({'chain': name, 'dt': float(dt), 'steps': r, 'method': method,
                             't_step': step, 't_total': step * r,
                             'eps_rot': cs[0]['eps_rot'], 'rotations': cs[0]['rotations'],
                             'peak_ancilla': max(c.get('peak_ancilla', 0) for c in cs)})
            print(name, dt, [round(x['t_step']) for x in rows[-5:]])
    with open(os.path.join(DATA, 'trotter.json'), 'w') as fh:
        json.dump({'n': N, 't': T_TOTAL, 'eps': EPS, 'batch': BATCH, 'rows': rows}, fh, indent=1)


if __name__ == '__main__':
    main()

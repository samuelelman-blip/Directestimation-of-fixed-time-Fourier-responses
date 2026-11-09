"""T counts of gridsynth (pygridsynth 2.0) for exp(-i theta Z) at diamond accuracy eps.

Writes data/gridsynth.json: the mean T count over random angles at each eps, the least-squares
fit T = a log2(1/eps) + c, and the counts for small angles at fixed eps.  About two minutes.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pgt.costs import DATA, gridsynth_t  # noqa: E402


def main():
    rng = np.random.default_rng(2026)
    rows = []
    for e in range(3, 16):
        eps = 10.0 ** -e
        ts = [gridsynth_t(float(th), eps) for th in rng.uniform(0, math.pi, 40)]
        rows.append({'eps': eps, 'mean': float(np.mean(ts)), 'min': int(min(ts)),
                     'max': int(max(ts)), 'n': len(ts)})
        print(eps, rows[-1]['mean'])
    x = np.array([math.log2(1 / r['eps']) for r in rows])
    y = np.array([r['mean'] for r in rows])
    a, c = np.polyfit(x, y, 1)
    small = []
    for eps in (1e-6, 1e-10):
        for s in range(0, 9):
            th = 10.0 ** -s * 0.7
            ts = [gridsynth_t(th * f, eps) for f in rng.uniform(0.9, 1.1, 20)]
            small.append({'eps': eps, 'theta': th, 'mean': float(np.mean(ts))})
            print(eps, th, small[-1]['mean'])
    out = {'source': 'pygridsynth 2.0.0, gridsynth_gates(theta=2 theta, epsilon=eps/2)',
           'rows': rows, 'fit': {'a': float(a), 'c': float(c)}, 'small_angles': small}
    with open(os.path.join(DATA, 'gridsynth.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('fit', a, c)


if __name__ == '__main__':
    main()

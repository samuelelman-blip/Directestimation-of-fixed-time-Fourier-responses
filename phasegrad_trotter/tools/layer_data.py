"""Per-rotation costs of a phase-gradient layer against synthesis, as the angles shrink.

For each per-rotation diamond error eps in {1e-6, 1e-10, 1e-14}, each angle scale theta (angles
drawn log-uniformly in [theta/2, theta] with random signs) and each rounding ('det', 'rand',
'sparse'), compiles L = 1024 rotations in batches of B = 64 and records T gates per rotation,
ANDs, digits and peak ancillas; also the batch-size trade-off at eps = 1e-10, and the counts
against the bound of the layer theorem.  Writes data/layer.json.  About two minutes.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pgt.costs import (DATA, b_det, b_rand, choose_K, layer_cost, t_gridsynth,  # noqa: E402
                       t_mixed_fallback)
from pgt.digits import digits_mod  # noqa: E402

L = 1024
B = 64


def angles(theta, rng, n=L):
    return np.exp(rng.uniform(math.log(theta / 2), math.log(theta), n)) * rng.choice([-1, 1], n)


def run(eps, theta, mode, rng, batch=B, n=L):
    th = angles(theta, rng, n)
    Ks = [choose_K(float(t), eps, mode, rng)[0] for t in th]
    b = b_det(eps) if mode == 'det' else b_rand(eps)
    c = layer_cost([(K, [j]) for j, K in enumerate(Ks)], b, batch)
    N = sum(1 for K in Ks for p, _ in digits_mod(K, b) if p <= b - 4)
    N3 = sum(1 for K in Ks for p, _ in digits_mod(K, b) if p == b - 3)
    bound = 4 * N + N3 + 4 * (b - 2) * c['batches']
    assert c['t'] <= bound
    return {'eps': eps, 'theta': theta, 'mode': mode, 'b': b, 'B': batch, 'L': n,
            't_per_rot': c['t'] / n, 'and_compute': c['and_compute'],
            'and_gradient': c['and_gradient'], 't_direct': c['t_direct'], 'digits': N,
            'bound_per_rot': bound / n, 'peak_ancilla': c['peak_ancilla'],
            'log2_theta_over_grid': math.log2(theta * 0.75 / (math.pi / 2 ** b))}


def main():
    rng = np.random.default_rng(11)
    rows = []
    for eps in (1e-6, 1e-10, 1e-14):
        for s in np.arange(0.0, 9.01, 0.5):
            theta = 10.0 ** -s
            for mode in ('det', 'rand', 'sparse'):
                rows.append(run(eps, theta, mode, rng))
        print(eps, 'done')
    batches = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        for mode in ('det', 'rand'):
            batches.append(run(1e-10, 0.1, mode, rng, batch=batch))
    out = {'L': L, 'B': B, 'rows': rows, 'batches': batches,
           'gridsynth': {str(e): t_gridsynth(e) for e in (1e-6, 1e-10, 1e-14)},
           'fallback': {str(e): t_mixed_fallback(e) for e in (1e-6, 1e-10, 1e-14)},
           'b_det': {str(e): b_det(e) for e in (1e-6, 1e-10, 1e-14)},
           'b_rand': {str(e): b_rand(e) for e in (1e-6, 1e-10, 1e-14)}}
    with open(os.path.join(DATA, 'layer.json'), 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()

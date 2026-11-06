"""ftfr_q04.accuracy against the reference simulator: the instances, their responses, the closed
form of the first order on a time grid, the selector's realised weights and the bounds."""
import math

import numpy as np
import pytest

from ftfr_q04.accuracy import (instance, instance_for, order1, orders, response, selector_realised,
                               tail_bounds, time_bound, weights)
from ftfr_q04.block import small_cell
from ftfr_q04.history import AliasSampler
from ftfr_q04.simulate import selector_probabilities
from reference_bridge import R, family_for, needs_reference

FAMILIES = [('dense', 'flat'), ('PD', 'F'), ('PD', 'T'), ('sparse', 'flat'), ('PS', 'F'),
            ('PS', 'T')]


@needs_reference
@pytest.mark.parametrize('fa', FAMILIES, ids=lambda fa: '-'.join(fa))
@pytest.mark.parametrize('b', [2, 3])
def test_instance_equals_reference_family(fa, b):
    """The O(n) instance is the G1 the circuit block-encodes (tests/reference_bridge.py)."""
    cell = small_cell(fa[0], fa[1], b, 1, 2)
    inst, fam = instance_for(cell), family_for(cell)
    assert np.max(np.abs(inst.dense_G1() - fam.G1)) < 1e-15
    assert abs(inst.gamma - fam.gamma) < 1e-15
    z = np.random.default_rng(b).normal(size=inst.n) + 1j
    assert np.max(np.abs(inst.G(z) - fam.G1 @ z)) < 1e-14
    assert np.max(np.abs(inst.GT(z) - fam.G1.T @ z)) < 1e-14


@needs_reference
@pytest.mark.parametrize('fa', FAMILIES, ids=lambda fa: '-'.join(fa))
def test_responses_equal_reference(fa):
    cell = small_cell(fa[0], fa[1], 3, 1, 2)
    inst, fam = instance_for(cell), family_for(cell)
    ref, _ = R.classical_orders(fam, 3)
    assert np.max(np.abs(orders(inst, 3) - ref)) < 1e-11
    assert abs(response(inst) - R.full_response(fam)) < 1e-11


@pytest.mark.parametrize('fa', FAMILIES, ids=lambda fa: '-'.join(fa))
@pytest.mark.parametrize('b', [3, 10])
def test_order1_closed_form_is_the_hierarchy(fa, b):
    inst = instance(fa[0], fa[1], b, 12, 14)
    assert abs(order1(inst) - orders(inst, 1)[1]) < 1e-11


@needs_reference
@pytest.mark.parametrize('fa', [('dense', 'flat'), ('PD', 'F'), ('PS', 'T')],
                         ids=lambda fa: '-'.join(fa))
@pytest.mark.parametrize('b_t', [1, 3, 6])
def test_order1_on_a_grid_equals_the_reference_programme(fa, b_t):
    """The grid sum in closed form equals the reference's dynamic programme over time cells."""
    cell = small_cell(fa[0], fa[1], 2, 1, b_t)
    inst, fam = instance_for(cell), family_for(cell)
    _, per_order = R.structural_amplitude(fam, 1, b_t, block=fam.G1 / fam.gamma)
    _, _, Xi = R.normalisation(fam, 1)
    assert abs(order1(inst, b_t) - Xi * per_order[1]) < 1e-14


@pytest.mark.parametrize('M', [1, 2, 3])
def test_selector_realised_equals_simulation(M):
    """The alias sampler's output distribution, from its tables and from the gates."""
    cell = small_cell('dense', 'flat', 2, M, 1, sel_precision=0.05)
    sim = selector_probabilities(cell)
    assert np.max(np.abs(selector_realised(M, cell.eta, 0.05) - sim)) < 1e-12


@pytest.mark.parametrize('M', [1, 2, 4])
@pytest.mark.parametrize('precision', [1e-2, 1e-4, 1e-6])
def test_selector_is_within_its_precision(M, precision):
    from ftfr_q04.history import selector_weights
    eta = 0.125 * (1 - math.exp(-1))
    q = selector_weights(M, eta)
    assert np.max(np.abs(selector_realised(M, eta, precision) - q)) <= precision


@needs_reference
def test_weights_and_bounds_equal_reference():
    cell = small_cell('dense', 'flat', 3, 3, 4)
    inst, fam = instance_for(cell), family_for(cell)
    W, Xi = weights(inst, 3)
    _, W_ref, Xi_ref = R.normalisation(fam, 3)
    assert np.max(np.abs(W - W_ref)) < 1e-15 and abs(Xi - Xi_ref) < 1e-15
    assert abs(tail_bounds(inst, 3)[0] - R.tail_bound(fam, 3)) < 1e-15
    assert abs(time_bound(inst, 3, 4, worst=True) - R.appendix_a_bound(fam, 3, 4)) < 1e-15

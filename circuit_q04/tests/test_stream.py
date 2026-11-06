"""The walker's leaf mapping and the simulator, against Qualtran 0.4's tensor simulation."""
from functools import cached_property

import numpy as np
import pytest
from attrs import frozen

from qualtran import Bloq, BloqBuilder, QBit, QUInt, Signature
from qualtran.bloqs.basic_gates import (CNOT, CZPowGate, Hadamard, Ry, Rz, SGate, TGate, Toffoli,
                                        XGate, YGate, ZGate, ZPowGate)
from qualtran.bloqs.mcmt import And

from ftfr_q04.stream import BloqStream, count, leaf_report, paper_counts, t_count
from ftfr_q04.testing import assert_close, stream_matrix, tensor_matrix
from ftfr_q04.wires import Wires

LEAVES = [XGate(), YGate(), ZGate(), Hadamard(), SGate(), SGate(is_adjoint=True), TGate(),
          TGate(is_adjoint=True), CNOT(), Toffoli(), ZPowGate(exponent=0.37),
          CZPowGate(exponent=1.0), CZPowGate(exponent=0.3), Rz(angle=0.71), Ry(angle=-1.3),
          And(), And(0, 1), And(1, 0), And(0, 0)]


@pytest.mark.parametrize('leaf', LEAVES, ids=[str(b) for b in LEAVES])
def test_leaf_matches_tensor(leaf):
    assert_close(stream_matrix(leaf), tensor_matrix(leaf), 1e-12, str(leaf))


@pytest.mark.parametrize('cv', [(1, 1), (0, 1), (1, 0), (0, 0)])
@pytest.mark.parametrize('measure', [False, True])
def test_and_uncompute(cv, measure):
    """AND then its uncompute is the identity, also with the target X-measured (random outcomes
    and the CZ fix-up)."""

    @frozen
    class AndRoundTrip(Bloq):
        @cached_property
        def signature(self):
            return Signature.build(a=1, b=1)

        def build_composite_bloq(self, bb, a, b):
            w = Wires(bb)
            ha, hb = w.put(a), w.put(b)
            for q in (ha, hb):
                w.h(q)
            t = w.and_(ha, hb, *cv)
            w.t(t)                                  # a phase on the AND, to make it visible
            w.anddg(ha, hb, t, *cv)
            return {'a': w.take(ha), 'b': w.take(hb)}

    bl = AndRoundTrip()
    for seed in range(4):
        m = stream_matrix(bl, measure_ands=measure, seed=seed)
        # expected: H x H, then phase e^{i pi/4} on the basis state (cv1, cv2)
        h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        hh = np.kron(h, h)
        ph = np.ones(4, dtype=complex)
        ph[2 * cv[0] + cv[1]] = np.exp(1j * np.pi / 4)
        assert_close(m, np.diag(ph) @ hh, 1e-12, f'cv={cv} seed={seed}')


def test_wires_toffoli_cswap_and_zpow():
    @frozen
    class Mixed(Bloq):
        @cached_property
        def signature(self):
            return Signature.build(c=1, x=1, y=1)

        def build_composite_bloq(self, bb, c, x, y):
            w = Wires(bb)
            hc, hx, hy = w.put(c), w.put(x), w.put(y)
            w.h(hc)
            w.cswap(hc, hx, hy)
            w.toffoli(hc, hx, hy)
            w.zpow(hy, 0.25)                        # emitted as T
            w.zpow(hx, 0.3)                         # emitted as ZPowGate
            w.cz(hc, hy)
            w.ry(hx, 0.4)
            return {'c': w.take(hc), 'x': w.take(hx), 'y': w.take(hy)}

    bl = Mixed()
    ref = tensor_matrix(bl.decompose_bloq())
    assert_close(stream_matrix(bl), ref, 1e-12, 'mixed')
    c = count(BloqStream(bl))
    assert c['TOFFOLI'] == 0 and c['AND'] == 2 and c['ANDDG'] == 2 and c['T'] == 1
    assert c['ZPOW'] == 1 and c['CZPOW'] == 1 and c['RY'] == 1


def test_leaf_report_and_paper_counts():
    @frozen
    class TwoAnds(Bloq):
        @cached_property
        def signature(self):
            return Signature.build(a=1, b=1)

        def build_composite_bloq(self, bb, a, b):
            w = Wires(bb)
            ha, hb = w.put(a), w.put(b)
            t = w.and_(ha, hb, 0, 1)
            w.zpow(t, 0.5 + 1e-14)                  # within 1e-12 of S
            w.anddg(ha, hb, t, 0, 1)
            return {'a': w.take(ha), 'b': w.take(hb)}

    rep = leaf_report(TwoAnds())
    assert rep['leaves']['AND'] == 1 and rep['leaves']['ANDDG'] == 1 and rep['peak'] == 3
    pc = paper_counts(BloqStream(TwoAnds()))
    assert pc['X'] == 4 and pc['S'] == 1 and t_count(pc) == 4

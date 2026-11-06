"""Every case of data/reference_counts.json, built in Qualtran 0.4 (tests/reference_cases.py), has
exactly the leaf counts and the peak qubit count of the reference implementation
(tools/reference_counts.py)."""
import json
import os

import pytest

from reference_cases import CASES
from ftfr_q04.stream import BloqStream, StreamContext, paper_counts

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
REF = json.load(open(os.path.join(DATA, 'reference_counts.json')))


@pytest.mark.parametrize('name', sorted(CASES), ids=str)
def test_counts_equal_reference(name):
    bloq = CASES[name]()
    ctx = StreamContext()
    c = paper_counts(BloqStream(bloq, ctx))
    got = {k: v for k, v in c.items() if k not in ('ALLOC', 'FREE')}
    assert got == REF[name]['counts']
    assert ctx.alloc.peak == REF[name]['peak']
    if 'alias' in REF[name]:                           # the selector's alias tables
        a = REF[name]['alias']
        assert (list(bloq.alt), list(bloq.keep), bloq.mu) == (a['alt'], a['keep'], a['mu'])


def test_every_reference_case_is_covered():
    assert sorted(CASES) == sorted(REF)

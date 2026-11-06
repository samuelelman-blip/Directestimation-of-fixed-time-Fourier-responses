"""The gate stream as cirq operations, produced one at a time from the leaf walk.

    stream = CirqStream(bloq)
    for op in stream:                  # cirq.Operation on cirq.LineQubit(id)
        ...
    stream.peak                        # qubits used (the number of distinct LineQubits)

The leaves become the cirq operations Qualtran 0.4 gives them (`as_cirq_op`): X Y Z H S S**-1 T
T**-1 CNOT, CZPowGate, ZPowGate, Rz, Ry, CCX.  And and And-dagger stay as Qualtran's `And` gate,
which is a cirq gate, unless `expand_and=True`; then they are written out exactly as
`And.decompose_from_registers` writes them: four T gates for And, and for And-dagger an X-basis
measurement of the target, a CZ on the controls conditioned on the outcome, and a reset.  An X on a
control conjugates each gate whose control value is 0.

Measurements made by measuring bloqs are `cirq.measure` followed by `cirq.reset` (an X-basis
measurement has a Hadamard first).  A gate conditioned on the parity of several outcomes carries
`with_classical_controls(ParityCondition(...))`, a `cirq.Condition` that is true when the XOR of
its outcomes is 1; its `keys` are computed only when read, because a lookup's conditions can name
thousands of outcomes.  One outcome alone is a plain key.  Keys are 'm<k>' for measuring bloqs and
'a<k>' for expanded And-daggers.

Qubits are cirq.LineQubit(id) with the walker's integer ids.  An id is reused after its qubit is
released (released qubits are |0>, which the simulator in stream.py checks), so the distinct
LineQubits number exactly the peak qubit count.  No operation is kept after it is yielded, so
memory does not grow with the length of the stream.  Operations are built without cirq's argument
check (see `_unchecked`), which for Qualtran's gates costs tens of microseconds each.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import cirq

from qualtran import Bloq
from qualtran.bloqs.mcmt import And

from .stream import BloqStream, Condition, Op, StreamContext

_ONE_QUBIT = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z, 'H': cirq.H, 'S': cirq.S,
              'SDG': cirq.S ** -1, 'T': cirq.T, 'TDG': cirq.T ** -1}


class _Qubits:
    def __init__(self, factory):
        self.factory = factory
        self.cache = []

    def __getitem__(self, i: int) -> cirq.Qid:
        while len(self.cache) <= i:
            self.cache.append(self.factory(len(self.cache)))
        return self.cache[i]


class ParityCondition(cirq.Condition):
    """True when the XOR of the outcomes `keys` is 1 (0 with `invert`).  The keys can be given as a
    function, called the first time they are read."""

    def __init__(self, keys=None, invert: bool = False, key_fn=None, label: str = ''):
        self._keys = None if keys is None else tuple(_mkey(k) for k in keys)
        self._key_fn = key_fn
        self.invert = bool(invert)
        self.label = label

    @property
    def keys(self):
        if self._keys is None:
            self._keys = tuple(_mkey(k) for k in self._key_fn())
        return self._keys

    def replace_key(self, current, replacement):
        return ParityCondition([replacement if k == current else k for k in self.keys], self.invert)

    def resolve(self, classical_data) -> bool:
        v = 0
        for k in self.keys:
            v ^= classical_data.get_int(k) & 1
        return bool(v ^ self.invert)

    @property
    def qasm(self):
        raise ValueError('a parity condition has no OpenQASM 2 form')

    def __eq__(self, other):
        return (isinstance(other, ParityCondition) and self.invert == other.invert
                and self.keys == other.keys)

    def __hash__(self):
        return hash((self.keys, self.invert))

    def __str__(self):
        if self.label and self._keys is None:
            return self.label
        body = ' ^ '.join(str(k) for k in self.keys[:6]) + (' ^ ...' if len(self.keys) > 6 else '')
        return f'!({body})' if self.invert else f'({body})'

    def __repr__(self):
        return f'ParityCondition({[str(k) for k in self.keys]!r}, invert={self.invert})'


def _mkey(k):
    return k if isinstance(k, cirq.MeasurementKey) else cirq.MeasurementKey(str(k))


def cirq_condition(cond: Condition):
    """The cirq form of a parity condition on outcomes 'm<k>' (None: never true, True: always)."""
    invert = bool(getattr(cond, 'invert', False))
    if getattr(cond, 'lazy', False):
        return ParityCondition(key_fn=lambda c=cond: [f'm{k}' for k in c.keys()], invert=invert,
                               label=repr(cond))
    keys = [f'm{k}' for k in cond.keys()]
    if not keys:
        return True if invert else None
    if len(keys) == 1 and not invert:
        return keys[0]
    return ParityCondition(keys, invert)


def _unchecked(gate, qubits):
    """cirq.GateOperation(gate, qubits) without gate.validate_args, which for a Qualtran gate costs
    tens of microseconds; the stream only ever passes the right number of distinct qubits."""
    op = cirq.GateOperation.__new__(cirq.GateOperation)
    op._gate = gate
    op._qubits = qubits
    return op


def _choose_constructor():
    """Use _unchecked only if it builds operations equal to cirq's own (it relies on the two
    attributes GateOperation.__init__ sets in cirq 1.4)."""
    a, b, c = cirq.LineQubit.range(3)
    try:
        for gate, qs in ((cirq.CNOT, (a, b)), (And(0, 1), (a, b, c)), (cirq.rz(0.3), (a,))):
            ref, op = gate.on(*qs), _unchecked(gate, qs)
            if not (op == ref and hash(op) == hash(ref) and repr(op) == repr(ref)):
                raise ValueError
        return _unchecked
    except Exception:                                  # pragma: no cover - other cirq versions
        return lambda gate, qs: gate.on(*qs)


_op = _choose_constructor()
_AND_GATES = {(cv1, cv2, u): And(cv1, cv2, uncompute=u) for cv1 in (0, 1) for cv2 in (0, 1)
              for u in (False, True)}
_TDG, _SDG = cirq.T ** -1, cirq.S ** -1


def _and_expanded(c1, c2, t, cv, uncompute: bool, key: Optional[str]):
    flips = [_op(cirq.X, (q,)) for q, v in zip((c1, c2), cv) if v == 0]
    out = list(flips)
    if uncompute:
        out += [_op(cirq.H, (t,)), cirq.measure(t, key=key),
                cirq.CZ(c1, c2).with_classical_controls(key), cirq.reset(t)]
    else:
        g = _op
        out += [g(cirq.H, (t,)), g(cirq.T, (t,)), g(cirq.CNOT, (c1, t)), g(cirq.CNOT, (c2, t)),
                g(cirq.CNOT, (t, c1)), g(cirq.CNOT, (t, c2)), g(_TDG, (c1,)), g(_TDG, (c2,)),
                g(cirq.T, (t,)), g(cirq.CNOT, (t, c1)), g(cirq.CNOT, (t, c2)), g(cirq.H, (t,)),
                g(cirq.S, (t,))]
    return out + flips


def to_cirq(ops: Iterable[Op], *, expand_and: bool = False, qubit=cirq.LineQubit) -> Iterator[cirq.Operation]:
    """Convert leaf ops (name, qubits, arg) of stream.py to cirq operations, lazily."""
    Q = _Qubits(qubit)
    n_and = 0
    gates = {}                                         # parametrised gates, by (name, parameter)
    for name, qs, arg in ops:
        if name in ('ALLOC', 'FREE'):
            continue
        cond = None
        if isinstance(arg, Condition):
            cond = cirq_condition(arg)
            if cond is None:
                continue
            arg = None
        if name in _ONE_QUBIT:
            out = (_op(_ONE_QUBIT[name], (Q[qs[0]],)),)
        elif name == 'CNOT':
            out = (_op(cirq.CNOT, (Q[qs[0]], Q[qs[1]])),)
        elif name in ('AND', 'ANDDG'):
            cv = tuple(arg) if isinstance(arg, tuple) else (1, 1)
            c1, c2, t = Q[qs[0]], Q[qs[1]], Q[qs[2]]
            if expand_and:
                key = None
                if name == 'ANDDG':
                    key = f'a{n_and}'
                    n_and += 1
                out = _and_expanded(c1, c2, t, cv, name == 'ANDDG', key)
            else:
                out = (_op(_AND_GATES[cv + (name == 'ANDDG',)], (c1, c2, t)),)
        elif name in ('CZ', 'CZPOW', 'ZPOW', 'RZ', 'RY'):
            p = 1.0 if (name == 'CZ' or arg is None) else float(arg)
            gate = gates.get((name, p))
            if gate is None:
                if name in ('CZ', 'CZPOW'):
                    gate = cirq.CZPowGate(exponent=p)
                elif name == 'ZPOW':
                    gate = cirq.ZPowGate(exponent=p)
                else:
                    gate = cirq.rz(p) if name == 'RZ' else cirq.ry(p)
                if len(gates) > 4096:
                    gates.clear()
                gates[(name, p)] = gate
            out = (_op(gate, tuple(Q[q] for q in qs)),)
        elif name == 'TOFFOLI':
            out = (_op(cirq.CCX, (Q[qs[0]], Q[qs[1]], Q[qs[2]])),)
        elif name == 'MZ':
            out = (cirq.measure(Q[qs[0]], key=f'm{arg}'), cirq.reset(Q[qs[0]]))
        elif name == 'MX':
            out = (_op(cirq.H, (Q[qs[0]],)), cirq.measure(Q[qs[0]], key=f'm{arg}'),
                   cirq.reset(Q[qs[0]]))
        else:
            raise ValueError(f'no cirq form for {name}')
        if cond is not None and cond is not True:
            out = [op.with_classical_controls(cond) for op in out]
        yield from out


class CirqStream:
    """Iterate over the leaf gates of `bloq` as cirq operations.  `inputs` gives qubit ids for the
    left registers (default: fresh ids 0, 1, ... in signature order).  Once iteration has started,
    `input_qubits(name)` gives the LineQubits of a left register; after it, `output_qubits(name)`
    those of a right register and `peak` the number of qubits used.  Qubits are listed as Qualtran
    splits a register, most significant first."""

    def __init__(self, bloq: Bloq, ctx: Optional[StreamContext] = None, inputs=None, *,
                 expand_and: bool = False, qubit=cirq.LineQubit):
        self.stream = BloqStream(bloq, ctx, inputs)
        self.expand_and = expand_and
        self.qubit = qubit

    def __iter__(self) -> Iterator[cirq.Operation]:
        return to_cirq(self.stream, expand_and=self.expand_and, qubit=self.qubit)

    def input_qubits(self, name: str):
        return [self.qubit(int(i)) for i in self.stream.inputs[name].reshape(-1)]

    def output_qubits(self, name: str):
        return [self.qubit(int(i)) for i in self.stream.outputs[name].reshape(-1)]

    @property
    def peak(self) -> int:
        return self.stream.ctx.alloc.peak

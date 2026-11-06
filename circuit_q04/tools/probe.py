"""Print what a compiler walking Qualtran 0.4 bloqs will see: for each bloq, its registers, its
decomposition tree (each composite bloq once, with the bloqs its decompose_bloq() contains), the
largest single decomposition, the leaf gates reached, the T count, the peak number of live qubits,
and whether pyLIQTR's estimate_resources gives the same T and rotation counts (where it does not,
the probe names the sub-bloqs that pyLIQTR counts from their unitary instead of their
decomposition).

    python3 tools/probe.py                 # every bloq in ftfr_q04/catalog.py
    python3 tools/probe.py Square CAdd     # those whose name contains one of the words
    python3 tools/probe.py --json out.json # also write the report as JSON

Leaves are XGate YGate ZGate Hadamard SGate TGate (and adjoints) CNOT Toffoli ZPowGate CZPowGate Rz
Ry And And-dagger, plus the bookkeeping bloqs Split Join Partition Cast Allocate Free.  A measuring
bloq (MeasureX, PhaseLeaf) is shown with the cirq operations of its decompose_from_registers.

Peak qubits assume each decomposition runs in the greedy order (allocations as late and frees as
early as possible: ftfr_q04.stream.greedy_bloqnections), which is what the paper's qubit counts
assume.  Qualtran 0.4's own CompositeBloq.iter_bloqnections() runs every allocation of a
decomposition at its start; where that raises the peak, the probe prints both numbers.
"""
import json
import os
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ftfr_q04.catalog import CATALOG  # noqa: E402
from ftfr_q04.stream import (BOOKKEEPING, LEAF_TYPES, BloqStream, Condition,  # noqa: E402
                             StreamContext, bloq_key, iter_decompositions, paper_counts, t_count)
from ftfr_q04.testing import pyliqtr_view  # noqa: E402


def measured_summary(bloq):
    """The cirq operations of a measuring bloq's decompose_from_registers, by kind, and the most
    outcomes any of its parity conditions names."""
    import cirq
    from qualtran._infra.gate_with_registers import get_named_qubits
    ops = list(cirq.flatten_op_tree(bloq.decompose_from_registers(
        context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
        **get_named_qubits(bloq.signature))))
    kinds, widest = Counter(), 0
    for op in ops:
        if isinstance(op, cirq.ClassicallyControlledOperation):
            kinds[f'{op.without_classical_controls().gate} if parity'] += 1
            widest = max(widest, max(len(c.keys) for c in op.classical_controls))
        elif cirq.is_measurement(op):
            kinds['measure'] += 1
        else:
            kinds[str(op.gate)] += 1
    body = ', '.join(f'{k} x{v}' for k, v in sorted(kinds.items()))
    return body + (f'; conditions on up to {widest} outcomes' if widest else '')


def leaf_name(b):
    n = type(b).__name__
    if n in ('SGate', 'TGate') and getattr(b, 'is_adjoint', False):
        return n + '†'
    if n == 'And' and b.uncompute:
        return 'And†'
    return n


def tree(bloq, max_depth=16):
    """Each distinct composite bloq once, with the bloqs its decompose_bloq() contains (leaves by
    name), and the largest decomposition (bloq instances in one decompose_bloq(), bookkeeping
    included).  Two different bloqs that print the same (Qualtran 0.4 prints BiQubitsMixer and its
    adjoint alike) are told apart by their repr."""
    nodes = []
    for depth, b, cb in iter_decompositions(bloq, max_depth):
        if isinstance(cb, Exception):
            nodes.append((depth, b, cb, 0))
            continue
        kids = sorted((bi.bloq for bi in cb.bloq_instances
                       if not isinstance(bi.bloq, BOOKKEEPING)), key=repr)
        nodes.append((depth, b, kids, len(cb.bloq_instances)))
    names = {}
    for _, b, kids, _ in nodes:
        for x in [b] + (kids if isinstance(kids, list) else []):
            if type(x) not in LEAF_TYPES:
                names.setdefault(str(x), set()).add(bloq_key(x))

    def label(x):
        if type(x) in LEAF_TYPES:
            return leaf_name(x)
        s = str(x)
        return s if len(names.get(s, ())) <= 1 else repr(x)[:120]

    out, largest, described = [], (0, ''), set()
    if getattr(bloq, 'measured', False):
        out.append(f'{label(bloq)}: measured (cirq): {measured_summary(bloq)}')
    for depth, b, kids, size in nodes:
        if isinstance(kids, Exception):
            out.append('  ' * depth + f'{label(b)}: DOES NOT DECOMPOSE '
                       f'({type(kids).__name__}: {kids})')
            continue
        if size > largest[0]:
            largest = (size, label(b))
        parts = Counter(label(c) for c in kids)
        body = ', '.join(f'{k} x{v}' for k, v in sorted(parts.items())) or '(no gates)'
        out.append('  ' * depth + f'{label(b)}: {body}')
        for c in kids:
            if getattr(c, 'measured', False) and bloq_key(c) not in described:
                described.add(bloq_key(c))
                out.append('  ' * (depth + 1) + f'{label(c)}: measured (cirq): '
                           f'{measured_summary(c)}')
    return out, largest


def report(name, bloq):
    ctx = StreamContext()
    leaves = Counter()
    ops = []
    for op in BloqStream(bloq, ctx):
        ops.append(op)
        if op[0] not in ('ALLOC', 'FREE'):
            if isinstance(op[2], Condition):
                leaves[op[0] + '_IF'] += 1
                continue
            leaves[op[0] + ('/cv=' + ''.join(map(str, op[2])) if op[0] in ('AND', 'ANDDG')
                            and op[2] != (1, 1) else '')] += 1
    pc = paper_counts(ops)
    ctx0 = StreamContext(greedy=False)                 # Qualtran 0.4's own iter_bloqnections order
    for _ in BloqStream(bloq, ctx0):
        pass
    try:                                               # pyLIQTR's own estimate, for comparison
        from pyLIQTR.utils.resource_analysis import estimate_resources
        est = estimate_resources(bloq, profile=True)
        pyliqtr = dict(T=int(est['T']), rotations=int(est['Rotations']))
    except Exception as e:                             # pragma: no cover - reported to the user
        pyliqtr = dict(error=f'{type(e).__name__}: {e}')
    walked = (t_count(pc), pc['RZ'] + pc['RY'])
    if 'T' in pyliqtr and (pyliqtr['T'], pyliqtr['rotations']) != walked:
        vc, direct = pyliqtr_view(bloq)                # gates pyLIQTR counts without decomposing
        if direct and (pyliqtr['T'], pyliqtr['rotations']) == (
                t_count(vc) + sum(tc.t for _, tc in direct),
                vc['RZ'] + vc['RY'] + sum(tc.rotations for _, tc in direct)):
            kinds = Counter(repr(sub) for sub, _ in direct)
            pyliqtr['counted_directly'] = dict(sorted(kinds.items()))
    regs = ', '.join(f'{r.name}{list(r.shape) if r.shape else ""}: {r.dtype}'
                     + ('' if r.side.name == 'THRU' else f' ({r.side.name})') for r in bloq.signature)
    lines, (size, where) = tree(bloq)
    return OrderedDict(name=name, bloq=str(bloq), registers=regs, tree=lines,
                       largest_decomposition=dict(bloq_instances=size, bloq=where),
                       leaves=dict(sorted(leaves.items())), T=t_count(pc),
                       rotations=pc['RZ'] + pc['RY'], peak_qubits=ctx.alloc.peak,
                       peak_qubits_q04_order=ctx0.alloc.peak, pyliqtr_estimate=pyliqtr)


def main(argv):
    out_json = None
    if '--json' in argv:
        i = argv.index('--json')
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    sel = [n for n in CATALOG if not argv or any(a.lower() in n.lower() for a in argv)]
    reps = []
    for name in sel:
        r = report(name, CATALOG[name]())
        reps.append(r)
        print(f'== {name}')
        print(f'   registers: {r["registers"]}')
        for line in r['tree']:
            print('   ' + line)
        print(f'   leaves: {r["leaves"]}')
        ld = r['largest_decomposition']
        print(f'   largest decomposition: {ld["bloq_instances"]} bloq instances ({ld["bloq"]})')
        extra = ('' if r['peak_qubits_q04_order'] == r['peak_qubits'] else
                 f' ({r["peak_qubits_q04_order"]} in Qualtran 0.4\'s iter_bloqnections order)')
        print(f'   T = {r["T"]}, rotations = {r["rotations"]}, peak qubits = {r["peak_qubits"]}{extra}')
        p = r['pyliqtr_estimate']
        same = p.get('T') == r['T'] and p.get('rotations') == r['rotations']
        if same:
            print('   pyLIQTR estimate_resources: same T and rotation counts\n')
        elif 'counted_directly' in p:
            what = ', '.join(f'{k} x{v}' for k, v in p['counted_directly'].items())
            print(f'   pyLIQTR estimate_resources: T = {p["T"]}, rotations = {p["rotations"]}\n'
                  f'     the difference: pyLIQTR counts {what} from its unitary, not its\n'
                  f'     decomposition (README, "pyLIQTR and the one-bit comparison")\n')
        else:
            print(f'   pyLIQTR estimate_resources: {p}\n')
    if out_json:
        json.dump(reps, open(out_json, 'w'), indent=1)
        print('wrote', out_json)


if __name__ == '__main__':
    main(sys.argv[1:])

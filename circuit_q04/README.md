# circuit_q04: the estimator circuit in Qualtran 0.4

This is the estimator circuit written from scratch for Qualtran 0.4.0 and pyLIQTR 1.4.2. Every bloq
here is a Qualtran 0.4 bloq, and a compiler that walks them (`decompose_bloq()` or the call graph)
reaches only the leaf gates listed below.

The repository's earlier implementation in `../circuit` serves only as the **reference
implementation**: it was written for Qualtran 0.7, nothing in `ftfr_q04` or `tests` imports it, and
its gate counts and peak qubit counts were recorded once into `data/reference_counts.json` and
`data/reference_cells.json` by `tools/reference_counts.py`, the one script here that runs in its
environment. The tests check that this package reproduces those counts exactly, and the cost
model's numbers behind Tables IV and V of the paper (`data/cells.json`). Where a docstring names a
Qualtran 0.7 bloq, it says which gate sequence of the reference implementation is reproduced; the
code itself uses Qualtran 0.4 only. (The *reference simulator* is a different thing: the paper's
state-vector simulation of one block, `../implementation_guide/reference/reference_block.py`,
which the block tests compare against.)

The port is done in four checkpoints:

1. **Primitives** (done): controlled Hadamards, basis-state phases, reflections, controlled swaps
   and routing, additions into the phase-gradient register, squaring, controlled addition,
   comparison and sorting.
2. **History preparation** (done): alias sampling, step flags, time registers, key sorting, parent
   pointers, and the inverse.
3. **The middle block and row access** (done): unary iteration, the select-swap lookup and its
   measurement-based uncompute, closed-form and tabulated row access, rows, vertex phases,
   contractions, the middle block C-X.
4. **The block, the readout and the run** (done): the block W, the Hadamard test, the reflection,
   the Grover iterate, the run plan; whole blocks simulated gate by gate against the reference
   simulator; the nine costed cells streamed at n = 2^20 and counted against the cost model and
   Tables IV and V.

## Environment

```
python3.11 -m venv venv && . venv/bin/activate
pip install -r requirements.txt        # qualtran 0.4.0, pyLIQTR 1.4.2 (pins qualtran 0.4.0),
                                       # cirq-core 1.4.0, numpy 1.26
python -m pytest tests -q              # 685 tests, about 4 min (15 skipped, see the next line)
FTFR_SLOW=1 python -m pytest tests -q  # also the cells with tables streamed again and all
                                       # sixteen Clifford+T blocks (about 15 min)
python tools/probe.py                  # what a compiler sees, for each bloq in ftfr_q04/catalog.py
python tools/probe.py Square CAdd      # only those whose name contains one of the words
python tools/probe.py --json out.json  # also write the report as JSON
python tools/plot_versions.py          # figures comparing this package with the reference
python tools/error_checks.py           # the errors at small n and against n (about 40 min)
python tools/plot_errors.py            # figures of those errors
```

## Streaming

Your compiler walks the bloqs itself (`decompose_bloq()`, depth first) and streams its own output.
At n = 2^20 one Grover iterate is 9,232 to 2.0 x 10^8 gates, depending on the cell, and a whole run
1.2 x 10^7 to 2.8 x 10^15, so nothing may need the whole circuit at once. Three rules keep that
possible:

1. **No large decomposition.** Every `decompose_bloq()` reachable from the Grover iterate of any
   of the nine n = 2^20 cells holds at most 10,000 bloq instances (`tests/test_levels.py`, which
   walks each iterate down to its leaves and checks every kind of bloq in the table lookups; the
   largest is 2,052, in the one-hot encoding of the largest lookup uncompute). The probe prints
   the largest decomposition of each bloq. A depth-first walk holds one decomposition per depth,
   so its memory does not grow with the length of the circuit. A table lookup is a tree of small
   bloqs that refer to their table instead of holding it (see the checkpoint 3 section).
2. **Repetition stays explicit.** A run is the phase-gradient preparation once, then the Grover
   iterate `nq` times (1,225 to 14.8 million, depending on the cell). `run_plan(cell)` gives the
   two bloqs and a header with `nq`, the T count of each bloq and the peak qubit count, so the
   iterate is walked once and repeated (see the checkpoint 4 section).
3. **Conditions are parities.** A gate conditioned on measurement outcomes is conditioned on the
   parity of a list of them. Such gates occur only in the lookup uncomputes (checkpoint 3), and
   there the lists are long: see "The size of the conditions" below.

For comparison, this package can stream the same gates itself. Both forms are generators; their
memory grows with the number of distinct bloqs (one template each, see below), not with the length
of the stream:

```python
from ftfr_q04.cirq_stream import CirqStream
from ftfr_q04.stream import BloqStream

stream = CirqStream(bloq)                 # cirq operations on cirq.LineQubit(id)
for op in stream:
    ...
stream.peak                               # qubits used

for name, qubits, arg in BloqStream(bloq):   # the same gates as tuples (ftfr_q04/stream.py)
    ...
```

`CirqStream` gives each leaf the cirq operation Qualtran 0.4 gives it (`as_cirq_op`). `And` and
`And†` stay as Qualtran's `And` gate, which is a cirq gate, unless `expand_and=True`. Then they are
written out exactly as `And.decompose_from_registers` writes them: 4 T for `And`, and for `And†` an
X-basis measurement, a CZ on the controls conditioned on the outcome (`with_classical_controls`),
and a reset. A parity condition is a `ParityCondition` (`ftfr_q04/cirq_stream.py`), a
`cirq.Condition` that is true when the XOR of the outcomes in its `keys` is 1; the keys are
computed only when read. Qubit ids are reused once a qubit is released, so the distinct
`LineQubit`s number exactly the peak.

### Compile each distinct bloq once

Walking every instance through Qualtran 0.4 runs at about 60,000 gates per second, because most of
the time goes into Qualtran's bookkeeping of soquets and connections. The circuit repeats itself at
every level, so `ftfr_q04/compiled.py` walks each distinct bloq's decomposition once and records it
as a template. A template is a list of leaf gates on local qubit slots, with the allocations and
releases in the order the walk makes them, and calls to the templates of larger sub-bloqs; a
sub-bloq whose template has at most 64 entries is copied in instead of called. Replaying a template
on real qubit ids gives exactly the stream that walking every instance gives: the same gates on the
same ids, the same outputs and the same peak (`tests/test_compiled.py`, in both decomposition
orders). On the machine used here (one core), once the templates are recorded (seconds), the
stream runs at about 1.1 to 1.5 million gates per second as tuples and 0.5 to 0.8 million as cirq
operations, so the largest iterate (2.0 x 10^8 gates) takes about three minutes as tuples and five
to six as cirq operations. `StreamContext(compiled=False)` walks every instance instead.

Your compiler can do the same. Every bloq reached here is hashable and compares by value (checked
in `tests/test_compiled.py`), so a cache keyed on the bloq itself finds each repeated sub-bloq;
`compiled.py` is a working example (about 200 lines).

## What the compiler sees

Every bloq here decomposes through `decompose_bloq()` into these leaves:

| leaf (Qualtran 0.4 class) | notes |
|---|---|
| `XGate`, `YGate`, `ZGate`, `Hadamard` | |
| `SGate`, `TGate` | also with `is_adjoint=True` |
| `CNOT` | |
| `CZPowGate` | `exponent=1.0` is CZ; Qualtran 0.4 has no CZ bloq |
| `ZPowGate` | `diag(1, e^{iπt})`; our bloqs use it only when t is not a multiple of 1/4 (Qualtran's `PhaseGradientState` also uses it for Z, S and T) |
| `Rz`, `Ry` | `exp(-iθZ/2)`, `exp(-iθY/2)`; left as rotations, not synthesised |
| `And(cv1, cv2)` | fresh target = [c1 = cv1] AND [c2 = cv2]; 4 T. Control values may be 0 |
| `And(cv1, cv2, uncompute=True)` | measurement-based uncompute of an `And`; 0 T |
| `Toffoli` | in the agreed leaf set, but nothing here emits it |
| `MeasureX(width, key, slot)` | X-basis measurement and reset of each qubit of a lookup slot (checkpoint 3); decomposes only through `decompose_from_registers`, like `And†` (see below) |
| `PhaseLeaf(spec, key, h, cv, controlled)` | CZ (or Z) gates conditioned on parities of `MeasureX` outcomes (checkpoint 3); likewise |

plus the bookkeeping bloqs `Split`, `Join`, `Partition`, `Cast`, `Allocate`, `Free`.

`And(uncompute=True)` is a leaf because Qualtran 0.4 cannot turn its decomposition into bloqs:
its `decompose_from_registers` measures the target in the X basis and applies a CZ conditioned on
the outcome (`cirq.measure` and `with_classical_controls`). From checkpoint 3 on, the measured
uncompute of a table lookup follows the same pattern, in two bloqs: `MeasureX` (X-basis
measurements) and `PhaseLeaf` (CZ gates conditioned on parities of their outcomes). Both are
`GateWithRegisters` with a `decompose_from_registers`, like `And`, and set the class attribute
`measured = True`; the walker in `ftfr_q04/stream.py` takes their gates from `measured_ops`,
which `tests/test_lookup.py` checks against the cirq decomposition, operation by operation.

A controlled swap of two qubits is CNOT, `And`, CNOT, `And†`, CNOT (4 T). Qualtran 0.4's
`TwoBitCSwap` decomposes into 7 T, so it is not used.

Qualtran 0.4's own `Add`, `AddK` (also controlled, and its `Adjoint`), `LessThanConstant`,
`LessThanEqual`, `PrepareUniformSuperposition`, `PhaseGradientState` and `QROM` already decompose to
these leaves, with the same gate counts and peak qubit counts as the reference implementation, so
they are used as they are. The walk also passes through Qualtran's `OnEach`, `MultiAnd`,
`MultiControlX`, `BiQubitsMixer` and `SingleQubitCompare`, which decompose to the same leaves. An
inverse is either the bloq's own `adjoint()`, where it defines one (`HalfLessThan`,
`RegisterCSwap` and `BasisPhase`, for example), or Qualtran's `Adjoint` of the bloq, which
decomposes into the inverses of its parts, down to the same leaves (`And` and `And†` swap).

## History preparation (checkpoint 2)

`HistoryPrep(M, b_t, eta, sel_precision)` is P, in the same order as the reference implementation:

| bloq | what it does | built from |
|---|---|---|
| `AliasSampler` | the selector m with probabilities q_m, by coherent alias sampling | Qualtran's `PrepareUniformSuperposition`, `OnEach(Hadamard)`, `QROM` and `LessThanEqual`, and `RegisterCSwap` |
| `StepFlags` | f_r = [m >= r] for r = 1..M | Qualtran's `LessThanConstant` and an X |
| `TimeRegister` | the time of one step, p_i proportional to exp(-mu t_i) | one `Ry` per qubit |
| `KeySort` | the steps sorted on (flag, time), descending; the comparison record is a new register | `BitonicSort` |
| parents | p_r uniform on 0..r-1, r = 2..M | Qualtran's `PrepareUniformSuperposition` |

`AliasSampler` has the structure of Qualtran's `StatePreparationAliasSampling`, with two changes:
its tables are tuples, so it is hashable, and its controlled swap costs one AND per qubit pair, as
in the reference implementation. Its tables are those of Qualtran's own routine, and for all nine
cells they equal the reference implementation's. P-dagger is `Adjoint(HistoryPrep(...))`; it
consumes the comparison record by running the sort backwards. At all nine cells the peak of P is
the same in both decomposition orders.

## The middle block and row access (checkpoint 3)

`MiddleBlock(M, b, b_t, w_phi, skeleton, access)` is C-X, in the same order as the reference
implementation: the folded controls, U_0 (`FanHadamard`), the root phase (`PhaseGradAdd`), the
vertex phases (`VertexPhase`), the contractions C_M ... C_1 (`Contraction`: `RouteToZero`,
`ControlledRowInverse`, `ContractionPhase`, and for the profiled families `RowAccess`), and the
inverse steps. `RowAccess` is closed form (`ValueOracleF`, `InequalityTest`, `RowPhaseF`, made
from Qualtran's `AddK` and this package's `HalfLessThan`, `Square` and `PhaseGradAdd`) or tabulated
(`SelectSwapLookup`, `InequalityTest`, a controlled `PhaseGradAdd`, `LookupUncompute`). At all nine
n = 2^20 cells C-X has exactly the gate counts and peak qubit count of the reference
implementation, the three cells with tables included.

### The lookups as bloq trees

A table holds 2^20 entries, so a lookup cannot be one decomposition. In `ftfr_q04/lookup.py`:

- `SelectSwapLookup(spec)`: allocates 2^lambda slots, `UnaryLoad` writes the block selected by the
  high selection bits into them, `RouteToZero` moves the selected entry to slot 0. Slot 0 is the
  output (`target0`, `target1`, one register per column); the other slots are junk.
- `UnaryLoad` is a unary iteration written as a tree: `LoadNode(level, prefix)` holds an `And`, a
  CNOT, an `And†` and its two children; a leaf `LoadLeaf(block)` writes one block with one
  `FanOut(width, mask)` per slot, a CNOT from the control onto each target qubit whose mask bit is
  1. A decomposition never holds more than one node's worth of bloqs (at the largest cell: 520 for
  the lookup itself, 128 for a leaf, 59 for a `FanOut`).
- `LookupUncompute(spec, key)`: `MeasureX` per slot (H, measurement, reset per qubit), a one-hot
  encoding of the low selection bits (`OneHotEncode`, Qualtran's `Adjoint` of it to undo), and
  `PhaseFixup`, a tree of `PhaseNode`s over the high bits whose leaves `PhaseLeaf(h)` apply, for
  each index j of their block, a CZ between the iteration flag and qubit (j mod 2^kk) of the
  one-hot register, conditioned on the parity f(j) of the measured outcomes.
- The bloqs refer to the table through a `LookupSpec`, which is compared and hashed by a digest of
  its contents, so equal lookups are equal bloqs and hashing one costs nothing.
- Measurement keys are named `{key}_{slot}_{qubit}`. Every use of an uncompute with the same key
  reuses these names, as Qualtran's `And†` reuses the name of its target; a condition always refers
  to the latest outcome of that name, measured earlier in the same uncompute.
- `UnaryLoad` and `PhaseFixup` also have `stream_ops`, a generator of exactly the gates of their
  trees, which this package's walker uses so that streaming a lookup does not build 10^6 Qualtran
  objects (`tests/test_lookup.py` checks that both give the same gates). Your compiler can walk
  the trees, or treat `FanOut` as a leaf if it has a fan-out instruction.

### The size of the conditions

Each uncompute applies 2^20 conditioned CZ gates (one per table entry), and each condition is the
parity of the outcomes of the measured qubits that hold a 1 after the lookup of that entry. At the
three cells with tables an uncompute measures 7,296 to 7,936 qubits, and a condition names 2,816
to 3,385 outcomes on average, depending on the cell (5,952 at most). An iterate has 2M uncomputes
(M = 1 or 4).
`PhaseLeaf` builds each condition only when it is read (`ParityCondition` with lazily computed
keys), but a compiler that expands every condition into its list of outcomes handles 5.9 x 10^9
to 2.4 x 10^10 outcome references per iterate. If that is too much, the usual alternative is to let
the classical controller compute the 2^20 bits f(j) once per uncompute, from the outcomes and the
table (about 10^8 word operations), and to condition each CZ on one of those bits. The gates and
the T count are the same either way.

## The block, the readout and the run (checkpoint 4)

In `ftfr_q04/block.py`, in the same order as the reference implementation:

| bloq | what it is |
|---|---|
| `Cell` | one costed configuration: family, row data, b, M, b_t, w_phi, selector precision, row-data widths, T per rotation, `nq`. `load_cells()` reads the nine costed cells (the eight of Tables IV and V, and PS-F) from `data/cells.json` |
| `Block(cell)` | W = P-dagger (C-X) P, on the Hadamard-test qubit `anc`, the history registers, the work registers and the phase-gradient register; the comparison record of the sort lives only inside the block |
| `HadamardTest(cell, imag=False)` | A: H on `anc` (then S-dagger for the imaginary part), W, H on `anc` |
| `ReflectZero(cell)` | S_0 = I - 2\|0><0\| on `anc` and every history and work register (not the phase gradient): one `ZeroReflection` |
| `GroverIterate(cell, imag=False)` | Q = A S_0 A-dagger S_0, so in time order S_0, `Adjoint(A)`, S_0, A |
| `run_plan(cell)` | the run: `prep` (`PhaseGradientState(w_phi)`) once, then `iterate` (`GroverIterate(cell)`) `repeats` = `nq` times, with a header (n, M, b_t, w_phi, T per rotation, T of `prep`, T of `iterate`, T of the run, qubits) |

A compiler takes a run as two walks and a loop:

```python
from ftfr_q04.block import load_cells, run_plan

for cell in load_cells():
    plan = run_plan(cell)
    plan['prep'], plan['iterate'], plan['repeats']   # walk each bloq once; repeat the iterate
```

### Whole blocks against the reference simulator

`ftfr_q04/simulate.py` streams a block at small n into the exact sparse simulator: every register
starts in |0>, `anc` is set to 1 and the phase gradient prepared, then W runs gate by gate, then the
phase gradient and `anc` are undone and the amplitude <0|W|0> is read off. `tests/test_block.py`
compares it with the paper's reference simulator
(`../implementation_guide/reference/reference_block.py`, numpy and scipy), which sums the
histories directly, using the selector weights the alias sampler actually realises:

- 15 blocks (dense, sparse, PD and PS; no row data, closed form and tables; b = 2 and 3; M = 1 to
  3) agree to within 1e-12 (in practice about 1e-15);
- the same with every `And` written out in Clifford+T gates and every uncompute measured with
  random outcomes, as `And.decompose_from_registers` writes them: seven blocks (covering every
  family and kind of row data) by default, sixteen with `FTFR_SLOW=1`;
- with a precise selector the realised weights approach q_m, and the time register equals the
  reference's product state.

### Counts at n = 2^20

`tools/count_cells.py` streams P, P-dagger, C-X, W, the reflection, the phase-gradient state and
Q for each of the nine cells and writes `data/stream_counts.json` (about 14 minutes, most of it the
three cells with tables). Every part of every cell has exactly the reference implementation's gate
counts and peak qubit count, and the totals are the cost model's (`data/cells.json`), which for the
eight cells in the paper are the numbers of Tables IV and V:

| cell | M | b_t | T per block | T per iterate | qubits | nq | T per run | gates per iterate |
|---|---|---|---|---|---|---|---|---|
| dense eps=0.01 | 1 | 7 | 3,693 | 7,986 | 180 | 1,349 | 1.08 x 10^7 | 9,232 |
| dense eps=1e-6 | 4 | 21 | 43,252 | 88,560 | 558 | 14,802,482 | 1.31 x 10^12 | 71,838 |
| PD-F eps=0.01 | 1 | 8 | 7,807 | 16,398 | 261 | 1,279 | 2.10 x 10^7 | 20,810 |
| PD-F eps=1e-6 | 4 | 22 | 60,144 | 123,080 | 743 | 14,246,636 | 1.75 x 10^12 | 118,846 |
| PD-T eps=0.01 | 1 | 10 | 61,178 | 123,124 | 8,093 | 1,225 | 1.51 x 10^8 | 29,990,114 |
| PD-T eps=1e-6 | 4 | 23 | 329,280 | 661,624 | 7,788 | 13,804,694 | 9.13 x 10^12 | 199,535,950 |
| sparse eps=1e-6 | 1 | 21 | 9,110 | 19,132 | 272 | 11,462,746 | 2.19 x 10^11 | 15,532 |
| PS-T eps=1e-6 | 1 | 23 | 64,508 | 130,192 | 7,651 | 10,812,967 | 1.41 x 10^12 | 25,332,024 |
| PS-F eps=1e-6 | 1 | 21 | 9,412 | 19,912 | 316 | 11,412,838 | 2.27 x 10^11 | 17,550 |

T counts here include the rotations at the cell's T per rotation; "qubits" is the peak of the
iterate, in the greedy order. A run is `nq` iterates plus the phase-gradient state. An iterate is
two blocks and two reflections (the Hadamard test adds only H and S-dagger), and in every block
each `And` is uncomputed by measurement. One block's counts by gate type are the columns of
Table V: ANDs, T, rotations, Cliffords (the conditioned CZs included), measurements (the `And`
uncomputes and the `MeasureX` qubits), T per block. `tests/test_block.py` checks all of this
against the recorded file; it also streams the six cells without tables again (seconds each),
and with `FTFR_SLOW=1` the three with tables.

The figures in `figures/` (`python tools/plot_versions.py`, a few seconds) show the comparison
with the reference implementation: every gate count of both versions plotted one against the
other (`fig1_counts_parity.png`: 545 counts at the nine cells and 169 in the 43 small test cases,
none different), one iterate by gate type at each cell (`fig2_iterate_gate_mix.png`), the T count
of each part at each cell (`fig3_t_by_part.png`), and the peak qubits of each part in the greedy
order and in Qualtran 0.4's own order (`fig4_qubit_peaks.png`). The numbers plotted are in
`figures/*.csv`.

### pyLIQTR and the one-bit comparison

pyLIQTR's estimate agrees with the leaf walk for every bloq in the catalog except the blocks with
M = 1, where it reports 4 T fewer per block. For a bloq without an explicit T-complexity
annotation, pyLIQTR asks Qualtran's `_from_directly_countable` before looking at its call graph,
and that counts a gate on at most two qubits whose unitary is a Clifford as one Clifford, whatever
its decomposition. With M = 1 the step flag f_1 = [m >= 1] is computed by Qualtran 0.4's
`LessThanConstant(1, 1)` and an X; x < 1 is NOT x, so the gate's unitary is a Clifford, but its
decomposition (what your compiler walks, and what the reference implementation and the cost model
count) spends one `And` on it. P-dagger holds `Adjoint(LessThanConstant(1, 1))`, which pyLIQTR
counts through its call graph, so the difference is in P only. `tests/test_pyliqtr.py` checks that this is the only difference.

Since m has one bit when M = 1, f_1 is m itself and the comparison with its X could be one CNOT.
That would save 4 T in P and 4 in P-dagger: 16 T per iterate at six of the nine cells (0.2% at
dense eps=0.01, less elsewhere). The circuit keeps the `And` so that its counts stay those of the
paper.

## Accuracy: errors at small scale and against n

`tools/error_checks.py` (about 40 minutes, most of it the integrations at n = 2^20) writes
`data/error_checks.json`, and `tools/plot_errors.py` draws figures 5 to 7 from it. The estimate of a
run is the scalar that the circuit's amplitude encodes, Xi_M <0|W|0>. Its error is measured against
the exact response g(u(T)) of the full quadratic equation, integrated directly (the dense family's
g agrees with the paper's `data/family.csv` to 1e-16). Amplitude estimation then reads the scalar to
within the readout allowance, and a compiler adds the synthesis error of the rotations; both are
allowances of the error budget and are not simulated.

- **Small instances (`fig5_small_scale_errors.png`).** 34 runs of the streamed circuit, simulated
  gate by gate at n = 4, 8 and 16 with M = 1 to 3 (seven of them with every `And` written out in
  Clifford+T and every uncompute measured at random), give the reference simulator's block
  amplitude to within 8.9e-15. Over these runs and 576 configurations of the reference engine
  (all six families, n = 4 and 8, M = 1 to 3, b_t = 2, 4, 6 and 8, selector precision 1e-1 to
  1e-4), the measured error of the estimate is at most 0.28 of the calculated error (truncation
  certificate plus time-grid bound plus selector bound). Taken one source at a time: the
  time-grid error is second order (4x per bit) under Appendix A's first-order bound (2x per bit);
  the dense family's truncation error is under its certified bound at n = 4 and at n = 2^20; the
  selector error is under Xi_M (M + 1) times the alias sampler's precision.
- **Against n (`fig6_error_vs_n.png`).** Each cell's circuit parameters (M, b_t and selector
  precision, as costed at n = 2^20) are kept fixed while n goes from 4 to 2^20. The circuit is not
  simulated here: the error of its estimate is formed from the exact responses, the realised
  selector weights and, for M = 1, the time grid in closed form (for M = 4 the grid's error is
  below 1e-13 at the cells' b_t and is left out). At n = 2^20 this error is 0.016 to 0.16 of the
  calculated truncation, time-grid and selector error in every cell. With the dense skeleton the
  calculated error changes by at most 1.3x for n >= 2^6 (by 1.2x to 2.7x from n = 4). With the
  sparse skeleton the error and its certified bound both fall as 1/n, and the costed M = 1 relies
  on that: for n up to 2^8 the error is above eps, and the bound says so too. The rotation and
  readout allowances use the rest of eps at n = 2^20.
- **Source by source (`fig7_error_sources_vs_n.png`).** Truncation, selector and time grid,
  measured and calculated, with the rotation and readout allowances, for four cells. The
  time-grid error of the M = 1 cells is computed exactly at every n, as the difference of the
  closed-form grid sum and its limit (`ftfr_q04.accuracy.order1`).

The PD-T and PS-T instances hold the closed-form profiles rounded to the table widths, which is
what the circuit's tables hold; their errors are therefore close to those of PD-F and PS-F. The
paper's tabulated profiles are random, which changes the data but not the circuit or its error
budget.

`ftfr_q04/accuracy.py` holds the instances at any n (G1 applied in O(n)), the integrations, the
closed form, the bounds and the selector's realised weights; `tests/test_accuracy.py` checks them
against the reference simulator. The calculated bounds reproduce the cost model's budget at
n = 2^20 (`data/error_budget.json`, from the best reading of the cost model): for dense eps = 0.01,
for example, 1.22e-3 truncation, 7.1e-4 time grid, 9.7e-5 selector and 1.3e-4 rotations, with
7.8e-3 left for the readout. The numbers plotted are in `figures/errors_gate_level.csv` and
`figures/errors_vs_n.csv`.

## Counting conventions

`paper_counts` in `ftfr_q04/stream.py` counts a gate stream as the paper's Table V does:

- an `And` or `And†` with a control value 0 counts two X gates for each such control;
- a `ZPowGate` or `Rz` within 1e-12 of a Clifford+T gate counts as that gate (T, S, Z and their
  adjoints), any other counts as one rotation;
- `CZPowGate(1)` counts as CZ;
- an X-basis measurement of `MeasureX` counts as one `MX` (its H is part of it), and a gate
  conditioned on outcomes counts under its name with `_IF` (`CZ_IF`, `Z_IF`).

`t_count(counts, t_per_rotation=0)` = T + T† + 4 per `And` + 4 per `Toffoli` + `t_per_rotation` per
rotation. The cells use their own T per rotation (`Cell.ts_per_rot`, from the cost model); with
the default 0 the rotations are left out and counted separately.

## Qubit counts and the order of a decomposition

The peak number of live qubits depends on the order in which the bloqs of each decomposition run.
The paper's qubit counts, taken from the reference implementation, use a greedy order: a
topological sort that runs each `Allocate` as late as possible and each `Free` as early as
possible. In other words, a qubit is allocated at its first use and released after its last use.
(This is the order Qualtran 0.7 uses, which the reference implementation inherited.) Qualtran 0.4's
`CompositeBloq.iter_bloqnections()` sorts by topological generation instead, which puts every
`Allocate` of a decomposition at its start.

The walker uses the greedy order by default (`StreamContext(greedy=True)`, via
`greedy_bloqnections`, which sorts Qualtran 0.4's own graph of the decomposition). `greedy=False`
gives Qualtran 0.4's order; if your compiler allocates in that order, its peaks are the
`greedy=False` ones. The circuits are the same either way; only the peak differs. The probe prints
both peaks where they differ: in the catalog, `HalfLessThan` (15 qubits, or 16 in the 0.4 order)
and the bloqs that contain it (`InequalityTest`, the row access and the small middle blocks with
row access), each by one qubit. At the nine cells P, C-X, W and Q have the same peak in both
orders, except at PS-T, where C-X needs 7,597 qubits instead of 7,596 and W and Q need 7,652
instead of 7,651 (`tools/count_cells.py --q04-order` records the peaks of P, C-X, W and Q in the
0.4 order as `peak_q04_order` in `data/stream_counts.json`).

## Checks

- `tests/test_stream.py`: each leaf, as walked and simulated, against Qualtran 0.4's tensor
  contraction; an `And`, a T gate on its target and the measured uncompute give the expected
  unitary for all four control values and several random measurement outcomes.
- `tests/test_arith.py`: every primitive, simulated from its leaf gates, against its
  specification (the full unitary for the small gadgets, every addend on the phase-gradient state
  for the phase-gradient additions, every basis input or a random sample of them for the
  arithmetic and sorting), including HalfLessThan followed by its measured uncompute; no
  primitive emits a Toffoli. A walk that completes has reached only leaf gates, since anything else
  must decompose.
- `tests/test_history.py`: the selector's distribution equals its alias tables' and is within
  the requested precision of q_m; step flags, the time register and the key sort against their
  specifications; P prepares exactly the distribution of (m, flags, sorted times, parents)
  computed classically, for six small (M, b_t); P then P-dagger returns |0...0> with the uncomputes
  measured at random; and at the parameters of the nine n = 2^20 cells, the selector has the
  reference implementation's alias tables and P and P-dagger have exactly its gate counts and peak
  qubit counts (`data/reference_cells.json`), with the same peak in Qualtran 0.4's decomposition
  order.
- `tests/test_reference_counts.py`: all 43 cases of `data/reference_counts.json` (including the
  QROM and the selector, with its alias tables) have exactly the leaf counts and the peak qubit
  count of the reference implementation.
- `tests/test_pyliqtr.py`: pyLIQTR's `estimate_resources(bloq, profile=True)`, which uses
  Qualtran's call graph and T-complexity annotations, gives the same T and rotation counts as the
  leaf walk for every bloq in the catalog, with the one exception explained above ("pyLIQTR and
  the one-bit comparison"), which the test pins down.
- `tests/test_lookup.py`: the select-swap lookup on every basis input (slot 0 holds the entry,
  the other slots the rest of its block, routed); lookup then measured uncompute returns any
  superposition of indices exactly, for random measurement outcomes, through the generators and
  through the bloq trees; the generators give the same gates as the trees; the cirq decompositions
  of `MeasureX` and `PhaseLeaf` give the walker's gates and conditions; at n = 2^20 the block sizes
  are the reference implementation's (Qualtran 0.7's rule for QROAMClean, ported) and 4 times
  the AND counts equal the T counts it charges for the lookup (61,716) and its uncompute (8,176).
- `tests/test_middle.py`: vertex phases, the row gadget, the controlled inverse with the phase
  gate (block i G1/gamma on control 1, identity on 0) and the row access (block w_j e^{i psi_j},
  closed form and tables) against their specifications; at the nine n = 2^20 cells C-X has the
  reference implementation's gate counts and peak qubit count (the three cells with tables only
  with `FTFR_SLOW=1`).
- `tests/test_cirq_stream.py`: the cirq stream of 25 bloqs, simulated in cirq on every basis input
  (or a sample), gives the same states as the stream simulator, with `And` kept or written out
  with measurements; parity conditions and the reset after a measurement behave as stated; the
  first operations come out before the walk is finished.
- `tests/test_levels.py`: the bound on the size of each decomposition: for the primitives at
  register sizes at or above the cells', for the Grover iterate of each of the nine cells down to
  its leaves (the lookups aside), and for every kind of bloq in the lookup trees at the largest
  cell, with the cirq operations of the measuring bloqs (at most 3,072, in a `PhaseLeaf`).
- `tests/test_compiled.py`: replaying templates gives the same stream as walking every instance,
  for every bloq in the catalog and six larger ones, in both decomposition orders; every bloq
  reached is hashable.
- `tests/test_block.py`: whole blocks simulated gate by gate against the reference simulator
  (`tests/reference_bridge.py` loads it; these tests are skipped if it is missing), also in
  Clifford+T with measured uncomputes; the selector's realised weights and the time register
  against the reference; at the nine n = 2^20 cells the counts and peaks of W, the reflection, the
  phase-gradient state and Q against the reference implementation, and their T counts, the run
  total and the qubit count against the cost model (`data/stream_counts.json` for the cells with
  tables, unless `FTFR_SLOW=1`); every column of Table V against the counts of one block; the
  peaks in Qualtran 0.4's order; the run plan.
- `tests/test_accuracy.py`: the O(n) instances equal the G1 the circuit block-encodes; their
  responses equal the reference simulator's; the closed form of the first order equals the
  hierarchy and, on a time grid, the reference's dynamic programme; the alias sampler's weights
  from its tables equal the simulated ones and lie within its precision; the bounds equal the
  reference's.

## Qualtran's own conversion to cirq

Qualtran 0.4 can also convert a bloq to cirq itself (`to_cirq_circuit`, and through it pyLIQTR's
`circuit_decompose_multi` and `openqasm`). That conversion does not work reliably for bloqs that
allocate and free qubits. When a composite bloq is converted, each `Free` hands its qubits back to
the qubit manager during the conversion, before the sub-bloqs that run earlier have been
decomposed. When one of those sub-bloqs is decomposed and allocates a qubit, it can be given one
that is still in use ("Duplicate qids"). This already happens for Qualtran 0.4's own `AddK`.
`CirqStream` does not have this problem, because the walker allocates qubits in time order.

## Contents

| path | |
|---|---|
| `ftfr_q04/stream.py` | the walker (`BloqStream`, `walk`), the leaf ops, the decomposition order, counting, an exact sparse simulator, the decomposition sizes (`iter_decompositions`, `largest_level`) |
| `ftfr_q04/cirq_stream.py` | the gate stream as cirq operations (`CirqStream`, `to_cirq`) |
| `ftfr_q04/compiled.py` | templates: each distinct bloq recorded once and replayed (`template`, `replay`) |
| `ftfr_q04/wires.py` | `Wires`: writes a decomposition gate by gate on top of `BloqBuilder`, emitting only leaf gates (registers split least significant bit first) |
| `ftfr_q04/gates.py` | `CHadamardExact`, `FanHadamard`, `BasisPhase`, `ZeroReflection`, `RegisterCSwap`, `RouteToZero` |
| `ftfr_q04/phasegrad.py` | `WindowAdd` (Gidney's adder into a window of qubits), `PhaseGradAdd` |
| `ftfr_q04/arith.py` | `Square`, `CAdd`, `BitwiseNot`, `OutOfPlaceAdder`, `HalfLessThan`, `GreaterThan`, `Comparator`, `ParallelComparators`, `BitonicMerge`, `BitonicSort` |
| `ftfr_q04/history.py` | `AliasSampler`, `StepFlags`, `TimeRegister`, `KeySort`, `HistoryPrep` |
| `ftfr_q04/lookup.py` | `LookupSpec`, `SelectSwapLookup` (`UnaryLoad`, `LoadNode`, `LoadLeaf`, `FanOut`), `LookupUncompute` (`MeasureX`, `OneHotEncode`, `PhaseFixup`, `PhaseNode`, `PhaseLeaf`) |
| `ftfr_q04/rows.py` | `AChi`, `RowPrepDense`, `ControlledRowInverse`, `ContractionPhase` |
| `ftfr_q04/access.py` | `ValueOracleF`, `InequalityTest`, `RowPhaseF`, `RowAccess`, `make_access` |
| `ftfr_q04/middle.py` | `VertexPhase`, `Contraction`, `MiddleBlock` |
| `ftfr_q04/block.py` | `Cell`, `load_cells`, `small_cell`, `Block`, `HadamardTest`, `ReflectZero`, `GroverIterate`, `run_plan` |
| `ftfr_q04/simulate.py` | a block streamed into the simulator (`simulate_block_amplitude`), `And` written out in Clifford+T (`expand_and`), the selector's realised weights |
| `ftfr_q04/catalog.py` | small instances of every bloq (`PRIMITIVES`, `HISTORY`, `MIDDLE`, `BLOCK`, `CATALOG`), used by the probe and the tests |
| `ftfr_q04/accuracy.py` | the instances at any n, exact and truncated responses, the time grid in closed form for M = 1, the error bounds, the selector's realised weights |
| `ftfr_q04/testing.py` | matrices of walked circuits by simulation, tensor contraction, basis-state maps, output distributions |
| `tools/probe.py` | the compiler probe |
| `tools/count_cells.py` | streams every part of the nine n = 2^20 cells and writes `data/stream_counts.json` (about 14 minutes) |
| `tools/plot_versions.py` | figures 1 to 4 in `figures/`: this package against the reference implementation, gate counts and qubit peaks, with the numbers as CSV |
| `tools/error_checks.py` | the errors of the estimate at small n (gate level and reference engine) and with the cells' parameters at n = 4 to 2^20; writes `data/error_checks.json` |
| `tools/plot_errors.py` | figures 5 to 7 in `figures/` from `data/error_checks.json`, with the numbers as CSV |
| `tools/reference_counts.py` | writes `data/reference_counts.json` and `data/reference_cells.json` from the reference implementation (the only file here that needs Qualtran 0.7 and `../circuit`; not needed to use or test this package) |
| `data/cells.json` | the nine costed cells (the eight of Tables IV and V, and PS-F): parameters and the cost model's numbers |
| `data/stream_counts.json` | the counts, peaks and T counts streamed by `tools/count_cells.py` |
| `data/error_budget.json` | the nine cells' error budget (truncation, time grid, selector, rotations, readout) from the cost model's best reading |
| `data/error_checks.json` | the errors computed by `tools/error_checks.py` |
| `data/reference_counts.json` | leaf counts and peaks of 43 small cases in the reference implementation |
| `data/reference_cells.json` | the nine n = 2^20 cells: parameters, and the reference implementation's counts and peaks of P, P-dagger, C-X, W, the reflection, the phase-gradient state and the Grover iterate |
| `data/probe.json` | the probe's report for the catalog |

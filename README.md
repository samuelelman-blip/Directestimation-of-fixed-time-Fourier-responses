# Direct estimation of fixed-time Fourier responses

Code and data for the paper *Direct estimation of fixed-time Fourier responses* (S. J. Elman,
G. Waite and A. Alase). The paper gives a quantum algorithm that estimates a polynomial of the state
of a damped quadratic system at a fixed time, directly from the initial data, by averaging
interaction histories in superposition; this repository holds the fault-tolerant circuit for that
estimator, written in [Qualtran](https://github.com/quantumlib/Qualtran) 0.4, with the tests, the
gate counts and the error checks behind the numbers quoted in the paper.

Everything is in one package, `circuit_q04/`. Its own [README](circuit_q04/README.md) describes the
circuit bloq by bloq; this page says what is here, how to run it, and where each number in the paper
comes from.

## The talk

`talk/Direct_Fourier_Responses.html` is the slide deck for the talk on this work, as a single
self-contained page: download it and open it in a browser. Press Play to run the talk with
captions, use the arrow keys to step through by hand, N for the speaker notes and F for full
screen.

## What is here

| path | |
|---|---|
| `circuit_q04/ftfr_q04/` | the circuit: history preparation, the contractions and row access, the block W, the Hadamard test, the reflection and the Grover iterate, as Qualtran 0.4 bloqs that decompose to Clifford gates, T gates, logical ANDs (uncomputed by measurement) and rotations; a walker that streams the gates, counts them and simulates them exactly at small n; the error bounds of the paper |
| `circuit_q04/tests/` | 685 tests: every primitive against its specification, whole blocks against an independent sum over histories, and the gate counts of every costed configuration against the recorded data |
| `circuit_q04/tools/` | the scripts that produced the data and figures (gate counts, error checks, the n = 2^40 point, plots, a compiler probe) |
| `circuit_q04/data/` | the costed configurations and their counts, the error budget, the error checks, and the recorded counts of the earlier Qualtran 0.7 version of the circuit |
| `circuit_q04/figures/` | the comparison figures and the numbers plotted, as CSV |
| `talk/` | the slide deck of the talk, as one HTML file |

## Running it

Python 3.11. Qualtran 0.4.0 is on PyPI; pyLIQTR 1.4.2 is not, and is installed from its GitHub tag
(`requirements.txt` does this).

```
cd circuit_q04
python3.11 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q              # about 3 minutes
```

616 tests pass and 69 are skipped. The skipped tests compare whole blocks with the paper's reference
simulator, a numpy/scipy state-vector simulation that is not part of this repository; the amplitudes
it produced are recorded in `circuit_q04/data/error_checks.json`, and the tests that check the
recorded counts and bounds run without it. The slow checks (the three configurations with table
lookups streamed again, and every Clifford+T block) run with `FTFR_SLOW=1`.

To stream the circuit yourself:

```python
from ftfr_q04.block import load_cells, run_plan
from ftfr_q04.stream import BloqStream

cell = load_cells()[1]                 # dense family, n = 2^20, eps = 1e-6
plan = run_plan(cell)                  # the phase-gradient state once, then the iterate nq times
for name, qubits, arg in BloqStream(plan['iterate']):
    ...
```

## The numbers in the paper

| in the paper | where it comes from |
|---|---|
| the estimator in Qualtran, compiled to Cliffords, T gates, measured ANDs and fixed rotations | `ftfr_q04/`; `python tools/probe.py` lists the leaf gates every bloq reaches |
| the simulated amplitude agrees with an independently coded sum over histories to 9 x 10^-15 | `data/error_checks.json`, key `gate`: 34 blocks simulated gate by gate at n = 4, 8 and 16 (seven of them in Clifford+T with measured uncomputes), largest difference 8.9 x 10^-15; `figures/fig5_small_scale_errors.png` |
| the error of the computed response against direct integration does not grow with n | `figures/errors_vs_n.csv` and `fig6_error_vs_n.png`: with each configuration's parameters fixed at their n = 2^20 values, the error falls from n = 4 to n = 2^20 for the dense family (6.2 x 10^-4 to 3.3 x 10^-4 at eps = 0.01, 4.9 x 10^-8 to 8.8 x 10^-9 at eps = 10^-6) and as 1/n for the sparse one |
| the gate-level simulations reach n = 16 and at most 66 live qubits | the `n` and `peak` fields of the same 34 runs |
| 1.3 x 10^12 T gates at n = 2^20 and eps = 10^-6 (dense family) | `data/cells.json`, the `dense eps=1e-06` cell: 88,560 T per Grover iterate, 14,802,482 iterates, 1.31 x 10^12 T in all, 558 qubits; `data/stream_counts.json` has the same counts streamed from the circuit, and `tests/test_block.py` checks the two agree |
| 3.6 x 10^14 T gates and 1,169 logical qubits at n = 2^40 and eps = 10^-8 | `python tools/headline.py`, which streams the Grover iterate at that point (M = 6 interactions, 28 time bits, 300,912 T per iterate, 1,169 qubits) and forms the run total from the error budget; its output is `data/headline.json` |
| the response magnitude of the dense family lies between 0.3584 and 0.3594 for n = 4 to 4096 | `data/error_checks.json`, key `responses`, family `dense-flat`: the exact responses from direct integration at n = 4 to 2^20 |

The paper's resource estimate costs nine configurations at n = 2^20 (`data/cells.json`): the dense
family at eps = 0.01 and 10^-6, the sparse family at 10^-6, and the profiled dense and sparse
families with closed-form and with tabulated row data. The table in `circuit_q04/README.md`
(section "Counts at n = 2^20") lists all nine with their T counts, qubit counts and numbers of
iterates.

## What is not here

- The earlier version of the circuit, written for Qualtran 0.7. The present package was written
  from scratch for Qualtran 0.4 and reproduces that version's gate counts exactly; the counts
  themselves are recorded in `data/reference_counts.json` and `data/reference_cells.json`, and
  `tools/reference_counts.py` is the script that recorded them.
- The reference simulator (the independent sum over histories that the block tests compare
  against). Its results are in `data/error_checks.json`; `tests/reference_bridge.py` loads it if it
  is placed at `../implementation_guide/reference/reference_block.py`.
- The cost model that chose each configuration's parameters and split its error budget. Its
  outputs are `data/cells.json` and `data/error_budget.json`, and the tests check the circuit
  against them.
- The classical sampler of the same histories, whose running time the paper quotes for comparison.

## Licence

Copyright (C) 2026 Samuel J. Elman. This code is free software, released under the GNU Lesser
General Public License, version 2.1 (see `LICENSE`). If you use it, please cite the paper.

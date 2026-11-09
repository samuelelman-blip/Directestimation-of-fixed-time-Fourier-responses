# phasegrad_trotter: Trotter layers by addition into a phase gradient

Code, data and manuscript for the paper *Trotter layers by addition into a phase gradient*
(S. J. Elman). The question is whether a catalytic phase-gradient register makes a Trotter step
cheaper than synthesising each small rotation on its own. For one rotation it does not; for a
layer it does. The angles of a layer are written in signed binary digits, the digits are reduced
column by column with full adders (one Toffoli per bit removed), and the bit left in each column
is added into the gradient register once. A rotation then costs about four T gates per nonzero
digit of its angle, so its cost falls with the angle, and randomised rounding to the grid halves
the number of digits.

| path | |
|---|---|
| `pgt/digits.py` | the grid, non-adjacent forms, nearest, randomised and sparse-pair rounding |
| `pgt/circuit.py` | the compiler: a layer of rotations (and equal-angle classes) to Clifford gates, T gates and ANDs with measured uncompute, one addition into the gradient per batch |
| `pgt/simulate.py` | exact checks: every system basis state against all 2^b branches of the gradient register, and a state-vector simulation with the uncomputes done by measurement |
| `pgt/bothe.py` | the comparison: the angle-dependent synthesis costs of Bothe et al. (arXiv:2605.31544), their costing script transcribed, falling back to the mixed fallback and mixed diagonal fits of Kliuchnikov, Lauter, Minko, Paetznick and Petit |
| `pgt/costs.py` | error budgets, gridsynth, batched layer costs, the Heisenberg chain |
| `tests/` | the tests |
| `tools/` | the scripts that made the data and the figures |
| `data/` | `gridsynth.json` (gridsynth T counts and the fit), `layer.json` (per-rotation costs), `trotter.json` (the chain) |
| `figures/` | the figures of the paper, PDF and PNG |
| `paper/` | `main.tex`, `refs.bib`, and the compiled `main.pdf` |

## Running it

Python 3.11 or later, with numpy, matplotlib, pytest, and pygridsynth 2.0.0 (only for
`tools/gridsynth_data.py`).

```
pip install -r requirements.txt
python -m pytest tests -q          # about 5 seconds
python tools/gridsynth_data.py     # about 1 minute
python tools/layer_data.py         # about 1.5 minutes
python tools/trotter_data.py       # about 2 minutes
python tools/figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## The numbers in the paper

| in the paper | where it comes from |
|---|---|
| gridsynth uses 3.03 log2(1/eps) + 5.4 T gates, whatever the angle | `data/gridsynth.json`: 40 random angles at each eps = 1e-3 .. 1e-15, and the small-angle runs |
| a layer of 1,024 rotations, batches of at most 256 signed digits, eps = 1e-10: 20.5, 16.2, 7.3, 2.6 T per rotation (sparse pairs) at angles 1, 1e-2, 1e-4, 1e-5, against 22.5, 22.5, 22.5, 18.9 for Bothe et al.'s fallback scheme; 0.09 against 0.28 at 1e-6; theirs is cheaper further down (0.02 against 0.05 at 3e-7) | `data/layer.json`, key `rows`; `figures/fig_layer.pdf` |
| batch sizes: one rotation costs 128 T (nearest) against gridsynth's 106; B = 16 gives 46.8 and 21.8 | `data/layer.json`, key `batches`; `figures/fig_batch.pdf` |
| the chains: the gradient layers use 0.16 to 0.88 times the T gates of Bothe et al.'s fallback scheme over both chains, both total errors (1e-3, 1e-1) and steps 0.3 to 1e-6; nearest rounding 0.24 to 0.51 times gridsynth; at most 589 ancillas (randomised) | `data/trotter.json`; `figures/fig_trotter.pdf` |
| the bound of the layer theorem holds on every layer counted | asserted in `tools/layer_data.py` and `tests/test_circuit.py` |

The synthesis costs of Bothe et al. are their formulas (the costing script of their Appendix A
and their Eq. 15), evaluated on our angles, not runs of their method. Their worst-case terms are
the Clifford+T mixed fallback and mixed diagonal fits of Kliuchnikov et al. (Quantum 7, 1208,
Table 1).

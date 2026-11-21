# phasegrad_trotter: randomised rounding onto a phase gradient

Code, data and manuscript for the paper *Randomised rounding onto a phase gradient*
(S. J. Elman). A rotation can be applied by adding an integer into a catalytic phase-gradient
register, and a layer of rotations can share one addition. The grid is usually fine enough that
the nearest grid point is within the error. Choosing instead between the two neighbouring grid
points at random, with the probability that cancels the coherent error, leaves the rotation
followed by pure dephasing, exactly, so half the bits suffice; and since each rotation in a
shared addition costs about four T gates per nonzero signed digit of its integer, that halves the
cost of large rotations and makes the cost of every rotation depend on theta^2/eps. Rounding onto
a sparse pair of grid points gives the theta^2/eps small-angle scaling of Bothe et al., and a
linear program over grid points (as Kliuchnikov et al. do for synthesis) finds the cheapest
mixture, of at most three points, saving a further 1 to 12 per cent. The layer
circuit itself (compute the weighted sum, add once, uncompute) assembles known parts: Sanders et
al., Wu et al., Gidney's Hamming-weight tree.

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

Python 3.11 or later, with numpy, scipy, matplotlib, pytest, and pygridsynth 2.0.0 (only for
`tools/gridsynth_data.py`).

```
pip install -r requirements.txt
python -m pytest tests -q          # about 5 seconds
python tools/gridsynth_data.py     # about 1 minute
python tools/layer_data.py         # about 7 minutes
python tools/trotter_data.py       # about 5 minutes
python tools/figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## The numbers in the paper

| in the paper | where it comes from |
|---|---|
| gridsynth uses 3.03 log2(1/eps) + 5.4 T gates, whatever the angle | `data/gridsynth.json`: 40 random angles at each eps = 1e-3 .. 1e-15, and the small-angle runs |
| a layer of 1,024 rotations, batches of at most 256 signed digits, eps = 1e-10: 20.2, 16.1, 6.9, 2.3 T per rotation (three-point mixtures) at angles 1, 1e-2, 1e-4, 1e-5, against 22.5, 22.5, 22.5, 18.8 for Bothe et al.'s fallback scheme; theirs is cheaper further down (0.02 against 0.05 at 3e-7) | `data/layer.json`, key `rows`; `figures/fig_layer.pdf` |
| batch sizes: one rotation costs 128 T (nearest) against gridsynth's 106; B = 16 gives 46.8 and 21.8 | `data/layer.json`, key `batches`; `figures/fig_batch.pdf` |
| the chains: the gradient layers (three-point mixtures) use 0.15 to 0.85 times the T gates of Bothe et al.'s fallback scheme over both chains, both total errors (1e-3, 1e-1) and steps 0.3 to 1e-6; nearest rounding 0.24 to 0.51 times gridsynth; at most 547 ancillas | `data/trotter.json`; `figures/fig_trotter.pdf` |
| the bound of the layer theorem holds on every layer counted | asserted in `tools/layer_data.py` and `tests/test_circuit.py` |

The synthesis costs of Bothe et al. are their formulas (the costing script of their Appendix A
and their Eq. 15), evaluated on our angles, not runs of their method. Their worst-case terms are
the Clifford+T mixed fallback and mixed diagonal fits of Kliuchnikov et al. (Quantum 7, 1208,
Table 1).

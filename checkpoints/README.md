# Trained CRAM models

The trained weights of every CRAM model in the paper's Table 2 (Sudoku-Extreme, Maze-Unique), Table 3 (N-Queens, graph coloring) and Maze-Hard appendix.
Table numbers are those of arXiv 2611.17309.
CRAM is a pairwise energy `E(x) = sum_(i<j) x_i^T K_ij x_j + sum_i B_i^T x_i` over one-hot cell states `x_i`; the pair costs **`K_ij`** (a matrix per cell pair) and the input costs **`B_i`** (a vector per cell) are learned.
Each file is a byte-for-byte copy of a run file under `experiments/runs/`, which is not tracked.
`manifest.json` gives the run folder and the SHA-256 of each file; `scripts/collect_checkpoints.py` writes it.
`manifest.json` also holds the recorded score of each model, which `checkpoints/evaluate.py` reproduces.

## Load

```python
from checkpoints.load import load, names
names()                                  # the seven model names
w = load('maze-unique', seed=0)          # dict: tensor name -> numpy array
w = load('graph-coloring-8')             # a model with one fit takes no seed
```

`uv run python checkpoints/load.py` checks every file against its SHA-256 and prints each model's tensors.
The loader needs only numpy and torch.

## Models

| Name | Fits | Weights file | Parameters | Paper |
|---|---|---|---|---|
| `sudoku-extreme` | seeds 0, 1, 2 | `model.pt` | 263,169 learned; 7,716, 7,750 and 7,761 nonzero | Table 2 |
| `maze-unique` | seeds 0, 1, 2 | `model.pt` | 18 | Table 2 |
| `nqueens-8` | seeds 0, 1, 2 | `model.pt` | 113 | Table 3 |
| `nqueens-10` | seeds 0, 1, 2 | `model.pt` | 181 | Table 3 |
| `graph-coloring-8` | one | `model.npz` | 7,332 learned; 869 nonzero | Table 3 |
| `graph-coloring-10` | one | `model.npz` | 18,660 learned; 475 nonzero | Table 3 |
| `maze-hard` | seeds 0, 1, 2 | `model.pt` | 29 | Maze-Hard appendix |

The N-Queens and graph-coloring fits are deterministic.
The three N-Queens seed folders therefore hold equal weights, and graph coloring has one model; the three seeds of Table 3 for these tasks are seeds of the search.
Each folder also holds the run's `final.json` (and `config.json`, `fit.json` or `model.json` where the run wrote one), with the recipe and the fit timings.

## Tensors

Sudoku-Extreme (`experiments/sudoku.py`, `cram/cram_sudoku_general.py`); 81 cells, 9 digits, 3,240 unordered cell pairs:
- `W` (3240, 9, 9): the pair cost table of each cell pair. `B` (81, 9): the input cost of each cell. The paper's model is `W` and `B`.
- `Wq`, `Bq`: a compressed copy that the script also writes. The paper does not use it.

Maze-Unique (`experiments/maze_unique.py`, model class `cram.cram_maze.CRAMMazeFreeEnergy`); 900 cells, 2 states:
- `T` (4, 2, 2): one 2 x 2 cost table for each of the 4 neighbor offsets of a 3 x 3 window that are not mirror images of each other, shared by all cells. `b` (2,): the input cost, shared by all cells.
- To get the model object, from `experiments/`: `m = CRAMMazeFreeEnergy(window=1); m.load_state_dict(torch.load(path))`, or `maze_unique.load_run`.

N-Queens (`experiments/nqueens_offset.py`, `cram/cram_nqueens.py`); `n*n` cells, 2 states:
- `A` (112 at 8 x 8, 180 at 10 x 10): one learned cost per relative position of two cells. `b` (1,): the learned input cost. These are the 113 and 181 parameters.
- `W` (pairs, 2, 2) and `B` (cells, 2): the pair and input cost tables that `A` and `b` expand to. `S`: `W` as a (cells, cells, 2, 2) tensor.
- `tau`, `unit`, `delta`: constants of the search, computed from the training solutions at the end of the fit (the stopping threshold, the temperature unit and the local stopping margin).
- To get the dict that the search takes, from `experiments/`: `cram_nqueens.load_model(path)`.

Graph coloring (`experiments/graph_coloring.py`, `cram/cram_graph_coloring.py`); `n` vertices, 3 colors, `p = n(n-1)/2` vertex pairs:
- `W0` (p, 3, 3): the base cost table of each vertex pair. `D` (p, p, 3, 3): the cost table that each possible edge adds to each pair. `B` (n, 3): the input cost of each vertex. The cost table of a pair is its `W0` entry plus the `D` entries of the edges in the input graph.
- `tpair` (p, 2), `tvert` (n,): stopping thresholds, computed from the training solutions at the end of the fit.

Maze-Hard (`experiments/maze_hard_latent.py`, model class `cram.cram_maze_hard_graph.GraphLatentEnergyMU`); 900 cells, 2 states, one hidden value per open cell:
- `T` (4, 2, 2) and `b` (2,): the Maze-Unique energy (its 3 x 3 kernel and input cost, 18 values). `mu` (1,): the log of the hidden layer's mass. `beta` (5,): the attention score of the cell itself and of its 4 side neighbours. `kappa` (5,): the pair cost that each attention weight adds. These are the 29 parameters. `k` (2,) is zero and unused.
- To get the model object, from `experiments/`: `m = maze_hard_latent.build(maze_hard_latent.Recipe(**json.load(open('final.json'))['recipe'])); m.load_state_dict(torch.load(path))`.

## Evaluate

`checkpoints/evaluate.py` scores each model on its test set and compares the score with the recorded one in `manifest.json`.
It builds each model from its checkpoint as the task's evaluation script does, and calls the same search with the same settings and seeds:

| Model | Built and searched as in | Search | Compared with |
|---|---|---|---|
| `sudoku-extreme` | `experiments/sudoku.py`, `search` | focused Gibbs search, 4 chains, fitted stopping threshold | `experiments/runs/CRAM-S/full/dev/` and `test/`, `cram0-hb-fit-s<seed>.json` |
| `maze-unique` | `experiments/maze_unique.py`, `score` | focused Gibbs search at temperature 0 | `score-test-gibbs.json` of the run |
| `nqueens-8`, `nqueens-10` | `experiments/nqueens_offset.py`, `run_final` | 20 chains per test row, local stopping test | `final.json` of the run |
| `graph-coloring-8`, `-10` | `experiments/graph_coloring.py`, `test` | 20 samples per test graph | `final.json` of each search seed |
| `maze-hard` | `experiments/maze_hard_latent.py`, `score` | hidden layer settled, then damped descent | `test-scores.json` of the run |

Run it on one GPU, from the repository root:

```sh
uv run python checkpoints/evaluate.py --seeds 0 1 2
```

- Maze-Unique, Maze-Hard, N-Queens and graph coloring score their whole test set, and each score must equal its record exactly.
- Sudoku-Extreme has 412,786 test puzzles (about 1.5 GPU hours for each seed).
  By default the script scores the 10,000 development puzzles (about 5 minutes for each seed), and the score must equal the development record in every field, the mean operation count included.
  `--sudoku-set test --sudoku-n N` scores the first N test puzzles.
  The search draws its random numbers for the whole batch, so a batch that differs from the recorded run takes other draws; for N below 412,786 the script compares only the solve rate.
- The datasets are read from `data/`, which `scripts/prepare_data.py` writes (`REPRODUCING.md`, "Data").


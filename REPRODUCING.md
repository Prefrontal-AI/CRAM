# Reproducing the results of CRAM

The full instructions: layout, setup, data, the trained models, the commands behind each table and figure of arXiv 2611.17309, and what was checked.
`README.md` has the short version; `AGENTS.md` has notes for coding agents.

## Layout

```
cram/           the library: energies, losses, search kernels, data loaders, operation counts
experiments/    one script per experiment; each writes under experiments/runs/
  recipes/      the settings that validation selected, so that a final run needs no tuning run first
  runs/table/   one JSON record per reported CRAM table row (tracked)
checkpoints/    the trained CRAM models of the paper, a loader and an evaluator
baselines/      wrappers around the authors' code of each baseline; see baselines/README.md
  records/      one JSON record per timed baseline fit (tracked)
results/        the exported numbers that the paper's tables and figures read
scripts/        prepare_data.py and the Sudoku split indices
slurm/          job scripts
tests/          pytest tests
```

Where to read the method:

| To inspect | Read |
| --- | --- |
| The Sudoku-Extreme energy, loss, augmentation and fit | `experiments/sudoku.py` (top docstring, `train`, `prepare`) |
| The Sudoku-Extreme search and its stopping test | `cram/cram_sudoku_general.py`, `cram/cram_sudoku_search.py` |
| The Maze-Unique energy and loss | `cram/cram_maze.py` (`CRAMMazeFreeEnergy`), `cram/maze_unique.py` (`pl_loss`, `gibbs`) |
| The Maze-Hard hidden layer and its adjoint | `cram/cram_maze_hard_graph.py`, `experiments/maze_hard_latent.py` |
| N-Queens and graph coloring | `cram/cram_nqueens.py`, `cram/cram_graph_coloring.py` |
| Data loading and splits | `scripts/prepare_data.py`, `cram/cram_sudoku_search.py` (`load_puzzles`), `cram/maze_unique.py`, `cram/maze_learned.py`, `cram/nqueens_data.py`, `cram/graph_coloring_data.py` |
| Evaluation of the trained models | `checkpoints/evaluate.py` |
| The operation-count rule | `baselines/flops/fullcount.py`, `cram/opcount.py` |
| The fit-time clock | `baselines/clock.py` |

Every module starts with a docstring that states its equation, its inputs and the meaning of its dimension names.

## Setup

```sh
uv sync                                  # creates .venv/ with torch 2.11 (CUDA 12.8) and installs cram/
uv run pytest                            # 91 tests, about 2 minutes on one GPU
```

The Sudoku search kernels are written in Triton and need an NVIDIA GPU.
The reported times are from one NVIDIA H200 and, for the CPU fit times, one CPU thread.
On a Slurm cluster, run every command of this file inside a job, not on a login node.

## Data

```sh
uv run python scripts/prepare_data.py sudoku        # Sudoku-Extreme: sapientinc/sudoku-extreme at 58942f96
uv run python scripts/prepare_data.py maze-unique   # Maze-Unique: locuslab/EqR-data at bcb7d45c
uv run python scripts/prepare_data.py maze-hard     # Maze-Hard: sapientinc/maze-30x30-hard-1k at 549754de
uv run python scripts/prepare_data.py multi-solution --gram-src third_party/gram   # N-Queens, graph coloring
```

The files go to `data/` (set `CRAM_DATA` for another folder).
The Sudoku command needs about 32 GB of memory and writes 2 GB.
`scripts/sudoku_extreme_splits.json` names the official rows of every Sudoku split, and the script checks a hash of each split.
The docstring of `scripts/prepare_data.py` describes the splits.
N-Queens and graph coloring come from GRAM's dataset builders: `baselines/fetch.sh gram` clones `https://github.com/ahn-ml/gram-release` at commit `f2e32b9` into `third_party/gram`.

## Use the trained models

`checkpoints/` holds the trained weights of every CRAM model in the paper (50 files, 7 MB, in git; no download is needed):

| Name | Task | Fits | Parameters |
| --- | --- | --- | --- |
| `sudoku-extreme` | Sudoku-Extreme (Table 2) | seeds 0, 1, 2 | 263,169 learned; about 7,700 nonzero |
| `maze-unique` | Maze-Unique (Table 2) | seeds 0, 1, 2 | 18 |
| `nqueens-8`, `nqueens-10` | N-Queens (Table 3) | seeds 0, 1, 2 | 113 and 181 |
| `graph-coloring-8`, `graph-coloring-10` | graph coloring (Table 3) | one fit each | 7,332 and 18,660 learned |
| `maze-hard` | Maze-Hard (appendix) | seeds 0, 1, 2 | 29 |

Load the weights as numpy arrays (needs only numpy and torch), from the repository root:

```python
from checkpoints.load import load, names

names()                                   # the seven model names
weights = load('sudoku-extreme', seed=0)  # dict: tensor name -> numpy array
weights['W'].shape                        # (3240, 9, 9): one 9 x 9 cost table per pair of cells
weights['B'].shape                        # (81, 9): one input cost per cell and digit
```

Rebuild a model and solve puzzles with the paper's search (one GPU; needs the dataset, see "Data"):

```python
import json, sys
import torch
sys.path.insert(0, 'experiments')         # the run scripts are imported by name
import maze_unique as U
from checkpoints.load import path

config = U.Train(**json.load(open('checkpoints/maze-unique/seed0/config.json')))
model = U.make_model(config)              # the 18-parameter energy, cram.cram_maze.CRAMMazeFreeEnergy
model.load_state_dict(torch.load(path('maze-unique', seed=0), map_location='cpu'))
result = U.score(model, 'test', U.Search(search='gibbs', T=0.0))   # focused Gibbs at temperature 0
print(result['solved'], 'of', result['n'], 'test mazes solved')   # 1000 of 1000
```

- `uv run python checkpoints/load.py` checks every file against its SHA-256 and prints the tensors of each model.
- `checkpoints/README.md` gives the meaning of every tensor and how to rebuild each of the other models.
- `checkpoints/evaluate.py` holds one function per task that rebuilds the model and runs its search (`sudoku`, `maze`, `nqueens`, `graph_coloring`, `maze_hard`); copy from it to run a model on other inputs.

## Check the results in ten minutes

This command rebuilds each model from its weights, runs the paper's search on the test set, and compares the score with the recorded one:

```sh
uv run python checkpoints/evaluate.py --seeds 0 1 2      # one GPU
```

It must end with `every score matches its record`.
Maze-Unique, Maze-Hard, N-Queens and graph coloring score their whole test sets.
Sudoku-Extreme scores the 10,000 development puzzles by default (about 5 minutes per seed); `--sudoku-set test --sudoku-n 412786` scores the whole test set (about 1.5 GPU hours per seed).
`checkpoints/README.md` describes the tensors of each model.

Training is also fast enough to check directly. From `experiments/`:

```sh
python sudoku.py diag                      # fits Sudoku-Extreme on the three training splits
M="--kernel free --free-init 0 --opt sgd --momentum 0.5 --cosine --lr 1 --updates 20 --l1 0"
python maze_unique.py train $M --split train --seed 0 --out runs/check/fg-s0 --score
```

On an NVIDIA H200, the fitted `model.pt` files are byte-identical to `checkpoints/sudoku-extreme/seed<k>/model.pt` and `checkpoints/maze-unique/seed0/model.pt` (compare with `sha256sum`).
The Sudoku fit draws its digit relabelings from the CUDA random generator, so another GPU type gives another model of the same quality.

## Reproducing the results

Table and figure numbers are those of arXiv 2611.17309: Table 2 is the main table (Sudoku-Extreme and Maze-Unique), and Table 3 is N-Queens and graph coloring.

Run every command from `experiments/` with the repository's Python (`uv run python`, or `../.venv/bin/python`).
On a Slurm cluster, `sbatch slurm/gpu.sbatch "<commands>"` and `sbatch slurm/cpu.sbatch "<commands>"`, submitted from the repository root, run a command line from `experiments/` on one GPU or one CPU thread.
A stage whose output exists is skipped, so a job that stops can be submitted again.
Each script's docstring states its energy, its training rule, its search and its options.

Seeds: `--split` or `--seed` takes 0, 1 and 2; the paper reports the mean over the three.
Every setting was chosen on development data, and the test set is scored once per final run.
The Sudoku, hand-set and Maze-Hard scripts refuse a test run until its development run or its frozen recipe exists.
Run-folder names such as `CRAM-S`, `CU13`, `NQ/nq11`, `ALT` or `CRAM-HE` are only labels; the tracked records name them, so the commands keep them.

### Table 2: Sudoku-Extreme

```sh
python sudoku.py diag                      # fits the energy on train-s0, s1, s2; writes runs/CRAM-S/full/full-aug-s<k>/model.pt
S="--scale hbar --hi 0.40 --lo-t 0.31 --theta 0.45 --stop fit"
python sudoku.py search --split 0 $S --data dev10k --out runs/CRAM-S/full/dev/cram0-hb-fit-s0.json   # 10,000 development puzzles
touch runs/CRAM-S/full/test-approved       # states that the settings are final
python sudoku.py final --split 0 $S        # the 412,786 test puzzles, once
python sudoku.py time --split 0 --device cuda --stop fit    # cold and warm fit time, fresh process
python sudoku.py time --split 0 --device cpu --stop fit     # on one CPU thread (slurm/cpu.sbatch)
python sudoku.py rows --stop fit           # runs/table/sudoku-extreme__cram-0-fit.json
```

Expected: the three fitted models hold 7,716, 7,750 and 7,761 nonzero costs, and solve 412,768, 412,770 and 412,769 of the 412,786 test puzzles.

### Table 2: Maze-Unique

One 3x3 kernel of free 2x2 cost tables (18 parameters), fitted with 20 SGD updates on batches of 64 mazes (momentum 0.5, cosine decay, learning rate 1, no L1).

```sh
M="--kernel free --free-init 0 --opt sgd --momentum 0.5 --cosine --lr 1 --updates 20 --l1 0"
python maze_unique.py train $M --split train --seed 0 --out runs/CU13/final/fg-s0 --score
python maze_unique.py score --run runs/CU13/final/fg-s0 --split test --search gibbs      # 1,000 test mazes
python maze_unique.py time $M --split train --seed 0 --out runs/CU13/final/fg-s0         # update loop on the GPU (gpu.json)
python maze_unique.py cpu $M --split train --seed 0 --out runs/CU13/final/fg-s0          # the same on one CPU thread (cpu.json)
python maze_unique_fit_time.py run --row maze-unique__cram-sgd-fg --seed 0 --device cuda # the complete fit: cold, then 7 warm
python maze_unique_fit_time.py run --row maze-unique__cram-sgd-fg --seed 0 --device cpu
python maze_unique.py rows --rows cram-sgd-fg
python maze_unique_fit_time.py summary --rows maze-unique__cram-sgd-fg --write-records
```

Expected: 1,000 of 1,000 test mazes for every seed.
The learning rate and the update count were chosen on the 100 development mazes (`--split train900`, scored on `dev100`).

### Table 2: hand-set energies

```sh
python hand_set.py sudoku --data dev --seed 0 && python hand_set.py sudoku --data test --seed 0
python hand_set.py maze --split dev100 --seed 0 && python hand_set.py maze --split test --seed 0
python hand_set.py rows                    # the two table records and ../results/hand-energy.json
```

### Table 3: N-Queens and graph coloring

```sh
# N-Queens, offset-table energy. The 10x10 recipe is selected on validation and reused unchanged at 8x8.
for method in pl mpl mpf dsm spl; do for arm in table ising; do
  python nqueens_offset.py --mode tune --n 10 --method $method --arm $arm; done; done
python nqueens_offset.py --mode select --n 10          # writes runs/NQ/nq11/n10/selected-headline.json
python nqueens_offset.py --mode final --n 10 --method pl --arm ising --seed 0
python nqueens_offset.py --mode shared --n 8 --seed 0
python nqueens_offset.py --mode retime --n 10 --stage final --method pl --arm ising --seed 0   # the reported fit times
python nqueens_offset.py --mode retime --n 8 --stage shared --method pl --arm ising --seed 0

# N-Queens, one free table per cell pair (a row of results/multi-solution-table.json), and the hand-set energy
python nqueens_free.py --mode tune --n 8 --method pl && python nqueens_free.py --mode select --n 8
python nqueens_free.py --mode final --n 8 --method pl --seed 0
python nqueens_free.py --mode hand --n 8 --split dev && python nqueens_free.py --mode hand --n 8 --split test --seed 0

# Graph coloring (n = 8 and 10 vertices)
python graph_coloring.py --mode screen --n 8 --arm full
python graph_coloring.py --mode fit --n 8 --arm full --device cuda     # the timed fit
python graph_coloring.py --mode fit --n 8 --arm full --device cpu      # the model that the test scores
python graph_coloring.py --mode test --n 8 --arm full --stop terms --temperature 2   # inference seeds 0, 1, 2; n = 10: --temperature 3
python graph_coloring_hand_set.py --mode dev --n 8 && python graph_coloring_hand_set.py --mode test --n 8 --seed 0

# Exports
python nqueens_export.py && python nqueens_free_export.py && python nqueens_offset_export.py
python graph_coloring_export.py && python multi_solution_export.py     # ../results/multi-solution-table.json
```

To skip the tuning runs, copy the selected settings first:
`cp recipes/nqueens-offset-n10.json runs/NQ/nq11/n10/selected-headline.json` and `cp recipes/graph-coloring.json runs/CRAM-GC/selected.json`.
The N-Queens and graph-coloring fits are full-batch and deterministic: the seed changes only the search.
The baseline columns of this table need the baseline runs (`baselines/README.md`).

### Maze-Hard (appendix)

The Maze-Unique energy with one hidden layer on the open cells (29 parameters).
The hidden layer settles in its own relaxation, and its parameter gradient comes from the adjoint at the fixed point: a second relaxation of the same linear system with the loss gradient as its source.
Training never differentiates through either relaxation.

```sh
mkdir -p runs/CRAM-HE && cp recipes/maze-hard-latent.json runs/CRAM-HE/selected-CHE8-mu.json
python maze_hard_latent.py fit --name CHE8-mu --seed 0 --warm 0 --triton     # about 30 s per fit; writes runs/CRAM-HE/final-triton/CHE8-mu/
python maze_hard_latent.py score --name CHE8-mu --final-dir final-triton --row cram-latent-zmu --no-require-warm   # the 1,000 test mazes, once, after seeds 0-2
python maze_hard_figure_export.py --maze 0 --snapshots 0 16 18 32 64         # ../results/maze-hard-figure.json
```

Expected: 999 of 1,000 test mazes for every seed.
The paper's fit time for this model (25.6 s) is one cold fit of seed 0 (`--warm 0`); `--warm 7` adds seven warm repeats, which `score` requires unless `--no-require-warm` is given.

### Training methods (Figure 4 and its appendix table)

`training_methods.py` implements the twelve methods on the Sudoku and Maze-Unique energies; `training_curves.py` runs the 20-seed curves.
The study needs the three Table 2 Sudoku fits (`python sudoku.py diag`), because its search takes its temperature unit from them.
The job scripts in `slurm/training_methods/` run the stages on three GPUs, submitted from the repository root:

```sh
sbatch --array=0-9 slurm/training_methods/tune.sbatch        # Sudoku: each method's grid, then its selection
sbatch --array=0-9 slurm/training_methods/final.sbatch       # the selected settings on seeds 0-2
sbatch slurm/training_methods/retest.sbatch                  # test at the search cap of Table 2
sbatch --array=0-4 slurm/training_methods/dense.sbatch       # the figure's methods with dense probes
sbatch --array=0-7 slurm/training_methods/many.sbatch train  # seeds 3-19; then the same with `probe`
sbatch --array=0-11 slurm/training_methods/sgd.sbatch tune   # Maze-Unique with SGD; then `final`
python training_methods.py export && python training_methods_export.py
python training_curves.py export && python training_curves.py export-table && python training_curves.py export-long
```

The header of each job script states its stage and its array indices.
Two environment variables select the Maze-Unique variant: `ALT_MAZE_KERNEL` (`free` for the 18-parameter kernel) and `ALT_MAZE_OPT` (`sgd-cosine` for the optimizer of Table 2).
The appendix table's Maze-Unique numbers (`alt-training.json`, `alt-t99.json`) are from `ALT_MAZE_KERNEL=free ALT_MAZE_OPT=sgd-cosine` (run folder `runs/ALT-sgd`).
The figure's Maze-Unique curves (`alt-dense.json`, `alt-many.json`) are from `ALT_MAZE_KERNEL=free` with the Adam optimizer (run folder `runs/ALT-free`); each file names its run folders in its `roots` field.
The reported training time to 99% is the fit time, less the one-time start-up of the first update, plus the time to fit the stopping threshold (`ready_s` in the exports).

### Data efficiency (Figure 3b)

```sh
export DE_VARIANTS=99 CRAM_E_ROOT=DE2-v99
python data_efficiency.py fit --k 30 --seed 0 && python data_efficiency.py search --k 30 --seed 0 --data test10k
python data_efficiency.py export
```

The training rows are the ones that TRM's dataset builder stores (each puzzle and its first 99 stored symmetry variants); `baselines/trm/trm_smalldata.py` writes them, and the baselines train on the same rows.

### Operation counts and figure data

```sh
cd ../baselines/flops
python cram_kernels.py ../runs/flops-full/cram-kernels.json    # counts of the Sudoku search kernels
python cram_kernels_extra.py sgd-maze ../runs/flops-full       # groups: hand-sudoku, hand-maze, nqueens, gc, maze-hard-zmu, sgd-maze
python fc2_export.py                                           # ../../results/op-counts.json (reads every baseline's count too)
cd ../../experiments
python compute_curve_export.py                                 # ../results/cram-anytime/sudoku-extreme.json (compute figure)
python difficulty_export.py                                    # ../results/difficulty/sudoku-difficulty.json (reads baseline records too)
```

A count covers matrix products, elementwise arithmetic and transcendental functions; indexing, comparisons and memory moves are not counted (`baselines/flops/fullcount.py`).
`difficulty_export.py` reads the Sudoku test records of the zero-energy stopping test (`python sudoku.py final --split <k> --scale hbar --hi 0.40 --lo-t 0.31 --theta 0.45 --stop zero`), not the fitted-threshold records of Table 2.

## Which file holds which result

| Paper | File | Written by |
| --- | --- | --- |
| Table 2, CRAM rows | `experiments/runs/table/*.json` | `sudoku.py rows`, `maze_unique.py rows`, `hand_set.py rows` |
| Table 2, baseline fit times and scores | `results/baseline-fit-times.json`, `baselines/records/` | `baselines/collect.py` |
| Operation counts in every table | `results/op-counts.json`, `results/flop-counts.json` | `baselines/flops/fc2_export.py`, `baselines/flops/report.py` |
| Table 3 | `results/multi-solution-table.json`, `nqueens-results.json`, `graph-coloring-results.json`, `nqueens-coverage-ebr.json`, `ebr-comparison.json` | `multi_solution_export.py` and the exports above; `baselines/ebr/comparison.py` |
| Compute figure | `results/cram-anytime/`, `results/compute-sweep/` | `compute_curve_export.py`, `baselines/sweep/build.py` |
| Learned-weights figure | `checkpoints/sudoku-extreme/`, `checkpoints/maze-unique/` | the fits above |
| Difficulty figure | `results/difficulty/sudoku-difficulty.json` | `difficulty_export.py` |
| Data-efficiency figure | `results/sudoku-data-eff-shared-v99.json`, `sudoku-data-eff-baselines-v99.json` | `data_efficiency.py export`, `data_efficiency_baselines.py` |
| Training methods | `results/alt-training/` | `training_methods.py export`, `training_curves.py export*`, `training_methods_export.py` |
| Halting of looped models | `results/halting-full.json`, `looped-dynamics.json`, `ebr-halting.json` | `baselines/dynamics/export_halting_full.py`, `baselines/dynamics/export.py`, `baselines/ebr/halting.py` |
| TRMP lattice model (footnote of Table 2) | `results/trmp-lattice-sudoku.json` | `baselines/trmp/lattice_export.py` |
| Maze-Hard table and figure | `experiments/runs/table/maze-hard__cram-latent-zmu.json`, `results/maze-hard-figure.json` | `maze_hard_latent.py score`, `maze_hard_figure_export.py` |

## What was checked from this layout

The experiments ran in an earlier private layout of this code; the files were then moved, cleaned and documented.
On one NVIDIA H200, from this layout:

- `pytest`: 91 tests pass.
- `checkpoints/evaluate.py --seeds 0 1 2`: every score equals its record, for all seven models (Sudoku-Extreme on the 10,000 development puzzles).
- Fresh fits of Sudoku-Extreme (splits 0, 1, 2) and Maze-Unique (seeds 0, 1, 2) give `model.pt` files that are byte-identical to the checkpoints.
- A fresh Maze-Hard fit agrees with its checkpoint to about 1e-7 relative in every parameter, not bit for bit: the fused Triton kernels of this fit are not bitwise repeatable between runs, also for the original code.
- The export scripts, run on the existing run folders, write the same numbers as the tracked files in `results/`, `experiments/runs/table/` and `baselines/records/`. Two exporters were not rerun: `baselines/flops/report.py` and `baselines/sweep/build.py`.
- The tracked `results/maze-hard-figure.json` was written with `--snapshots 0 16 18 32 64`.

Not run again from this layout: the 412,786-puzzle Sudoku test, the tuning and selection stages, the training-method and data-efficiency studies, and every baseline fit.
The baseline wrappers compile, but no baseline was trained again; `baselines/README.md` states what each wrapper needs.
The FRM fits need an adapter from an earlier campaign folder (`FRM_CAMPAIGNS`) and cannot be repeated from this repository alone.


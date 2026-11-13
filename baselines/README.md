# Baselines

Wrappers that train, score and count each baseline with its authors' code.
Each wrapper adds only a fit clock (`clock.py`), fixed data, and a scorer on the paper's test puzzles.
The wrappers do not hold the authors' code.
`baselines/fetch.sh` clones every repository at the commit the paper used into `third_party/`; every repository is public, and none needs a source patch.

These scripts were moved into this layout from the layout the experiments ran in, and their paths were rewritten.
They compile, but no baseline has been trained again from this layout.
`records/` holds the record of every timed fit that the paper reports.

## Locations

| Variable | Default | Holds |
| --- | --- | --- |
| `THIRD_PARTY` | `<repository>/third_party` | the clones and Python environments below |
| `CRAM_DATA` | `<repository>/data` | datasets from `scripts/prepare_data.py` |
| `V1_DATA` | `<repository>/data` | the same folder as `CRAM_DATA`; the scorers of the unseen Sudoku-Extreme puzzles read this name |
| per-baseline variables (`HRM_SRC`, `HRM_PY`, `TRM_SRC`, `TRM_PY`, `EQR_SRC`, `EBR_SRC`, `GRAM_SRC`, `FPRM_SRC`, `FRM_PY`, ...) | under `$THIRD_PARTY`, set at the top of each job script | one clone or interpreter |
| `EBR_PY`, `EBR_DATA` | `experiments/runs/ebr/.venv/bin/python`, `experiments/runs/ebr/data` | EBR's interpreter and the datasets that `ebr/build_data.sh` writes |
| `FRM_CAMPAIGNS` | none | the earlier FRM campaign folder that `frm/setup.sh` copies from (see below) |

Run outputs go to `baselines/runs/` and, for datasets that a baseline's builder writes, to `experiments/runs/<baseline>/`.
Both folders are ignored by git.
Submit every job script from `baselines/` (`sbatch trm/fit.sbatch sudoku 0`).
The job scripts name no Slurm partition; the cluster's default partition is used.
Each baseline runs with its own Python environment (for example `$THIRD_PARTY/hrm/.venv`); the top of each job script names the interpreter it expects.

## Baselines and their code

`fetch.sh` holds the full commit hashes.

| Baseline | Authors' code | Commit | Clone in `$THIRD_PARTY` |
| --- | --- | --- | --- |
| HRM (Hierarchical Reasoning Model) | `https://github.com/sapientinc/HRM` | `ac15626` | `hrm/HRM` |
| TRM (Tiny Recursive Model) | `https://github.com/SamsungSAILMontreal/TinyRecursiveModels` | `c011037` | `trm` |
| GRAM | `https://github.com/ahn-ml/gram-release` | `f2e32b9` | `gram` |
| EqR | `https://github.com/locuslab/EqR` | `134c754` | `eqr` |
| FRM (Flow Reasoning Model) | `https://github.com/helblazer811/Flow-Reasoning-Models` | `6149a01` | `frm` |
| EBR | `https://github.com/wwillchen/ebr` | `8c5271c` | `ebr` |
| FPRM | `https://github.com/nilskiKonjIzDunava/fprm` | `b291252` | `fprm/fprm` |
| TRMP | none released; `trmp/` is a reimplementation from the paper (arXiv 2610.18649) | | |

## Fit, score and count (Tables 2 and 3)

A fit is one complete training run under the fit clock.
A score is the accuracy of the fitted model on the test puzzles.
A count is the number of arithmetic operations of one inference, measured by `flops/fullcount.py`.

| Baseline | Fit | Score | Operation count |
| --- | --- | --- | --- |
| HRM | `HRM_EPOCHS=40000 sbatch hrm/fit.sbatch sudoku <seed>`; `sbatch hrm/fit.sbatch mazeunique <seed>` | `hrm/select.sbatch` (checkpoint on development data, then test) | `flops/hrm_count.py`, `flops/hrm_count_mazeunique.sh` |
| TRM | `sbatch trm/fit.sbatch <sudoku\|mazeunique> <seed>` (settings: `trm/trm_configs.sh`) | `trm/trm_bench.py score`, `trm/unseen.py` | `trm/trm_bench.py flops-full` |
| GRAM | `gram/submit.sh` (settings: `gram/configs.sh`) | `gram/submit_score.sh` (`gram/score.sbatch`, `gram/merge.sbatch`); N-Queens and graph coloring: `gram/submit_score_multi.sh` (`gram/score_multi.sbatch`) | `gram/flops.py` |
| EqR | `sbatch eqr/fit.sbatch <sudoku\|maze-repo\|maze-paper> <seed>` | `eqr/eqr_score.py` | `flops/eqr_count.py` |
| FRM | `frm/setup.sh <task> frm-compiled`, then `sbatch frm/fit.sbatch <sudoku\|mazeunique> <seed>` | the adapter's `finish` action; `frm/score_unseen.sbatch` | `flops/frm_count.py` |
| TRMP | `sbatch trmp/fit.sbatch <sudoku-extreme\|maze-unique> <seed>` | `trmp/score.py` | `flops/trmp_count.py` |
| TRMP lattice model (Sudoku-Extreme footnote) | `trmp/lattice_fit.sbatch <seed>`; cold and warm fit times: `trmp/lattice_fit_warm.sbatch <seed>` (`warm.py`) | `trmp/lattice_score.sbatch` | `flops/trmp_lattice_count.py` |
| EBR | `ebr/build_data.sh`, then `ebr/submit.sh` (settings: `ebr/configs.sh`) | `ebr/score.sbatch`, then `ebr/report_fc2.sbatch` | `ebr/flops.py` |
| FPRM (Maze-Hard appendix) | not trained here: the authors' released checkpoint | `fprm/score.sbatch` | `flops/fprm_count.py` |

TRM and HRM on N-Queens and graph coloring train inside GRAM's repository (`gram/configs.sh`).
The GRAM jobs expect the `adam-atan2-pytorch` package in `experiments/runs/gram/pydeps` and GRAM's datasets in `$THIRD_PARTY/cache/gram-data`; no script in this repository builds either.
FRM runs through the adapters `frm/adapters/frm_reproduction.py` (Sudoku-Extreme) and `frm/adapters/frm_maze.py` (Maze-Unique), which call the authors' training and sampling code.
`frm/setup.sh` builds its run folder from an earlier campaign folder (`FRM_CAMPAIGNS`) that holds the adapter's `plan.json` and prepared data.
Building that folder from `fetch.sh` and `scripts/prepare_data.py` is not done yet, so the FRM fits cannot be repeated from this repository alone.
The other baselines of the Maze-Hard appendix table are reported with their published accuracies and are not trained here.

## Shared steps

- `clock.py` is the fit clock.
  It starts when the training data are in host memory and stops when the inference-ready model exists; it excludes evaluation, saving and disk reads.
- `collect.py` copies the fit records from `runs/` into `records/` (one JSON file per method, task and seed, and `records/table.md`) and writes `results/baseline-fit-times.json`.
- `flops/fullcount.py` is the operation counter used for every method.
  `flops/cram_kernels.py` and `flops/cram_kernels_extra.py` count CRAM's own search under the same convention.
  `flops/fc2_export.py` reads every count and writes `results/op-counts.json`.
  `flops/report.py` writes `results/flop-counts.json` (HRM and TRMP, matrix products only), which `sweep/build.py` reads.
- `sweep/` scores each trained checkpoint again at a grid of inference budgets (compute figure, Sudoku-Extreme); `python baselines/sweep/build.py` writes `results/compute-sweep/sudoku-extreme.json`.
- `dynamics/` measures the step-to-step state change and the halting decisions of each looped model; `dynamics/export.py` writes `results/looped-dynamics.json` and `dynamics/export_halting_full.py` writes `results/halting-full.json`.
  `ebr/halting.py` writes `results/ebr-halting.json`.
- `ebr/comparison.py` writes `results/ebr-comparison.json`.
- `trmp/lattice_export.py` writes `results/trmp-lattice-sudoku.json`.
- Data efficiency: TRM with `SRC=$(bash trm/data_efficiency_freeze.sh); sbatch trm/data_efficiency.sbatch "$SRC" <N> <seed>`; EBR with `ebr/de2_submit.sh`; FRM with `frm/de2.sbatch`.

## What was run

Every record in `records/` and every file in `results/` comes from runs in the earlier layout.
After the move, each Python file was checked with `python -m py_compile` and each shell file with `bash -n`.
No fit, score or count was repeated from this layout.

# Notes for coding agents

The facts an agent needs to check or extend this repository without reading every file.
`REPRODUCING.md` has the commands behind each result.

- **Environment.** `uv sync` creates `.venv/`. Use `uv run python` or `.venv/bin/python`. Do not install packages globally.
- **Compute.** Everything except the CPU-only exports needs one NVIDIA GPU. On a cluster, submit jobs (`slurm/gpu.sbatch`, `slurm/cpu.sbatch`); do not run fits, searches or tests on a login node.
- **Working directory.** Run experiment scripts from `experiments/` and baseline scripts from `baselines/`. `checkpoints/evaluate.py`, `scripts/` and `pytest` run from the repository root.
- **Locations.** Datasets: `data/` (`CRAM_DATA`). Clones of the baselines' code: `third_party/` (`THIRD_PARTY`; `baselines/fetch.sh` fills it). Run outputs: `experiments/runs/` and `baselines/runs/`. All four are ignored by git, except `experiments/runs/table/`.
- **First check.** `uv run pytest`, then `uv run python checkpoints/evaluate.py --seeds 0 1 2`. The second command exits with status 0 only if every score equals its record.
- **Ground truth.** A number in the paper comes from a file in the table "Which file holds which result". The records in `experiments/runs/table/` and `baselines/records/` and the files in `results/` are data: do not edit them by hand; rerun the script that writes them.
- **Results must not change by accident.** The Sudoku-Extreme, Maze-Unique, N-Queens and graph-coloring fits are deterministic for a fixed seed and GPU type; the Maze-Hard fit repeats to about 1e-7 relative. After a change to `cram/` or to a fit or search function, refit (`python sudoku.py diag`, `python maze_unique.py train ...`) and compare the `model.pt` hashes with `checkpoints/manifest.json`, then rerun `checkpoints/evaluate.py`.
- **Test sets.** The test split is scored once per final run. Tune on the development splits (`dev10k`, `dev100`, `train900`), never on test data.
- **Training rule.** Training never differentiates through a solver or a relaxation: no backpropagation through inference steps. The Maze-Hard hidden layer uses the adjoint at the fixed point (`experiments/maze_hard_latent.py`).
- **Names.** Run-folder names (`CRAM-S`, `CU13`, `NQ/nq11`, `ALT`, `CRAM-HE`, `CHE8-mu`) and row keys (`cram-0-fit`, `cram-sgd-fg`, `cram-latent-zmu`) are labels that the tracked records use. Do not rename them.
- **Where to start reading.** The top docstring of `experiments/sudoku.py` (energy, loss, augmentation, search), then `cram/cram_maze.py` and `experiments/maze_unique.py`.


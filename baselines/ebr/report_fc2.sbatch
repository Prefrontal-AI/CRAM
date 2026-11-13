#!/bin/bash
#SBATCH --job-name=ebr-report-fc2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# EBR scores with operation counts under the convention of baselines/flops/fullcount.py (all arithmetic).
# Reruns ebr/report.py on the saved score of every benchmark and seed, with the per-component counts that
# `ebr/flops.py --full` wrote to runs/ebr/flops-full/<bench>.json. CPU only; no model runs.
# Writes runs/ebr/<bench>-s<k>/report-fc2.json. A report.json beside it, when present, holds the same scores with a
# count of matrix multiplications only.
# Usage (from baselines/): sbatch ebr/report_fc2.sbatch
set -euo pipefail
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
T=$BASELINES
R=$T/runs/ebr
PY=${EBR_PY:-$T/../experiments/runs/ebr/.venv/bin/python}
for k in 0 1 2; do
  $PY $T/ebr/report.py sudoku $R/sudoku-s$k/score $R/flops-full/sudoku.json $R/sudoku-s$k/report-fc2.json
  $PY $T/ebr/report.py maze $R/mazeunique-s$k/score/maze.npz $R/flops-full/mazeunique.json $R/mazeunique-s$k/report-fc2.json
  for b in nqueens8 nqueens10; do $PY $T/ebr/report.py gram $R/$b-s$k/score/nq.npz $R/flops-full/$b.json $R/$b-s$k/report-fc2.json; done
  for b in gc8 gc10; do $PY $T/ebr/report.py gram $R/$b-s$k/score/gc.npz $R/flops-full/$b.json $R/$b-s$k/report-fc2.json; done
done
echo done

#!/bin/bash
#SBATCH --job-name=ALT-retest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-5
# Training-method study: score the Sudoku test of the final runs of six methods (array index = method) with the
# search cap of 4e11 flop per puzzle, the cap of Table 2. Seeds 0-2 run in parallel, one seed per GPU.
# A run that has test.json is skipped: remove that file to score the run again.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
METHODS=(pl mpl mpf dsm imp tbptt)
method=${METHODS[$SLURM_ARRAY_TASK_ID]}
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python -u training_methods.py test --run runs/ALT/final/sudoku-$method-s$s &
done
wait

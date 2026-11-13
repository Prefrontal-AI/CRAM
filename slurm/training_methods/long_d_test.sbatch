#!/bin/bash
#SBATCH --job-name=ALT-longD-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-2
# Training-method study: test of one long-schedule setting. RUN is the run folder without the seed suffix, for example
#   RUN=runs/ALT/longD/sudoku-bptt5-lr6 sbatch slurm/training_methods/long_d_test.sbatch
# The array index is the seed. The test is the one of the training-method table (training_methods.py test: 50,000
# puzzles, search cap 4e11 flop). A run that has test.json is skipped.
# training_curves.py (LONG_TABLE) lists the settings that the paper reports.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
python -u training_methods.py test --run ${RUN}-s$SLURM_ARRAY_TASK_ID

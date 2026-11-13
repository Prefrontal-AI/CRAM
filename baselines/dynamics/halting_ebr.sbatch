#!/bin/bash
#SBATCH --job-name=ld-halt-ebr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# EBR's gradient-norm stop as a halting test, on all 412,786 Sudoku-Extreme test puzzles (subset `unseen`); from saved records, no model run.
# Usage (from baselines/): sbatch dynamics/halting_ebr.sbatch    (submit_halting_full.sh submits it)
# ebr/halting.py writes to the repository that holds it, so this job runs the live copy; CODE (the frozen copy) records the code state.
set -euo pipefail
REPO=$(realpath ${SLURM_SUBMIT_DIR:?submit from baselines/}/..)
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID $(hostname) code ${CODE:-live}"
cd $REPO && .venv/bin/python baselines/ebr/halting.py
echo "end $(date -u +%FT%T)"

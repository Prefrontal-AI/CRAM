#!/bin/bash
#SBATCH --job-name=cram
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Run one command line on one GPU, from experiments/, with the repository's Python as `python`.
# Usage (from the repository root): sbatch slurm/gpu.sbatch "python sudoku.py diag && python sudoku.py time --split 0 --device cuda --stop fit"
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO/experiments"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ): $*"
bash -euo pipefail -c "$*"

#!/bin/bash
#SBATCH --job-name=ld-halt-fill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# After every halting pass has ended: export results/halting-full.json (dynamics/export_halting_full.py, CPU).
# Usage (from baselines/): sbatch --dependency=afterany:<pass job ids> dynamics/halting_fill.sbatch    (submit_halting_full.sh submits it)
set -euo pipefail
REPO=$(realpath ${SLURM_SUBMIT_DIR:?submit from baselines/}/..)
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID $(hostname)"
cd $REPO/baselines && ../.venv/bin/python dynamics/export_halting_full.py
echo "end $(date -u +%FT%T)"

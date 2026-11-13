#!/bin/bash
#SBATCH --job-name=trmp-score-helper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Fill test shards of a running TRMP score.py job from one more GPU (trmp/score_helper.py).
# Usage (from baselines/): sbatch -J trmp-help-s<k> trmp/score_helper.sbatch runs/trmp/sudoku-extreme-s<k> [--verify N]
set -euo pipefail
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
../.venv/bin/python trmp/score_helper.py --run "$@"
echo "end $(date -u +%FT%T)"

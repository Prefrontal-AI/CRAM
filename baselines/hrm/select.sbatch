#!/bin/bash
#SBATCH --job-name=select-hrm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Dev-based checkpoint selection for one finished HRM fit, then the test score of the selected checkpoint (hrm/select.py).
# Usage (from baselines/): sbatch -J select-hrm-<tag>-s<seed> --dependency=afterok:<fit job> hrm/select.sbatch <sudoku|mazeunique> [seed]
# HRM_EPOCHS as for the fit (20000 by default; 40000 for the sudoku-e40000 runs).
set -euo pipefail
BASELINES=$PWD
EXPERIMENTS=$BASELINES/../experiments
TASK=$1; SEED=${2:-0}
HRM_SRC=${HRM_SRC:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/HRM}
PY=${HRM_PY:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/.venv/bin/python}
EQR_MAZE=${EQR_MAZE:-${CRAM_DATA:-${SLURM_SUBMIT_DIR:-$PWD}/../data}/eqr/maze-30x30-unique-1k}
case $TASK in
  sudoku) DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data}/sudoku-extreme-1k-aug-1000 ;;
  mazeunique) DATA=$EQR_MAZE ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac
EPOCHS=${HRM_EPOCHS:-20000}
CFG=(data_path=$DATA epochs=$EPOCHS eval_interval=$((EPOCHS / 10)) lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0)
export WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1
unset DISABLE_COMPILE
TAG=$([[ $EPOCHS == 20000 ]] && echo "$TASK" || echo "$TASK-e$EPOCHS")
OUT=${HRM_OUT:-$BASELINES/runs/hrm/$TAG-s$SEED}
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor-select  # apart from the fit's cache (the fit is over; this is scoring)
echo "start $(date -u +%FT%T) job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
cd $HRM_SRC
$PY $BASELINES/hrm/select.py $OUT "${CFG[@]}" +seed=$SEED ${HRM_EXTRA:-}
echo "end $(date -u +%FT%T)"

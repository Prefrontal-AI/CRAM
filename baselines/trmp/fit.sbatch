#!/bin/bash
#SBATCH --job-name=time-trmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold TRMP fit on one H200 (bf16 autocast, torch.compile), then validation choice of the rollout noise
# and scoring of the selected EMA model on the test split, both outside the fit clock.
# Usage (from baselines/): sbatch -J time-trmp-<task> trmp/fit.sbatch <sudoku-extreme|maze-unique> [seed] [extra fit.py flags]
set -euo pipefail
BASELINES=$PWD
TASK=$1; SEED=${2:-0}; shift; shift || true
OUT=${TRMP_OUT:-$BASELINES/runs/trmp/$TASK-s$SEED}
mkdir -p $OUT
# A restart without a checkpoint is a new cold fit: drop its compile cache and partial logs.
[[ -f $OUT/fit.json || -f $OUT/ckpt.pt ]] || rm -rf $OUT/.inductor $OUT/log.jsonl $OUT/validation.jsonl
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor  # a fresh compile cache: the fit is cold
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${TRMP_PY:-$BASELINES/../.venv/bin/python}
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
$PY $BASELINES/trmp/fit.py --task $TASK --seed $SEED --out $OUT "$@"
$PY $BASELINES/trmp/score.py --run $OUT ${TRMP_SCORE_FLAGS:-}
echo "end $(date -u +%FT%T)"

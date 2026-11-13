#!/bin/bash
#SBATCH --job-name=time-trmplat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --constraint=gpu_h200_sxm
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold fit of the TRMP lattice model (Sudoku-Extreme, factor graph, 16,543 parameters; trmp/lattice_fit.py)
# on one H200, then the test search and the operation count, both outside the fit clock.
# Usage (from baselines/): CODE=<frozen copy> sbatch -J time-trmplat-s<seed> trmp/lattice_fit.sbatch <seed> [extra lattice_fit.py flags]
set -euo pipefail
BASELINES=$PWD
CODE=${CODE:-$BASELINES}
export V1_DATA=${V1_DATA:-$BASELINES/../data}
SEED=${1:-0}; shift || true
OUT=${TRMP_OUT:-$BASELINES/runs/trmp-lattice/sudoku-extreme-s$SEED}
mkdir -p $OUT
# A restart without a checkpoint is a new cold fit: drop its compile cache and partial logs.
[[ -f $OUT/fit.json || -f $OUT/ckpt.pt ]] || rm -rf $OUT/.inductor $OUT/log.jsonl $OUT/validation.jsonl
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor  # a fresh compile cache: the fit is cold
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${TRMP_PY:-$BASELINES/../.venv/bin/python}
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE"
$PY -u $CODE/trmp/lattice_fit.py --seed $SEED --out $OUT "$@"
$PY -u $CODE/trmp/lattice_score.py --run $OUT ${TRMP_SCORE_FLAGS:-}
[[ -n "${TRMP_SCORE_FLAGS:-}" ]] || $PY -u $CODE/flops/trmp_lattice_count.py --run $OUT
echo "end $(date -u +%FT%T)"

#!/bin/bash
#SBATCH --job-name=warm-trmplat
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
# TRMP lattice model, Sudoku-Extreme, one cold fit and 7 warm fits in one
# process on one H200 (trmp/lattice_fit_warm.py, which calls trmp/lattice_fit.py unchanged), then the test search and
# the operation count of the cold fit's model. The compile cache of the set starts empty and is shared by its fits.
# A requeued job starts the whole set again.
# Usage (from baselines/): CODE=<frozen copy> sbatch -J warm-trmplat-s<seed> trmp/lattice_fit_warm.sbatch <seed> [n_warm]
set -euo pipefail
BASELINES=$PWD
CODE=${CODE:-$BASELINES}
export V1_DATA=${V1_DATA:-$BASELINES/../data}
SEED=${1:-0}; NWARM=${2:-7}
OUT=${TRMP_OUT:-$BASELINES/runs/trmp-lattice/warm-s$SEED}
mkdir -p $OUT
[[ -f $OUT/repeats.json ]] || rm -rf $OUT/.inductor  # a fresh set starts with a cold fit: no compile cache
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${TRMP_PY:-$BASELINES/../.venv/bin/python}
echo "start $(date -u +%FT%T) job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE"
$PY -u $CODE/trmp/lattice_fit_warm.py $SEED $OUT $NWARM
$PY -u $CODE/trmp/lattice_score.py --run $OUT/cold
$PY -u $CODE/flops/trmp_lattice_count.py --run $OUT/cold
echo "end $(date -u +%FT%T)"

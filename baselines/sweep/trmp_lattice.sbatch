#!/bin/bash
#SBATCH --job-name=cs-trmplat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --constraint=gpu_h200_sxm
#SBATCH --requeue
#SBATCH --open-mode=append
# Compute sweep of the TRMP lattice model (trmp/lattice.py; sweep/trmp_lattice_sweep.py) at the (chains, budget) settings of its
# GRID on full-all rows 10,000-19,999, on one GPU. No fit time is measured here. Each setting writes its json last and is skipped when it
# exists, so a requeued job redoes only unfinished settings.
# Usage (from baselines/): CODE=<frozen copy of baselines/> sbatch -J cs-trmplat-s<k> sweep/trmp_lattice.sbatch <run folder> <out folder> [K:B ...]
set -euo pipefail
BASELINES=$PWD
CODE=${CODE:-$BASELINES}
export V1_DATA=${V1_DATA:-$BASELINES/../data}
RUN=$1; OUT=$2; shift 2
mkdir -p $OUT
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=${TRMP_PY:-$BASELINES/../.venv/bin/python}
SETTINGS=(); [[ $# -gt 0 ]] && SETTINGS=(--settings "$@")
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE run $RUN out $OUT settings ${*:-all}"
$PY -u $CODE/sweep/trmp_lattice_sweep.py --run $RUN --out $OUT "${SETTINGS[@]}"
echo "end $(date -u +%FT%T)"

#!/bin/bash
#SBATCH --job-name=score-trmplat
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
# Test search of one finished TRMP lattice fit (trmp/lattice_score.py) (412,786 puzzles, 64 chains, budget 1,000,
# check every step) in 4 shards of 103,197 puzzles, so that one search tail is paid per shard instead of 26; then the
# operation count. Resumes from finished shards.
# Usage (from baselines/): CODE=<frozen copy> sbatch -J score-trmplat-<name> trmp/lattice_score.sbatch <run folder>
set -euo pipefail
BASELINES=$PWD
CODE=${CODE:-$BASELINES}
export V1_DATA=${V1_DATA:-$BASELINES/../data}
RUN=$1
export TORCHINDUCTOR_CACHE_DIR=$RUN/.inductor-score OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${TRMP_PY:-$BASELINES/../.venv/bin/python}
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE run $RUN"
$PY -u $CODE/trmp/lattice_score.py --run $RUN --shard 103197
$PY -u $CODE/flops/trmp_lattice_count.py --run $RUN
echo "end $(date -u +%FT%T)"

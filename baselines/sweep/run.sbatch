#!/bin/bash
#SBATCH --job-name=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Compute sweep: one scoring pass of one trained baseline at a grid of its inference settings, with its authors' code, on one GPU
# (the authors used one H200). No fit time is measured here.
# Every pass writes <out>.json last and exits at once when it exists, so a requeued job redoes only unfinished passes.
# Usage (from baselines/): sbatch -J cs-<tag> sweep/run.sbatch <hrm|trm|eqr|trmp|trmp-caps|frm> <arguments of the method's script>
#   hrm|trm: <out prefix> <checkpoint> [--ref <score.right.npy>] <hydra overrides...>    (sweep/act_steps.py)
#   eqr: eqr_sweep.py arguments; trmp: trmp_sweep.py arguments; trmp-caps: trmp_caps.py arguments;    frm: sudoku <root> frm_sweep.py arguments
set -euo pipefail
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments; M=$1; shift
export WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset DISABLE_COMPILE
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) $M $*"
case $M in
  hrm)
    export TORCHINDUCTOR_CACHE_DIR=$(dirname $1)/.inductor-$(basename $1)
    cd ${HRM_SRC:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/HRM}
    ${HRM_PY:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/.venv/bin/python} $BASELINES/sweep/act_steps.py hrm "$@" ;;
  trm)
    export TORCHINDUCTOR_CACHE_DIR=$(dirname $1)/.inductor-$(basename $1) PYTHONPATH=$BASELINES/trm/trm_shim
    cd ${TRM_SRC:-$THIRD_PARTY/trm}
    ${TRM_PY:-$THIRD_PARTY/trm-env/bin/python} $BASELINES/sweep/act_steps.py trm "$@" 2>&1 | grep --line-buffered -v "^Processing batch" ;;
  eqr)
    export EQR_DATA=${EQR_DATA:-$EXPERIMENTS/runs/eqr/data}
    cd $BASELINES/eqr
    ${EQR_PY:-$THIRD_PARTY/eqr/.venv/bin/python} $BASELINES/sweep/eqr_sweep.py "$@" 2>&1 | grep --line-buffered -v "^\[Info\]\|Plotly" ;;
  trmp)
    $BASELINES/../.venv/bin/python $BASELINES/sweep/trmp_sweep.py "$@" ;;
  trmp-caps)
    $BASELINES/../.venv/bin/python $BASELINES/sweep/trmp_caps.py "$@" ;;
  frm)
    TASK=$1; ROOT=$(realpath $2); shift 2
    export PYTHONPATH=$ROOT/upstream PYTHONDONTWRITEBYTECODE=1 DIT_USE_COMPILE=0 FLM_DISABLE_FLASH=1 SPECFLOW_DISABLE_FLASH=1 \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
    cd $ROOT
    ${FRM_PY:-$THIRD_PARTY/frm-env/bin/python} $BASELINES/sweep/frm_sweep.py $TASK $ROOT "$@" ;;
  *) echo "unknown method $M"; exit 1 ;;
esac
echo "end $(date -u +%FT%T)"

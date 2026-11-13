#!/bin/bash
#SBATCH --job-name=ld
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
# Looped-model dynamics: run every pass listed in a pass file on one GPU (the authors used one H200), inference only.
# Each pass runs a trained baseline past its trained depth with its authors' code (act_dyn.py, eqr_dyn.py, trmp_dyn.py, frm_dyn.py).
# Each pass writes <out>.json last and exits at once when it exists, so a requeued job redoes only unfinished passes.
# Submit with dynamics/submit.sh (from baselines/), which freezes the code into runs/looped-dynamics/code/<stamp>/ (CODE).
# Usage: sbatch -J ld-<tag> --export=ALL,CODE=<frozen dir> dynamics/run.sbatch <pass file>
# Pass file: one pass per line, `<hrm|trm|eqr|trmp|frm> <arguments of the method's script>` (see submit.sh).
set -euo pipefail
set -f  # hydra overrides such as data_paths=[...] must not glob
CODE=${CODE:?frozen code folder}; PASSES=$(realpath $1)
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
REPO=$BASELINES/..; EXPERIMENTS=$REPO/experiments
export WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export V1_DATA=$REPO/data TRMP_MAZE_VALIDATION=$EXPERIMENTS/runs/TRMP-audit/generated-validation EQR_DATA=$EXPERIMENTS/runs/eqr/data
unset DISABLE_COMPILE
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE passes $PASSES"
one() {
  local M=$1; shift
  echo "pass $(date -u +%FT%T) $M $*"
  case $M in
    hrm)
      (export TORCHINDUCTOR_CACHE_DIR=$(dirname $1)/.inductor-$(basename $1)
       cd ${HRM_SRC:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/HRM}
       ${HRM_PY:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/.venv/bin/python} $CODE/baselines/dynamics/act_dyn.py hrm "$@") ;;
    trm)
      (export TORCHINDUCTOR_CACHE_DIR=$(dirname $1)/.inductor-$(basename $1) PYTHONPATH=$CODE/baselines/trm/trm_shim
       cd ${TRM_SRC:-${THIRD_PARTY:-$EXPERIMENTS/../third_party}/trm}
       ${TRM_PY:-$THIRD_PARTY/trm-env/bin/python} $CODE/baselines/dynamics/act_dyn.py trm "$@" 2>&1 | grep --line-buffered -v "^Processing batch") ;;
    eqr|eqr-nocompile)
      (if [[ $M == eqr-nocompile ]]; then export DISABLE_COMPILE=1; fi  # the Maze-Unique repository recipe trains and scores uncompiled
       cd $CODE/baselines/eqr
       ${EQR_PY:-$THIRD_PARTY/eqr/.venv/bin/python} $CODE/baselines/dynamics/eqr_dyn.py "$@" 2>&1 | grep --line-buffered -v "^\[Info\]\|Plotly") ;;
    trmp)
      (cd $CODE/baselines && $REPO/.venv/bin/python dynamics/trmp_dyn.py "$@") ;;
    frm)
      (TASK=$1; ROOT=$(realpath $2); shift 2
       export PYTHONPATH=$ROOT/upstream PYTHONDONTWRITEBYTECODE=1 DIT_USE_COMPILE=0 FLM_DISABLE_FLASH=1 SPECFLOW_DISABLE_FLASH=1 \
         OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
       cd $ROOT
       ${FRM_PY:-$THIRD_PARTY/frm-env/bin/python} $CODE/baselines/dynamics/frm_dyn.py $TASK $ROOT "$@") ;;
    *) echo "unknown method $M"; return 1 ;;
  esac
}
while read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  one $line < /dev/null
done < $PASSES
echo "end $(date -u +%FT%T)"

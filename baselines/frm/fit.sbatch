#!/bin/bash
#SBATCH --job-name=time-frm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Timed cold fits of FRM (Flow Reasoning Model) on one GPU (the paper used one H200): Stage A, then Stage B (each a
# fresh process, frm/fit.py around the adapter's train action), then the adapter's test scoring (its finish action).
# The adapter and the authors' code (Flow-Reasoning-Models at 6149a01) are in the run folder that frm/setup.sh built.
# Usage (from baselines/, after frm/setup.sh <task> frm-compiled):
#   sbatch -J time-frm-<task> frm/fit.sbatch <sudoku|mazeunique> [seed]
set -euo pipefail
TOP=$PWD; TASK=$1; SEED=${2:-0}  # TOP: baselines/, the submit folder
FRM_RUNS=${FRM_RUNS:-frm-compiled}  # runs subfolder: frm = no compilation (the reference), frm-compiled = whole-model torch.compile (the reported fit)
ROOT=$TOP/runs/$FRM_RUNS/$TASK
export FRM_COMPILE_MODEL=${FRM_COMPILE_MODEL:-$([[ $FRM_RUNS == frm ]] && echo 0 || echo 1)}
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor  # a fresh compile cache: the fit is cold
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${FRM_PY:-$THIRD_PARTY/frm-env/bin/python}
export PYTHONPATH=$ROOT/upstream PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DIT_USE_COMPILE=0 FLM_DISABLE_FLASH=1 SPECFLOW_DISABLE_FLASH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
case $TASK in
  sudoku)     ADAPTER=$ROOT/frm_reproduction.py; ARGS=(--seed $SEED) ;;
  mazeunique) ADAPTER=$ROOT/frm_maze.py;         ARGS=(--protocol paper1000 --arm det --seed $SEED) ;;
esac
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) root $ROOT whole-model-compile $FRM_COMPILE_MODEL"
cd $ROOT
for STAGE in A B; do $PY $TOP/frm/fit.py $ROOT $ADAPTER $STAGE -- "${ARGS[@]}"; done
$PY $ADAPTER --root $ROOT --action finish "${ARGS[@]}"
echo "end $(date -u +%FT%T)"

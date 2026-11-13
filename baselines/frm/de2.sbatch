#!/bin/bash
#SBATCH --job-name=DE2-frm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Data efficiency of FRM (Flow Reasoning Model) on Sudoku-Extreme: one run on k unique training puzzles (100 stored
# boards each: the puzzle and 99 augmented copies), seed s, on one GPU (the paper used one H200).
# Builds the run root (frm/de2_setup.py), then Stage A and Stage B (frm/fit.py, each a fresh process, whole-model
# torch.compile as runs/frm-compiled), then the adapter's finish: selection on the development puzzles and scoring on
# test10k (10,000 test puzzles). Needs runs/frm-compiled/sudoku (frm/setup.sh sudoku frm-compiled). Every stage skips
# finished work and resumes from its newest step_*.pt, so a requeued job repeats no finished stage.
# Usage (from baselines/, CODE a frozen copy made by frm/de2_freeze.sh):
#   sbatch -J DE2-frm-n<k>-s<s> frm/de2.sbatch <CODE> <k> <seed>
set -euo pipefail
TOP=$PWD; CODE=$1; K=$2; SEED=$3  # TOP: baselines/, the submit folder
ROOT=${FRM_DE2_ROOT:-$TOP/runs/frm/de2-v99/n$K-s$SEED}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${FRM_PY:-$THIRD_PARTY/frm-env/bin/python}
export FRM_COMPILE_MODEL=1 TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor  # a fresh compile cache per run: the fit is cold
export PYTHONPATH=$ROOT/upstream PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DIT_USE_COMPILE=0 FLM_DISABLE_FLASH=1 SPECFLOW_DISABLE_FLASH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
echo "start $(date -u +%FT%T) job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE root $ROOT k $K seed $SEED"
$PY $CODE/frm/de2_setup.py $ROOT $K $SEED
cd $ROOT
for STAGE in A B; do $PY $CODE/frm/fit.py $ROOT $ROOT/frm_de2.py $STAGE -- --seed $SEED; done
[[ -f $ROOT/runs/s$SEED/test10k.json ]] || $PY $ROOT/frm_de2.py --root $ROOT --action finish --seed $SEED
cat $ROOT/runs/s$SEED/test10k.json
echo "end $(date -u +%FT%T)"

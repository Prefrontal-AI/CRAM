#!/bin/bash
#SBATCH --job-name=time-hrm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold HRM fit on one H200, then scoring of the final model on the test split (outside the clock).
# Usage (from baselines/): sbatch -J time-hrm-<task> hrm/fit.sbatch <sudoku|mazeunique> [seed]
# Recipes: the HRM README's 8-GPU commands (global batch 768 kept on one GPU, same update count).
set -euo pipefail
BASELINES=$PWD
EXPERIMENTS=$BASELINES/../experiments  # datasets written by a baseline's builder: experiments/runs/<baseline>/
TASK=$1; SEED=${2:-0}
HRM_SRC=${HRM_SRC:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/HRM}  # sapientinc/HRM at commit ac15626, clean clone
PY=${HRM_PY:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/hrm/.venv/bin/python}
EQR_MAZE=${EQR_MAZE:-${CRAM_DATA:-${SLURM_SUBMIT_DIR:-$PWD}/../data}/eqr/maze-30x30-unique-1k}
case $TASK in
  sudoku) DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data}/sudoku-extreme-1k-aug-1000 ;;  # HRM's builder, TRM's 1,000 puzzles
  mazeunique) DATA=$EQR_MAZE ;;  # EqR's Maze-Unique split as published: HRM's format, no augmentation (as maze-30x30-hard-1k)
  *) echo "unknown task $TASK"; exit 1 ;;
esac
EPOCHS=${HRM_EPOCHS:-20000}  # README: 20,000; the released Sudoku checkpoint's config: 40,000 (eval_interval 4,000)
CFG=(data_path=$DATA epochs=$EPOCHS eval_interval=$((EPOCHS / 10)) lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0)
export WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1
unset DISABLE_COMPILE  # the authors' torch.compile stays on
TAG=$([[ $EPOCHS == 20000 ]] && echo "$TASK" || echo "$TASK-e$EPOCHS")
OUT=${HRM_OUT:-$BASELINES/runs/hrm/$TAG-s$SEED}
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor TRITON_CACHE_DIR=$OUT/.triton  # fresh compile caches: the fit is cold
echo "start $(date -u +%FT%T) job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
cd $HRM_SRC
$PY $BASELINES/hrm/fit.py $OUT "${CFG[@]}" +seed=$SEED +project_name=timing +run_name=$TASK-s$SEED ${HRM_EXTRA:-}
CKPT=$OUT/$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['final_checkpoint'])" $OUT/fit.json)
[[ -f $OUT/score.json ]] || $PY $BASELINES/hrm/score.py $OUT/score.json $CKPT "${CFG[@]}" +seed=$SEED ${HRM_EXTRA:-} 2>&1 | tail -30
echo "end $(date -u +%FT%T)"

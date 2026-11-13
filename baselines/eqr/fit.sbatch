#!/bin/bash
#SBATCH --job-name=time-eqr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold EqR fit on one H200, then the test scores of its final EMA weights (outside the clock).
# eqr/fit.py runs the training pieces of locuslab/EqR's pretrain.py (through eqr/eqr_train.py); eqr/eqr_score.py scores as EqR's evaluate.py.
# Usage (from baselines/): sbatch -J time-eqr-<task> eqr/fit.sbatch <sudoku|maze-repo|maze-paper> [seed]
# maze-repo is the EqR repository's Maze-Unique recipe; export DISABLE_COMPILE=1 to run it as the authors' sweep.sh does.
# maze-paper is the configuration of the released Maze-Unique checkpoint (presets: eqr/eqr_common.py). Data: eqr/eqr_data.py.
set -uo pipefail
TOP=$PWD; EXPERIMENTS=$TOP/../experiments; TASK=$1; SEED=${2:-0}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${EQR_PY:-${EQR_SRC:-$THIRD_PARTY/eqr}/.venv/bin/python}
export EQR_DATA=${EQR_DATA:-$EXPERIMENTS/runs/eqr/data} WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
RUN=$TOP/runs/eqr/$TASK-s$SEED
export EQR_INDUCTOR_CACHE=$RUN/.inductor  # a fresh compile cache: the fit is cold
if [ "$TASK" = sudoku ]; then
  DATA=$EQR_DATA/sudoku-s$SEED; EVAL=$EQR_DATA/sudoku-eval
  STAGES=("unseen 16 1 2048 0.01" "unseen 64 1 2048 0.01" "first10k 64 128 128 0.5")
else
  DATA=$EQR_DATA/maze-unique; EVAL=$EQR_DATA/maze-unique
  STAGES=("test 16 1 250 0.01" "test 64 1 250 0.01" "test 64 128 64 0.01")
fi
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) compile=${DISABLE_COMPILE:+off}"
cd $TOP/eqr
$PY $TOP/eqr/fit.py --run-dir $RUN --preset $TASK --data $DATA --seed $SEED --ckpt-minutes 30 2>&1 \
  | grep --line-buffered -v "^\[Info\]\|Plotly\|^\s*$"
[ -f $RUN/fit.json ] || { echo "fit failed"; exit 1; }
for st in "${STAGES[@]}"; do
  read -r SPLIT DEP BRE BAT NOISE <<< "$st"
  OUT=$RUN/score-$SPLIT-D$DEP-B$BRE
  [ -f "$OUT.json" ] && continue
  $PY eqr_score.py --ckpt $RUN/model.pt --data $EVAL --split $SPLIT --depth $DEP --breadth $BRE --batch $BAT \
    --noise $NOISE --seed $SEED --out $OUT 2>&1 | grep -v "^\[Info\]\|Plotly" | tail -5
done
echo "end $(date -u +%FT%T)"

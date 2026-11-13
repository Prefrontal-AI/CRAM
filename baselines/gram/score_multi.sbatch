#!/bin/bash
#SBATCH --job-name=score-gram-multi
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
# Score one GRAM N-Queens or graph-coloring fit (gram/score_multi.py run: 20 samples, steps 1-32), then merge it
# (gram/score_multi.py merge: GRAM's own evaluators and the shared checkers, CPU only; submitted as its own job with
# --gpus-per-node=0). A finished stage (its output exists) is skipped, so a requeue resumes.
# Submitted by gram/submit_score_multi.sh.
# Usage: sbatch --export=ALL,CODE=<frozen dir> gram/score_multi.sbatch <nqueens8|nqueens10|gc8|gc10> <seed> <run|merge>
set -euo pipefail
CODE=${CODE:?frozen code folder}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments
BENCH=$1; SEED=$2; STAGE=$3
STEPS=1,2,4,8,12,16,24,32
export TRM_DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data} PYTHONPATH=$EXPERIMENTS/runs/gram/pydeps:$EXPERIMENTS/../baselines/trm/trm_shim WANDB_MODE=disabled \
  OMP_NUM_THREADS=8 TQDM_DISABLE=1 CHECKERS=$CODE/checkers
unset DISABLE_COMPILE GRAM_SCORE_BATCHES
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
source $CODE/gram/configs.sh
declare -n cfg=GRAM_${BENCH^^}
T=${BENCH##*_}  # trm_nqueens8 and hrm_gc10 score as nqueens8 and gc10
case $T in nqueens*) TASK=nqueens; N=${T#nqueens}; D=nqueens-$N;; gc*) TASK=gc; N=${T#gc}; D=graph-coloring-$N;; esac
DATA=${GRAM_DATA:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/cache/gram-data}/$D
FIT=$BASELINES/runs/gram/$BENCH-s$SEED
[[ -f $FIT/fit.json ]] || { echo "no finished fit in $FIT"; exit 1; }
CKPT=$FIT/$(ls $FIT | grep "^step_[0-9]*$" | sort -t_ -k2 -n | tail -1)/checkpoint.pt
OUT=$BASELINES/runs/gram/score/$BENCH-s$SEED; mkdir -p $OUT
export TORCHINDUCTOR_CACHE_DIR=/tmp/gram-score-$SLURM_JOB_ID
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID $(hostname) $CKPT code $CODE"
cd ${THIRD_PARTY:-$EXPERIMENTS/../third_party}/gram
[[ $STAGE == run && ! -f $OUT/shard-0-20.json ]] && $PY $CODE/gram/score_multi.py run $OUT/shard-0-20.npz $CKPT 0 20 $STEPS "${cfg[@]}" seed=$SEED 2>&1 \
  | grep --line-buffered "sample\|Error\|checkpoint"
[[ $STAGE == merge && ! -f $OUT/score.json ]] && $PY $CODE/gram/score_multi.py merge $TASK $N $DATA $OUT/score.json $OUT/shard-0-20.npz 2>&1 | grep -v "^Processing\|warn"
echo "end $(date -u +%FT%T)"
true

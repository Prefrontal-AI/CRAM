#!/bin/bash
#SBATCH --job-name=score-gram
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Score samples [first, last) of one GRAM fit on its test split (gram/score.py run). Submitted by gram/submit_score.sh.
# A finished shard (its .json exists) is a no-op, so a requeue reruns only unfinished shards.
# Usage: sbatch --export=ALL,CODE=<frozen dir> gram/score.sbatch <sudoku|mazeunique> <seed> <first> <last>
set -euo pipefail
CODE=${CODE:?frozen code folder}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments
BENCH=$1; SEED=$2; FIRST=$3; LAST=$4
STEPS=1,2,4,8,12,16,24,32
export TRM_DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data} PYTHONPATH=$EXPERIMENTS/runs/gram/pydeps:$EXPERIMENTS/../baselines/trm/trm_shim WANDB_MODE=disabled \
  OMP_NUM_THREADS=8 TQDM_DISABLE=1
unset DISABLE_COMPILE GRAM_SCORE_BATCHES
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
source $CODE/gram/configs.sh
declare -n cfg=GRAM_${BENCH^^}
FIT=$BASELINES/runs/gram/$BENCH-s$SEED
[[ -f $FIT/fit.json ]] || { echo "no finished fit in $FIT"; exit 1; }
CKPT=$FIT/$(ls $FIT | grep "^step_[0-9]*$" | sort -t_ -k2 -n | tail -1)/checkpoint.pt
OUT=$BASELINES/runs/gram/score/$BENCH-s$SEED; mkdir -p $OUT
SHARD=$OUT/shard-$FIRST-$LAST
[[ -f $SHARD.json ]] && { echo "$SHARD done"; exit 0; }
export TORCHINDUCTOR_CACHE_DIR=/tmp/gram-score-$SLURM_JOB_ID
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID $(hostname) $CKPT samples $FIRST-$LAST code $CODE"
cd ${THIRD_PARTY:-$EXPERIMENTS/../third_party}/gram
$PY $CODE/gram/score.py run $SHARD.npz $CKPT $FIRST $LAST $STEPS "${cfg[@]}" seed=$SEED 2>&1 | grep --line-buffered "sample\|Error\|checkpoint"
echo "end $(date -u +%FT%T)"

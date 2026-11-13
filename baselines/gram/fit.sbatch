#!/bin/bash
#SBATCH --job-name=time-gram
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold GRAM fit (training only; scoring is separate) on one H200 (gram/fit.py, which runs the training pieces of
# ahn-ml/gram-release's pretrain.py). Every reported fit uses one GPU; the two-GPU path below is not used by any reported fit.
# Submit with gram/submit.sh, which freezes the code into runs/gram/code/<stamp>/ and passes it as $CODE.
# Usage: sbatch -J time-gram-<bench>-s<seed> --export=ALL,CODE=<frozen dir> gram/fit.sbatch <bench of gram/configs.sh, lower case> <seed>
set -euo pipefail
CODE=${CODE:?frozen code folder}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments
BENCH=$1; SEED=$2
export TRM_DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data} PYTHONPATH=$EXPERIMENTS/runs/gram/pydeps:$EXPERIMENTS/../baselines/trm/trm_shim WANDB_MODE=disabled \
  OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
unset DISABLE_COMPILE
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
GRAM_SRC=${GRAM_SRC:-${THIRD_PARTY:-$EXPERIMENTS/../third_party}/gram}  # ahn-ml/gram-release at f2e32b9
[[ $(git -C $GRAM_SRC rev-parse HEAD) == f2e32b9b11737890bfd4f39ead4cbea90835f91b && -z $(git -C $GRAM_SRC status --porcelain) ]]
source $CODE/gram/configs.sh
declare -n cfg=GRAM_${BENCH^^}
OUT=$BASELINES/runs/gram/${OUT_NAME:-$BENCH-s$SEED}  # OUT_NAME and EXTRA_OVERRIDES: smoke runs only
read -r -a EXTRA <<< "${EXTRA_OVERRIDES:-}"
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor  # a fresh compile cache: the fit is cold
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE"
cd $GRAM_SRC
NGPU=${SLURM_GPUS_ON_NODE:-1}
if (( NGPU > 1 )); then  # data parallel; the global batch is unchanged
  LAUNCH=($PY -m torch.distributed.run --standalone --nproc-per-node $NGPU)
else
  LAUNCH=($PY)
fi
"${LAUNCH[@]}" $CODE/gram/fit.py $OUT "${cfg[@]}" seed=$SEED "${EXTRA[@]}" +project_name=timing +run_name=$BENCH-s$SEED 2>&1 \
  | grep --line-buffered -v "^Processing batch"
echo "end $(date -u +%FT%T)"

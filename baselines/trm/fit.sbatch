#!/bin/bash
#SBATCH --job-name=time-trm
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
# One timed cold TRM fit on one H200 (trm/fit.py, which runs the training pieces of TinyRecursiveModels' pretrain.py), then
# scoring of the final EMA (exponential moving average) model on the test split (trm/trm_bench.py score).
# The datasets come from TRM's own builders and trm/trm_data.py; TRM_DATA names their folder.
# Usage (from baselines/): sbatch -J time-trm-<bench> trm/fit.sbatch <sudoku|mazeunique> [seed]
set -euo pipefail
TOP=$PWD
EXPERIMENTS=$TOP/../experiments
BENCH=$1; SEED=${2:-0}
export TRM_DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data} PYTHONPATH=$EXPERIMENTS/../baselines/trm/trm_shim WANDB_MODE=disabled \
  OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
unset DISABLE_COMPILE
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
TRM_SRC=${TRM_SRC:-${THIRD_PARTY:-$EXPERIMENTS/../third_party}/trm}  # TinyRecursiveModels at commit c011037
source $TOP/trm/trm_configs.sh
declare -n cfg=TRM_${BENCH^^}
OUT=$TOP/runs/trm/$BENCH-s$SEED
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor  # a fresh compile cache: the fit is cold
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
cd $TRM_SRC
$PY $TOP/trm/fit.py $OUT "${cfg[@]}" seed=$SEED +project_name=timing +run_name=$BENCH-s$SEED 2>&1 \
  | grep --line-buffered -v "^Processing batch"
CKPT=$OUT/$(ls $OUT | grep "^step_[0-9]*$" | sort -t_ -k2 -n | tail -1)
[[ -f $OUT/score.json ]] || $PY $TOP/trm/trm_bench.py score $OUT/score.json $CKPT "${cfg[@]}" seed=$SEED 2>&1 | tail -30
echo "end $(date -u +%FT%T)"

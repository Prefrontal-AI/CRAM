#!/bin/bash
#SBATCH --job-name=dataeff-trm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Data efficiency of TRM: its Sudoku-Extreme recipe (as baselines/trm/fit.sbatch runs it for the
# 1,000-puzzle rows) on N unique training puzzles, with the update count of the 1,000-puzzle run (65,100) kept.
# One timed cold fit on one H200, then the final EMA model scored on the official test split and on the 412,786
# unseen puzzles. Idempotent: each stage skips when its output exists; fit.py resumes from <run>/full.pt (rolling,
# every TIMING_CKPT_EVERY = 600 s, os.replace) after a requeue and flags the fit as resumed.
# The sets come from trm/trm_smalldata.py (1,001 rows per puzzle) or, with E3_SET=sudoku-smalldata-v99, from
# trm/trm_smalldata_v99.py (100 rows per puzzle, the sets of the paper's figure; set E3_OUT to experiments/runs/trm/de2-v99).
# The names E3_SMOKE, E3_OUT, E3_SET and the project name E3 are only labels.
# Usage (from baselines/): SRC=$(bash trm/data_efficiency_freeze.sh); sbatch -J dataeff-trm-n<N>-s<k> trm/data_efficiency.sbatch "$SRC" <N> <k>
set -euo pipefail
SRC=$1; N=$2; SEED=$3
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments  # built datasets and these runs live under experiments/runs/trm/
REPO=$(cd $BASELINES/.. && pwd)
export TRM_DATA=${TRM_DATA:-$EXPERIMENTS/runs/trm/data} PYTHONPATH=$SRC/baselines/trm/trm_shim WANDB_MODE=disabled OMP_NUM_THREADS=8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1 TIMING_CKPT_EVERY=${TIMING_CKPT_EVERY:-600}
unset DISABLE_COMPILE
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
# epochs and epochs per iteration that give 65,100 updates of 768 (the 1,000-puzzle run: 50,000 epochs, 10 x 6,510)
# N = 1, 3, 10, 30: N x EI is a multiple of 768, total_steps = EP x N / 768 = 65,100
case $N in
  1) EP=49996800; EI=499968 ;;  # 100 iterations x 651 updates
  3) EP=16665600; EI=166656 ;;  # 100 x 651
  10) EP=4999680; EI=499968 ;;  # 10 x 6,510
  30) EP=1666560; EI=166656 ;;  # 10 x 6,510
  100) EP=500000; EI=50000 ;;   # 10 iterations x 6,510 updates; pretrain.py total_steps 65,104
  300) EP=166670; EI=16667 ;;   # 10 x 6,510; total_steps 65,105
  3000) EP=16666; EI=8333 ;;    # 2 x 32,550; total_steps 65,101
  10000) EP=5000; EI=500 ;;     # 10 x 6,510; total_steps 65,104
  1000) EP=50000; EI=5000 ;;    # the 1,000-puzzle run's setting: 10 x 6,510; total_steps 65,104
  30000) EP=1667; EI=1667 ;;    # 65,117 updates (int(1667 * 30000 / 768)); no setting gives 65,100
  100000) EP=500; EI=5 ;;       # 100 x 651; total_steps 65,104
  50) EP=999940; EI=99994 ;;     # 10 x 6,510; total_steps 65,100
  70) EP=714240; EI=71424 ;;     # 10 x 6,510; total_steps 65,100
  150) EP=333320; EI=33332 ;;    # 10 x 6,510; total_steps 65,101
  200) EP=249990; EI=24999 ;;    # 10 x 6,510; total_steps 65,101
  500) EP=100000; EI=10000 ;;    # 10 x 6,510; total_steps 65,104
  700) EP=71430; EI=7143 ;;      # 10 x 6,510; total_steps 65,105
  *) echo "unknown N $N"; exit 2 ;;
esac
# Smoke run: E3_SMOKE=1 runs one iteration (EP = EI) into E3_OUT and skips the scoring.
[[ -n ${E3_SMOKE:-} ]] && EP=$EI
source $SRC/baselines/trm/trm_configs.sh
cfg=()
for o in "${TRM_SUDOKU[@]}"; do
  case $o in data_paths=*|epochs=*|eval_interval=*) ;; *) cfg+=("$o") ;; esac
done
DATA=$TRM_DATA/${E3_SET:-sudoku-smalldata}/s$SEED-n$N  # E3_SET=sudoku-smalldata-v99: the 100-row sets
[[ -f $DATA/subset.json ]] || { echo "missing $DATA"; exit 2; }
cfg+=(data_paths="[$DATA]" epochs=$EP eval_interval=$EI seed=$SEED)
OUT=${E3_OUT:-$EXPERIMENTS/runs/trm/smalldata}/n$N-s$SEED
mkdir -p $OUT
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor  # a fresh compile cache: the fit is cold
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) src $SRC N $N seed $SEED"
printf '%s\n' "${cfg[@]}" > $OUT/overrides.txt
cd ${TRM_SRC:-$THIRD_PARTY/trm}  # TinyRecursiveModels c011037, unmodified
$PY $SRC/baselines/trm/fit.py $OUT "${cfg[@]}" +project_name=E3 +run_name=n$N-s$SEED 2>&1 \
  | grep --line-buffered -v "^Processing batch"
[[ -n ${E3_SMOKE:-} ]] && { echo "smoke: no scoring; end $(date -u +%FT%T)"; exit 0; }
CKPT=$OUT/$(ls $OUT | grep "^step_[0-9]*$" | sort -t_ -k2 -n | tail -1)
[[ -f $OUT/score.json ]] || $PY $SRC/baselines/trm/trm_bench.py score $OUT/score.json $CKPT "${cfg[@]}" 2>&1 | tail -30
[[ -f $OUT/score-unseen.json ]] || V1_DATA=$REPO/data $PY $SRC/baselines/trm/unseen.py $OUT
echo "end $(date -u +%FT%T)"

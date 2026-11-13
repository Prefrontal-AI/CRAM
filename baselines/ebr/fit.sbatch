#!/bin/bash
#SBATCH --job-name=time-ebr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# One timed cold fit of EBR (Energy-Based Reasoner) on one GPU (the paper used one H200). Training only; scoring is
# separate (ebr/score.sbatch). Runs ebr/fit.py inside the clone of github.com/wwillchen/ebr at 8c5271c ($EBR_SRC).
# Submit with ebr/submit.sh, which freezes the code into runs/ebr/code/<stamp>/ and passes it as $CODE.
# <bench> selects the array EBR_<BENCH> of ebr/configs.sh: sudoku, mazeunique, nqueens8, nqueens10, gc8, gc10, sudokude1.
# Usage (from baselines/):
#   sbatch -J time-ebr-<bench>-s<seed> --export=ALL,CODE=<frozen dir> ebr/fit.sbatch <bench> <seed> [overrides...]
set -euo pipefail
CODE=${CODE:?frozen code folder}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments
BENCH=$1; SEED=$2; shift 2
export EBR_DATA=${EBR_DATA:-$EXPERIMENTS/runs/ebr/data} WANDB_MODE=disabled OMP_NUM_THREADS=8 TQDM_DISABLE=1
PY=${EBR_PY:-$EXPERIMENTS/runs/ebr/.venv/bin/python}  # a Python environment with EBR's requirements
EBR_SRC=${EBR_SRC:-$THIRD_PARTY/ebr}  # wwillchen/ebr at 8c5271c, unmodified (the next line checks it)
[[ $(git -C $EBR_SRC rev-parse HEAD) == 8c5271c05cf35f816e36db41ff324b3081175c64 && -z $(git -C $EBR_SRC status --porcelain) ]]
source $CODE/ebr/configs.sh
declare -n cfg=EBR_${BENCH^^}
OUT=${OUT:-$BASELINES/runs/ebr/$BENCH-s$SEED}
mkdir -p $OUT
# Fresh compile caches in the run folder, so the fit is cold (inductor and the fused attention's Triton kernels).
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor TRITON_CACHE_DIR=$OUT/.triton
[[ ${cfg[0]} == sudoku || ${cfg[0]} == *_fast ]] && export EBR_FUSED_ATTN_IMPL=triton  # as EBR's scripts/train.sh
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) code $CODE"
cd $EBR_SRC
$PY $CODE/ebr/fit.py $OUT "${cfg[@]}" seed=$SEED "$@"
echo "end $(date -u +%FT%T)"

#!/bin/bash
#SBATCH --job-name=score-ebr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Score one EBR checkpoint with EBR's own evaluate.py and its benchmark protocol (config/eval/<task>.yaml), unchanged,
# inside the clone of github.com/wwillchen/ebr at 8c5271c ($EBR_SRC). Maze-Unique uses EBR's Maze-Hard protocol.
# Sudoku: submit as an array, one shard per array task (16 shards, the NUM_SHARDS of EBR's README); ebr/report.py reads
# the 16 shards together. The other benchmarks: one job, no array.
# The output holds, per sample and segment end, the exact-match flag, the energy and the gradient norm; ebr/report.py
# turns it into scores and operation counts.
# Usage (from baselines/):
#   sbatch -J score-ebr-<bench>-s<seed> [--array=0-15] ebr/score.sbatch <bench> <checkpoint> <out_dir> [--key value...]
set -euo pipefail
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
EXPERIMENTS=$BASELINES/../experiments
BENCH=$1; CKPT=$2; OUT=$3; shift 3
D=${EBR_DATA:-$EXPERIMENTS/runs/ebr/data}
PY=${EBR_PY:-$EXPERIMENTS/runs/ebr/.venv/bin/python}
EBR_SRC=${EBR_SRC:-$THIRD_PARTY/ebr}
[[ $(git -C $EBR_SRC rev-parse HEAD) == 8c5271c05cf35f816e36db41ff324b3081175c64 && -z $(git -C $EBR_SRC status --porcelain) ]]
case $BENCH in
  sudoku) CFG=sudoku; DATA=sudoku-extreme-1k-aug-1000;;
  mazeunique) CFG=maze_hard; DATA=maze-unique;;
  nqueens8) CFG=nqueens_8; DATA=nqueens-8;;  nqueens10) CFG=nqueens_10; DATA=nqueens-10;;
  gc8) CFG=graph_coloring_8; DATA=graph-coloring-8;;  gc10) CFG=graph_coloring_10; DATA=graph-coloring-10;;
esac
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor TRITON_CACHE_DIR=$OUT/.triton TQDM_DISABLE=1
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-} $(hostname) $BENCH $CKPT"
cd $EBR_SRC
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  [[ -f $OUT/shard_${SLURM_ARRAY_TASK_ID}_of_16.json ]] && { echo "shard exists"; exit 0; }
  $PY evaluate.py --config config/eval/$CFG.yaml --checkpoint $CKPT --out $OUT --data_path $D/$DATA --shard $SLURM_ARRAY_TASK_ID --num_shards 16 "$@"
else
  $PY evaluate.py --config config/eval/$CFG.yaml --checkpoint $CKPT --out $OUT --data_path $D/$DATA "$@"
fi
echo "end $(date -u +%FT%T)"

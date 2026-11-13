#!/bin/bash
#SBATCH --job-name=merge-gram
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Merge the score shards of one GRAM fit (gram/score.py merge); Sudoku also on the 412,786 unseen puzzles (gram/unseen.py).
# Usage: sbatch --export=ALL,CODE=<frozen dir> gram/merge.sbatch <sudoku|mazeunique> <seed>
set -euo pipefail
CODE=${CODE:?frozen code folder}
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${TRM_PY:-$THIRD_PARTY/trm-env/bin/python}
export V1_DATA=${V1_DATA:-$BASELINES/../data} TRM_DATA=${TRM_DATA:-$BASELINES/../experiments/runs/trm/data}  # unseen.py runs from the frozen copy
OUT=$BASELINES/runs/gram/score/$1-s$2
$PY $CODE/gram/score.py merge $OUT/score.json $(ls $OUT/shard-*.npz)
[[ $1 == sudoku ]] && $PY $CODE/gram/unseen.py $OUT
echo "end $(date -u +%FT%T)"

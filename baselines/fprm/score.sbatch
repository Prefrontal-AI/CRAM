#!/bin/bash
#SBATCH --job-name=score-fprm-mazehard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Scores the authors' FPRM Maze-Hard checkpoint with their published evaluation setting (fprm/score.py).
# The second argument is the operation-count record of flops/fprm_count.py, which score.py reads to add the count per maze.
# Usage (from baselines/): CODE=<frozen copy of baselines/> sbatch fprm/score.sbatch
set -euo pipefail
BASELINES=$PWD
CODE=${CODE:-$BASELINES}
export FPRM_SRC=${FPRM_SRC:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/fprm/fprm}  # nilskiKonjIzDunava/fprm at commit b291252
OUT=$BASELINES/runs/fprm/mazehard-published
export TORCHINDUCTOR_CACHE_DIR=$OUT/.inductor OMP_NUM_THREADS=4
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
cd $FPRM_SRC
.venv/bin/python $CODE/fprm/score.py $OUT $BASELINES/runs/flops-full/fprm-mazehard.json
echo "end $(date -u +%FT%T)"

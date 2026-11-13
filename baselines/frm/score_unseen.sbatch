#!/bin/bash
#SBATCH --job-name=score-frm-unseen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Score the selected checkpoint of an FRM (Flow Reasoning Model) Sudoku campaign on some shards of the 412,786 unseen
# test puzzles, on one GPU (frm/score_unseen.py shard). After all 17 shards: python frm/score_unseen.py aggregate.
# A shard whose output exists is skipped, so a requeued job resumes at the next shard.
# Usage (from baselines/): sbatch -J score-frm-unseen-s<k>-g<i> frm/score_unseen.sbatch <campaign root> <seed> <shard>...
set -euo pipefail
TOP=$PWD; ROOT=$(realpath $1); SEED=$2; shift 2  # TOP: baselines/, the submit folder
BASELINES=${SLURM_SUBMIT_DIR:?submit from baselines/}; THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
PY=${FRM_PY:-$THIRD_PARTY/frm-env/bin/python}
export PYTHONPATH=$ROOT/upstream PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DIT_USE_COMPILE=0 FLM_DISABLE_FLASH=1 SPECFLOW_DISABLE_FLASH=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) root $ROOT seed $SEED shards $*"
cd $ROOT
for SH in "$@"; do
  [[ -f $ROOT/runs/s$SEED/unseen/$SH.json ]] || $PY $TOP/frm/score_unseen.py shard $ROOT --seed $SEED --shard $SH --batch 1024
done
echo "end $(date -u +%FT%T)"

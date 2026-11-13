#!/bin/bash
#SBATCH --job-name=ALT-sgd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Training-method study, the Maze-Unique arms of the paper: the free 3x3 kernel (18 parameters) trained with SGD
# (momentum 0.5, cosine decay). This script sets ALT_MAZE_KERNEL=free and ALT_MAZE_OPT=sgd-cosine, which select this
# setting in training_methods.py; the runs go to runs/ALT-sgd/ (the folder name is only a label).
#   sbatch --array=0-11 slurm/training_methods/sgd.sbatch tune     # array index = method; the grid in parallel on 3 GPUs, then select
#   sbatch --array=0-11 slurm/training_methods/sgd.sbatch final    # the choice on seeds 0-2, one per GPU, each then tested once
# Set the same two variables for the operation count (training_methods.py count --tasks maze, with and without --full)
# and for the exports. Finished runs and tests are skipped, so a requeued job resumes.
set -euo pipefail
export ALT_MAZE_KERNEL=free ALT_MAZE_OPT=sgd-cosine
STAGE=$1
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
PY=python
R=runs/ALT-sgd
METHODS=(pl mpl mpf dsm cd pcd chl eqp imp bptt1 bptt5 bptt)
i=${SLURM_ARRAY_TASK_ID:-0}
echo "host $(hostname) job ${SLURM_JOB_ID:-} task $i stage $STAGE start $(date -u)"
case $STAGE in
  tune)
    m=${METHODS[$i]}
    $PY training_methods.py plan --stage tune --tasks maze --methods $m > $R/plan-tune-maze-$m.txt
    g=0
    while read -r line; do
      CUDA_VISIBLE_DEVICES=$((g % 3)) $PY -u training_methods.py $line &
      g=$((g + 1))
    done < $R/plan-tune-maze-$m.txt
    wait
    $PY training_methods.py select --task maze --method $m ;;
  final)
    m=${METHODS[$i]}
    $PY training_methods.py plan --stage final --tasks maze --methods $m > $R/plan-final-maze-$m.txt
    g=0
    while read -r line; do
      run=$(echo "$line" | sed 's/.*--out //')
      (CUDA_VISIBLE_DEVICES=$g $PY -u training_methods.py $line && CUDA_VISIBLE_DEVICES=$g $PY -u training_methods.py test --run $run) &
      g=$((g + 1))
    done < $R/plan-final-maze-$m.txt
    wait ;;
esac
echo "end $(date -u)"

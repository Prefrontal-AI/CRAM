#!/bin/bash
#SBATCH --job-name=ALT-many
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Training-method study, curve runs on more seeds: the five methods of the figure on seeds 3-19 (training_curves.py).
# Write the plan files first: python training_curves.py plan (add --long for the long runs).
#   sbatch --array=0-7 slurm/training_methods/many.sbatch train   # array task i: the runs of runs/ALT/many/plan-train-g<i>.txt, one after
#                                              # another, each alone on this task's GPU, so that no two runs share a fit clock
#   sbatch --array=0-7 slurm/training_methods/many.sbatch probe   # score the saved checkpoints: PROBES_PER_GPU processes share the GPU
#   PLAN_DIR=runs/ALT/long sbatch --array=0-7 slurm/training_methods/many.sbatch train   # the long runs of training_curves.LONG (same for probe)
# A finished run is skipped, an unfinished one resumes from its checkpoint, and the probe resumes from probes.json.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
i=$SLURM_ARRAY_TASK_ID
d=${PLAN_DIR:-runs/ALT/many}
case "$1" in
  train)
    while read -r line; do
      python -u training_curves.py $line
    done < $d/plan-train-g$i.txt
    ;;
  probe)
    p=${PROBES_PER_GPU:-3}
    n=$((SLURM_ARRAY_TASK_COUNT * p))
    for j in $(seq 0 $((p - 1))); do
      runs=$(awk -v k=$((i * p + j)) -v n=$n '(NR - 1) % n == k' $d/plan-probe.txt)
      [ -n "$runs" ] && python -u training_curves.py probe --runs $runs &
    done
    wait
    ;;
esac

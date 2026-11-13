#!/bin/bash
#SBATCH --job-name=ALT-tune
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
#SBATCH --array=0-19
# Training-method study, tuning stage. Array index i = task (0-9 sudoku, 10-19 maze) x method (METHODS[i % 10]).
# The grid of the (task, method) runs in parallel on three GPUs (at most 18 runs, seed 0, the full schedule), then
# `training_methods.py select` writes the choice. A finished run is skipped, so a requeued job resumes.
# The paper's Maze-Unique arms are tuned by sgd.sbatch, which sets ALT_MAZE_KERNEL=free and ALT_MAZE_OPT=sgd-cosine.
# Without these variables, array indices 10-19 train an earlier maze setting (separable kernel, Adam) that the paper
# does not report. For the methods bptt1, bptt5 and bptt, set ALT_METHODS (10 names, indexed by i % 10).
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
METHODS=(${ALT_METHODS:-pl mpl mpf dsm cd pcd chl eqp imp tbptt})  # ALT_METHODS: a subset, indexed by the array (i % 10)
i=$SLURM_ARRAY_TASK_ID
task=$([ $i -lt 10 ] && echo sudoku || echo maze)
method=${METHODS[$((i % 10))]}
python training_methods.py plan --stage tune --tasks $task --methods $method > runs/ALT/plan-tune-$task-$method.txt
g=0
while read -r line; do
  CUDA_VISIBLE_DEVICES=$((g % 3)) python -u training_methods.py $line &
  g=$((g + 1))
done < runs/ALT/plan-tune-$task-$method.txt
wait
python training_methods.py select --task $task --method $method

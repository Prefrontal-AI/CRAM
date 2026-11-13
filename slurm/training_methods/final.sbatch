#!/bin/bash
#SBATCH --job-name=ALT-final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-19
# Training-method study, final stage. Array index i = task (0-9 sudoku, 10-19 maze) x method (METHODS[i % 10]), as
# in tune.sbatch. The setting that tune.sbatch selected runs on seeds 0-2, one seed per GPU, so that no two runs
# share a fit clock. Each run is followed by its search tuning and its test (training_methods.py test).
# Finished runs and tests are skipped, so a requeued job resumes.
# The paper's Maze-Unique arms come from sgd.sbatch, not from array indices 10-19 of this script (see tune.sbatch).
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
METHODS=(${ALT_METHODS:-pl mpl mpf dsm cd pcd chl eqp imp tbptt})  # ALT_METHODS: a subset, indexed by the array (i % 10)
i=$SLURM_ARRAY_TASK_ID
task=$([ $i -lt 10 ] && echo sudoku || echo maze)
method=${METHODS[$((i % 10))]}
python training_methods.py plan --stage final --tasks $task --methods $method > runs/ALT/plan-final-$task-$method.txt
g=0
while read -r line; do
  run=$(echo "$line" | sed 's/.*--out //')
  (CUDA_VISIBLE_DEVICES=$g python -u training_methods.py $line && \
   CUDA_VISIBLE_DEVICES=$g python -u training_methods.py test --run $run) &
  g=$((g + 1))
done < runs/ALT/plan-final-$task-$method.txt
wait

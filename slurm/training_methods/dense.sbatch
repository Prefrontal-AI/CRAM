#!/bin/bash
#SBATCH --job-name=ALT-dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-9
# Training-method study, curve runs: the selected setting of five methods (pl, mpf, cd, bptt1, bptt) on seeds 0-2,
# trained again with a probe after every update up to the 64th and 8 log-spaced probes per doubling after that.
# Array index i = task (0-4 sudoku, 5-9 maze) x method (METHODS[i % 5]); one run per GPU.
# The probes restore the random-number state, so the training is the same as in final.sbatch. No test.
# Input: runs/ALT/plan-dense-<task>-<method>.txt, one `training_methods.py train` command line per seed. No script
# writes these files. Write them from the final stage's lines (training_methods.py plan --stage final) with
# `--probe-per-octave 8 --probe-every-until 64` added and `--out <root>/dense/<task>-<method>-s<seed>`.
# `runs/ALT` is the run folder of this study; the name is only a label.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
METHODS=(pl mpf cd bptt1 bptt)
i=$SLURM_ARRAY_TASK_ID
task=$([ $i -lt 5 ] && echo sudoku || echo maze)
method=${METHODS[$((i % 5))]}
g=0
while read -r line; do
  (CUDA_VISIBLE_DEVICES=$g python -u training_methods.py $line) &
  g=$((g + 1))
done < runs/ALT/plan-dense-$task-$method.txt
wait

#!/bin/bash
#SBATCH --job-name=ALT-longD2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-3
# Training-method study, longer schedules than long_d.sbatch on Sudoku: bptt with 480,000 updates (learning rate 2
# or 6) and eqp with 240,000 updates (learning rate 6 or 60), each with the knob and L1 strength that tune.sbatch
# selected. Array index i = row i of ARMS (method, learning rate, updates); seeds 0-2 in parallel, one per GPU.
# A finished run is skipped and an unfinished one resumes from its rolling checkpoint, so a requeued job resumes.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
ARMS=("bptt 2 480000" "bptt 6 480000" "eqp 6 240000" "eqp 60 240000")
read -r m lr updates <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
choice=$(python3 -c "import json; c = json.load(open('runs/ALT/tune/sudoku-$m/choice.json'))['choice']; print(c['knob'], c['l1'])")
read -r knob l1 <<< "$choice"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python -u training_methods.py train --task sudoku --method $m --lr $lr --knob $knob \
    --l1 $l1 --updates $updates --seed $s --out runs/ALT/longD2/sudoku-$m-lr$lr-s$s &
done
wait

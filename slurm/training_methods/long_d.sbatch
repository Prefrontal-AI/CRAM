#!/bin/bash
#SBATCH --job-name=ALT-longD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-11
# Training-method study, long schedules: the Sudoku methods that run relaxations during training (class D in
# training_methods.py) with 60,000 cosine-decay updates instead of 2,000. Array index i = method (METHODS[i / 3]) x
# learning rate (LRS[i % 3]); seeds 0-2 run in parallel, one per GPU, so that no two runs share a fit clock.
# Each method keeps the method parameter (knob) and L1 strength that tune.sbatch selected. A finished run is skipped
# and an unfinished one resumes from its rolling checkpoint (training_methods.train), so a requeued job resumes.
# `longD` in the run folder name is only a label.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
METHODS=(bptt bptt5 eqp imp)
LRS=(6 20 60)
i=$SLURM_ARRAY_TASK_ID
m=${METHODS[$((i / 3))]}
lr=${LRS[$((i % 3))]}
choice=$(python3 -c "import json; c = json.load(open('runs/ALT/tune/sudoku-$m/choice.json'))['choice']; print(c['knob'], c['l1'])")
read -r knob l1 <<< "$choice"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python -u training_methods.py train --task sudoku --method $m --lr $lr --knob $knob \
    --l1 $l1 --updates 60000 --seed $s --out runs/ALT/longD/sudoku-$m-lr$lr-s$s &
done
wait

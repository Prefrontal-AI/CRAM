#!/bin/bash
#SBATCH --job-name=ALT-longD3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Training-method study, the longest schedule on Sudoku: bptt with 3,840,000 cosine-decay updates at learning rate 2,
# with its selected knob (10 relaxation steps) and L1 strength (0.1), and a fit-time cap of 36,000 s.
# Seeds 0-2 run in parallel, one per GPU. An unfinished run resumes from its rolling checkpoint
# (training_methods.train), so a requeued job resumes.
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:?submit from the repository root}; cd "$REPO/experiments"; export PATH="$REPO/.venv/bin:$PATH"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s python -u training_methods.py train --task sudoku --method bptt --lr 2 --knob 10 \
    --l1 0.1 --updates 3840000 --cap-s 36000 --seed $s --out runs/ALT/longD3/sudoku-bptt-lr2-s$s &
done
wait

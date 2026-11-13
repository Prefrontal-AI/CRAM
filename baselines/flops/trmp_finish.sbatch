#!/bin/bash
#SBATCH --job-name=flop-trmp-finish
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# TRMP per-puzzle matmul-only flop totals from the saved test step counts (trmp_count.py), then results/flop-counts.json (report.py).
# Usage (from baselines/): sbatch --dependency=afterany:<the six TRMP fit jobs> flops/trmp_finish.sbatch
# Safe to rerun: a run without score/test.json contributes no total.
set -euo pipefail
echo "start $(date -u +%FT%T) job $SLURM_JOB_ID $(hostname)"
../.venv/bin/python flops/trmp_count.py --out runs/flops/trmp.json
../.venv/bin/python flops/report.py
echo "end $(date -u +%FT%T)"

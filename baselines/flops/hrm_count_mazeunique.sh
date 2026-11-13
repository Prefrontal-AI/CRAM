#!/bin/bash
set -euo pipefail
# HRM Maze-Unique inference flop (flops/hrm_count.py with the authors' code in $THIRD_PARTY/hrm/HRM): the seed-0 checkpoint
# that hrm/select.py selected on development data (step 26,040), with the fit overrides of records/hrm-mazeunique-s0.json.
# Run on one GPU: bash baselines/flops/hrm_count_mazeunique.sh
BASELINES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
T=$BASELINES
cd $THIRD_PARTY/hrm/HRM
DISABLE_COMPILE=1 $THIRD_PARTY/hrm/.venv/bin/python $T/flops/hrm_count.py $T/runs/flops/hrm-mazeunique.json $T/runs/hrm/mazeunique-s0/step_26040 \
  data_path=${CRAM_DATA:-$BASELINES/../data}/eqr/maze-30x30-unique-1k \
  epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +seed=0

#!/bin/bash
# Data efficiency of EBR on Sudoku-Extreme: submit one timed fit per training-set size k and seed s, and after each fit
# one scoring job of its final checkpoint.
# Training set: experiments/runs/ebr/data/sudoku-smalldata-v99/s<s>-n<k>, k unique puzzles with 100 stored boards each
# (the puzzle and 99 augmented copies), built by `DE_SET=sudoku-smalldata-v99 python ebr/smalldata.py build`.
# Test set: experiments/runs/ebr/data/sudoku-test10k (10,000 test puzzles, built by ebr/test10k.py), scored by EBR's
# evaluate.py with its Sudoku config unchanged, in one shard. Summarize each score with ebr/de1_report.py.
# The script freezes the code it runs into runs/ebr/code/<UTC stamp>/ (or reuses CODE=<a frozen copy>).
# DEP=<job id> makes every fit wait for that job (for example the job that builds the data).
# Usage (from baselines/; the script only calls sbatch):
#   bash ebr/de2_submit.sh "<k...>" "<seed...>" [hydra overrides...]
set -euo pipefail
KS=$1; SEEDS=$2; shift 2
if [[ -z ${CODE:-} ]]; then  # CODE=<a frozen copy> reuses it
  STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$PWD/runs/ebr/code/$STAMP
  mkdir -p $CODE/ebr && cp ebr/fit.py ebr/configs.sh ebr/smalldata.py $CODE/ebr/ && cp clock.py $CODE/ && chmod -R a-w $CODE
fi
DATA=$PWD/../experiments/runs/ebr/data/sudoku-smalldata-v99
TEST10K=$PWD/../experiments/runs/ebr/data/sudoku-test10k
for k in $KS; do for s in $SEEDS; do
  [[ -n ${DEP:-} || -f $DATA/s$s-n$k/subset.json ]] || { echo "missing $DATA/s$s-n$k"; exit 2; }  # with DEP, the job DEP builds the data
  OUT=$PWD/runs/ebr/de2-v99/n$k-s$s; NAME=DE2-ebr-n$k-s$s
  fit=$(sbatch --parsable -J $NAME --time=06:00:00 ${DEP:+--dependency=afterok:$DEP} --export=ALL,CODE=$CODE,OUT=$OUT,EBR_DE1_DATA=$DATA/s$s-n$k ebr/fit.sbatch sudokude1 $s "$@")
  echo "$fit $NAME $(date -u +%FT%TZ) code $CODE out $OUT" | tee -a runs/ebr/jobs.txt
  sc=$(sbatch --parsable -J score10k-$NAME --time=01:30:00 --dependency=afterok:$fit \
    ebr/score.sbatch sudoku $OUT/run/checkpoints/step_60000_sudoku-s$s.pth $OUT/score10k --data_path $TEST10K --num_shards 1 --shard 0)
  echo "$sc score10k-$NAME after $fit $(date -u +%FT%TZ)" | tee -a runs/ebr/jobs.txt
done; done

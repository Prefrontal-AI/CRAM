#!/bin/bash
# Submit the timed EBR fits of one benchmark (ebr/fit.sbatch), one job per seed (default seeds 0 1 2).
# The script first copies ebr/fit.py, ebr/configs.sh and clock.py into runs/ebr/code/<UTC stamp>/, read-only, so that
# later edits cannot change a queued job. It appends each job id to runs/ebr/jobs.txt. TIME sets the wall-time limit.
# Usage (from baselines/; the script only calls sbatch):
#   bash ebr/submit.sh <sudoku|mazeunique|nqueens8|nqueens10|gc8|gc10> [seed...]
set -euo pipefail
BENCH=$1; shift; SEEDS=${*:-0 1 2}
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$PWD/runs/ebr/code/$STAMP
mkdir -p $CODE/ebr && cp ebr/fit.py ebr/configs.sh $CODE/ebr/ && cp clock.py $CODE/ && chmod -R a-w $CODE
TIME=${TIME:-12:00:00}
for s in $SEEDS; do
  id=$(sbatch --parsable -J time-ebr-$BENCH-s$s --time=$TIME --export=ALL,CODE=$CODE ebr/fit.sbatch $BENCH $s)
  echo "$id time-ebr-$BENCH-s$s $(date -u +%FT%TZ) code $CODE" | tee -a runs/ebr/jobs.txt
done

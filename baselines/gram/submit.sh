#!/bin/bash
# Freeze baselines/gram + clock.py into runs/gram/code/<stamp>/ and submit the timed fits there.
# Usage (from baselines/; only calls sbatch): bash gram/submit.sh <sudoku|mazeunique|[trm_|hrm_]nqueens8|nqueens10|gc8|gc10> [seed...]
set -euo pipefail
BENCH=$1; shift; SEEDS=${*:-0 1 2}
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$PWD/runs/gram/code/$STAMP
mkdir -p $CODE/gram && cp gram/fit.py gram/configs.sh $CODE/gram/ && cp clock.py $CODE/ && chmod -R a-w $CODE
case $BENCH in  # one H200 each; wall-time limits
  sudoku) RES=(--time=08:00:00);; nqueens8|gc8) RES=(--time=04:00:00);; gc10) RES=(--time=06:00:00);;
  nqueens10) RES=(--time=12:00:00);;
  trm_nqueens8|trm_gc8|hrm_nqueens8|hrm_gc8) RES=(--time=08:00:00);; trm_*|hrm_*) RES=(--time=20:00:00);;
  *) RES=(--time=36:00:00);;
esac
for s in $SEEDS; do
  id=$(sbatch --parsable -J time-gram-$BENCH-s$s "${RES[@]}" --export=ALL,CODE=$CODE gram/fit.sbatch $BENCH $s)
  echo "$id time-gram-$BENCH-s$s $(date -u +%FT%TZ) code $CODE" | tee -a runs/gram/jobs.txt
done

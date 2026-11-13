#!/bin/bash
# Freeze gram/ + clock.py + trm/unseen.py and submit the scoring of one GRAM fit: <shards> GPU jobs of the 20 samples,
# then the merge. With a fit job id, every job waits for that fit to end successfully.
# Usage (from baselines/; only calls sbatch): bash gram/submit_score.sh <sudoku|mazeunique> <seed> [fit job id]
set -euo pipefail
BENCH=$1; SEED=$2; DEP=${3:-}
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$PWD/runs/gram/code/score-$STAMP-$BENCH-s$SEED
mkdir -p $CODE/gram $CODE/trm && cp gram/score.py gram/configs.sh gram/unseen.py $CODE/gram/ && cp trm/unseen.py $CODE/trm/ \
  && cp clock.py $CODE/ && chmod -R a-w $CODE
OPT=(--export=ALL,CODE=$CODE); [[ -n $DEP ]] && OPT+=(--dependency=afterok:$DEP)
if [[ $BENCH == sudoku ]]; then PER=2; else PER=20; fi  # Sudoku: 10 shards of 2 samples (~1 h each); maze: one job
ids=()
for ((f = 0; f < 20; f += PER)); do
  id=$(sbatch --parsable -J score-gram-$BENCH-s$SEED-$f "${OPT[@]}" gram/score.sbatch $BENCH $SEED $f $((f + PER)))
  ids+=($id); echo "$id score-gram-$BENCH-s$SEED samples $f-$((f + PER)) $(date -u +%FT%TZ) code $CODE" | tee -a runs/gram/jobs.txt
done
id=$(sbatch --parsable -J merge-gram-$BENCH-s$SEED --export=ALL,CODE=$CODE --dependency=afterok:$(IFS=:; echo "${ids[*]}") gram/merge.sbatch $BENCH $SEED)
echo "$id merge-gram-$BENCH-s$SEED $(date -u +%FT%TZ) code $CODE" | tee -a runs/gram/jobs.txt

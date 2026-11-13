#!/bin/bash
# Freeze gram/score.py, gram/score_multi.py, configs.sh and the shared checkers (cram/nqueens_data.py,
# graph_coloring_data.py) and submit the scoring and merge of GRAM N-Queens / graph-coloring fits, one job per seed.
# With fit job ids, each job waits for its fit to end successfully.
# Usage (from baselines/; only calls sbatch): bash gram/submit_score_multi.sh <[trm_|hrm_]nqueens8|nqueens10|gc8|gc10> <seed>[:<fit job id>]...
set -euo pipefail
BENCH=$1; shift
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$PWD/runs/gram/code/score-multi-$STAMP-$BENCH
mkdir -p $CODE/gram $CODE/checkers && cp gram/score.py gram/score_multi.py gram/configs.sh $CODE/gram/ \
  && cp ../cram/nqueens_data.py ../cram/graph_coloring_data.py $CODE/checkers/ && chmod -R a-w $CODE
for arg in "$@"; do
  SEED=${arg%%:*}; DEP=; [[ $arg == *:* ]] && DEP=--dependency=afterok:${arg#*:}
  id=$(sbatch --parsable -J score-gram-$BENCH-s$SEED --export=ALL,CODE=$CODE $DEP gram/score_multi.sbatch $BENCH $SEED run)
  echo "$id score-gram-$BENCH-s$SEED $(date -u +%FT%TZ) code $CODE" | tee -a runs/gram/jobs.txt
  id=$(sbatch --parsable -J merge-gram-$BENCH-s$SEED --gpus-per-node=0 --cpus-per-task=4 --mem=64G --export=ALL,CODE=$CODE \
    --dependency=afterok:$id gram/score_multi.sbatch $BENCH $SEED merge)
  echo "$id merge-gram-$BENCH-s$SEED $(date -u +%FT%TZ) code $CODE" | tee -a runs/gram/jobs.txt
done

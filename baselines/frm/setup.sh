#!/bin/bash
# Build a fresh run folder ("campaign folder") for FRM (Flow Reasoning Model) on one task.
# The script copies these items from the folder of an earlier reproduction campaign: the adapter (frm_reproduction.py
# for Sudoku-Extreme, frm_maze.py for Maze-Unique), plan.json, source-files.json, environment.txt, the upstream source
# (github.com/helblazer811/Flow-Reasoning-Models at 6149a01, Python package flow_map_reasoning) and the prepared data.
# It does not copy training outputs. The adapter verifies the hash of every copied file before it trains.
# FRM_CAMPAIGNS names the folder that holds the earlier campaigns (default: $THIRD_PARTY/frm-reproduction/outputs);
# the two campaign folders below must exist in it. That folder is not part of this repository, and this repository
# has no script that builds it from baselines/fetch.sh and scripts/prepare_data.py. baselines/frm/adapters/ holds a
# copy of the two adapters for reading.
# Usage (from baselines/): frm/setup.sh <sudoku|mazeunique> [runs subfolder, default frm]
#   frm/setup.sh sudoku frm-compiled      # the folder that frm/fit.sbatch uses by default
set -euo pipefail
BASELINES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
FRM_CAMPAIGNS=${FRM_CAMPAIGNS:-$THIRD_PARTY/frm-reproduction/outputs}
case $1 in
  sudoku)     SRC=$FRM_CAMPAIGNS/frm-reproduction-20260918/20260918T111411Z; ADAPTER=frm_reproduction.py ;;
  mazeunique) SRC=$FRM_CAMPAIGNS/frm-maze-20260924/20260924T094000Z;      ADAPTER=frm_maze.py ;;
  *) echo "unknown task $1"; exit 2 ;;
esac
DST=$PWD/runs/${2:-frm}/$1
[[ -d $DST/upstream ]] && { echo "$DST exists"; exit 0; }
mkdir -p $DST
cp -a $SRC/$ADAPTER $SRC/plan.json $SRC/source-files.json $SRC/environment.txt $SRC/upstream $SRC/data $SRC/cache $DST/
echo "$SRC" > $DST/SOURCE
echo "built $DST from $SRC"

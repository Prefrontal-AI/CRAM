#!/bin/bash
# Build the datasets of EBR (Energy-Based Reasoner) with its own builders (github.com/wwillchen/ebr at 8c5271c,
# unmodified): Sudoku-Extreme, N-Queens 8 and 10, graph coloring 8 and 10. Build Maze-Unique with ebr/maze_unique.py.
# Output: experiments/runs/ebr/data/ (or $EBR_DATA), with a SHA256SUMS file of every array.
# Needs the clone $THIRD_PARTY/ebr (baselines/fetch.sh ebr) and a Python environment with EBR's requirements
# ($EBR_PY, default experiments/runs/ebr/.venv/bin/python). CPU only, about one hour. Usage (from any folder):
#   bash baselines/ebr/build_data.sh
set -euo pipefail
BASELINES=$(cd "$(dirname "$0")/.." && pwd)
EXPERIMENTS=$BASELINES/../experiments
SRC=${EBR_SRC:-${THIRD_PARTY:-$BASELINES/../third_party}/ebr}; PY=${EBR_PY:-$EXPERIMENTS/runs/ebr/.venv/bin/python}; D=${EBR_DATA:-$EXPERIMENTS/runs/ebr/data}
cd $SRC
[[ $(git rev-parse HEAD) == 8c5271c05cf35f816e36db41ff324b3081175c64 && -z $(git status --porcelain) ]]
[[ -d $D/sudoku-extreme-1k-aug-1000 ]] || $PY -m dataset.build_sudoku_dataset --output-dir $D/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
[[ -d $D/nqueens-8 ]] || $PY -m dataset.build_nqueens_dataset --n 8 --output-dir $D/nqueens-8
[[ -d $D/nqueens-10 ]] || $PY -m dataset.build_nqueens_dataset --n 10 --output-dir $D/nqueens-10
[[ -d $D/graph-coloring-8 ]] || $PY -m dataset.build_graph_coloring_dataset --nodes 8 --output-dir $D/graph-coloring-8
[[ -d $D/graph-coloring-10 ]] || $PY -m dataset.build_graph_coloring_dataset --nodes 10 --output-dir $D/graph-coloring-10
[[ -d $D/maze-unique ]] || PYTHONPATH=$SRC $PY $BASELINES/ebr/maze_unique.py $D/maze-unique
(cd $D && find . -name '*.npy' -o -name '*.json' | sort | xargs sha256sum) > $D/SHA256SUMS
echo done

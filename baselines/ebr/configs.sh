# EBR's recipe and data for each benchmark (github.com/wwillchen/ebr at 8c5271c). Source this file with EBR_DATA set
# (ebr/fit.sbatch does). It defines one array EBR_<BENCH> per benchmark: the recipe name under EBR's config/train/,
# then Hydra overrides. Data: experiments/runs/ebr/data, built by ebr/build_data.sh.
D=${EBR_DATA:?}
EBR_SUDOKU=(sudoku dataset.data_path=$D/sudoku-extreme-1k-aug-1000)
# Maze-Unique: EBR publishes no recipe. Its Maze-Hard recipe maze_hard_fast (30x30, the README's Maze-Hard command),
# unchanged, on EqR's Maze-Unique split (as GRAM's Maze-Unique row uses TRM's maze recipe).
EBR_MAZEUNIQUE=(maze_hard_fast dataset.data_path=$D/maze-unique dataset.name=Maze-30x30-unique-1k)
EBR_NQUEENS8=(nqueens_8 dataset.data_path=$D/nqueens-8)
EBR_NQUEENS10=(nqueens_10 dataset.data_path=$D/nqueens-10)
EBR_GC8=(graph_coloring_8 dataset.data_path=$D/graph-coloring-8)
EBR_GC10=(graph_coloring_10 dataset.data_path=$D/graph-coloring-10)
# Data efficiency: the recipe sudoku, unchanged, on a subset of the training puzzles. EBR_DE1_DATA names the subset
# folder (built by ebr/smalldata.py; ebr/de2_submit.sh sets it).
EBR_SUDOKUDE1=(sudoku dataset.data_path=${EBR_DE1_DATA:-$D/unset})

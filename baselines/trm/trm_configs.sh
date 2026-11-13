# TRM's hydra overrides for each benchmark: the official README commands (TinyRecursiveModels c011037), with data paths
# under $TRM_DATA (default experiments/runs/trm/data). Source this file; it defines the TRM_<BENCH> arrays.
D=${TRM_DATA:?}
TRM_COMMON=(arch=trm evaluators="[]" lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 ema=True)
# Sudoku-Extreme, TRM-MLP (the paper's 87.4%): README "pretrain_mlp_t_sudoku", one GPU.
TRM_SUDOKU=("${TRM_COMMON[@]}" data_paths="[$D/sudoku-extreme-1k-aug-1000]" epochs=50000 eval_interval=5000 arch.mlp_t=True arch.pos_encodings=none arch.H_cycles=3 arch.L_cycles=6)
# Maze-Unique: TRM's Maze-Hard recipe (TRM-Att, README "pretrain_att_maze30x30", global batch 768) unchanged on EqR's split.
TRM_MAZEUNIQUE=("${TRM_COMMON[@]}" data_paths="[$D/maze-unique-aug8]" epochs=50000 eval_interval=5000 arch.H_cycles=3 arch.L_cycles=4)

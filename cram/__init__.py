"""CRAM: pairwise energies trained from solved examples and searched by single-cell moves.

Per task, the modules hold a data loader, the energy with its training loss, and the search.
The docstring of each energy module states its energy function and the meaning of its symbols and dimension names;
the docstring of each data module states the source, the token encoding, the splits and the augmentation.

  Sudoku-Extreme   cram_sudoku_search (data loader, search protocol, dense move kernel)
                   cram_sudoku_general (the model of the paper: a free cost table per cell pair, and its search)
                   cram_sudoku (same-digit pair energy, digit relabeling), cram_sudoku_sparse (its sparse search;
                   the hand-set energy)
  Maze-Unique      maze_unique (data, loss, search), cram_maze (the model of the paper and the separable form)
  Maze-Hard        maze_learned (data, window helpers), cram_maze_hard_graph (the model of the paper),
                   cram_maze_hard_simple (hidden layer and its adjoint), latent_settle_triton (its fused kernels),
                   maze_h3 (gated answer links, answer search, diagnostics), maze_h3p (a diagnostic)
  N-Queens         nqueens_data (data, rule checker, metrics), cram_nqueens (energy, fit, search)
  Graph coloring   graph_coloring_data (data, rule checker), cram_graph_coloring (energy, fit, search, scoring)

`opcount` rebuilds per-puzzle operation counts from saved records, `table` writes the table records, and `paths` names
the data and run folders.
"""
